"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

Code for fine-tuning pre-trained models for annotated MAIJU data. The fine-tuning is run
in two stages:
  1. Two randomly initialized fully-connected layers are added after the pre-trained model
     in order to turn the output of the Transformer encoder blocks into categorical
     probabilities for each movement category. These layers are fine-tuned first while
     the weights of the encoder and the Transformer blocks are freezed.
  2. The entire network is fine-tuned.

This implementation contains simulated (= randomly generated) MAIJU data.

"""

import numpy as np 
import os
import time
import sys

from importlib.machinery import SourceFileLoader
from py_conf_file_into_text import convert_py_conf_file_to_text
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from copy import deepcopy

from torch import cuda, no_grad, save, load, unsqueeze, squeeze
from torch.utils.data import DataLoader
from torch.nn import Softmax


# Read the configuration file
if len(sys.argv) > 2:
    sys.exit('\nUsage: \n1) python finetune_pretrained_maiju_model.py \nOR \n2) python finetune_pretrained_maiju_model.py <configuration_file>')
if len(sys.argv) == 2:
    conf = SourceFileLoader('', sys.argv[1]).load_module()
    conf_file_name = sys.argv[1]
else:
    try:
        import conf_finetune_pretrained_maiju_model as conf
        conf_file_name = 'conf_finetune_pretrained_maiju_model.py'
    except ModuleNotFoundError:
        sys.exit('\nUsage: \n1) python finetune_pretrained_maiju_model.py \nOR \n2) python finetune_pretrained_maiju_model.py <configuration_file>\n\n' \
                 'By using the first option, you need to have a configuration file named "conf_finetune_pretrained_maiju_model.py" in the same ' \
                 'directory as "finetune_pretrained_maiju_model.py"')

# Import our models
encoder_model = getattr(__import__('maiju_nn_model', fromlist=[conf.encoder_model]), conf.encoder_model)
timeseries_model = getattr(__import__('maiju_nn_model', fromlist=[conf.timeseries_model]), conf.timeseries_model)

# Import our dataset for our data loader
dataset = getattr(__import__('maiju_data_loader', fromlist=[conf.dataset_name]), conf.dataset_name)

# Import our loss function
maiju_loss = getattr(__import__('torch.nn', fromlist=[conf.loss_name]), conf.loss_name)

# Import our optimization algorithm
optimization_algorithm = getattr(__import__('torch.optim', fromlist=[conf.optimization_algorithm]), conf.optimization_algorithm)


def frame_sig(X, winlen, hop):
    Nframes = int(np.floor(((X.shape[0] - winlen)/hop) + 1))
    numchans = X.shape[1]
    X_framed = np.zeros([Nframes, numchans, winlen], dtype=np.float32) # [Nframes, Nchans, winlen]
    for i in range(0, Nframes):
        start = i * hop
        stop = start + winlen
        X_framed[i,:,:] = np.transpose(X[start:stop,:])

    return X_framed







if __name__ == "__main__":
    
    # We make sure that we are able to write the logging file
    textfile_path, textfile_name = os.path.split(f'{conf.result_dir}/{conf.name_of_log_textfile}')
    if not os.path.exists(textfile_path):
        if textfile_path != '':
            os.makedirs(textfile_path)
    file = open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'w')
    file.close()
    
    # Read the text in the configuration file and add it to the logging file
    if conf.print_conf_contents:
        conf_file_lines = convert_py_conf_file_to_text(conf_file_name)
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write(f'The configuration settings in the file {conf_file_name}:\n\n')
            for line in conf_file_lines:
                f.write(f'{line}\n')
            f.write('\n########################################################################################\n\n\n\n')
    
    # Use CUDA if it is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write(f'Process on {device}\n\n')
    
    # Read data
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write('Reading & processing raw data...\n')
    
    if conf.print_training_updates_to_command_line:
        print(f'Initializing data... (see log file {conf.result_dir}/{conf.name_of_log_textfile} for further information)\n')
    
    Data = []
    for iBaby in range(conf.num_randomly_generated_babydata):
        babyData = {}
        
        # We generate random signals to simulate having MAIJU recordings
        num_samples = np.random.randint(50000, high=300000)
        num_channels = 12
        x = np.linspace(0, num_samples, num_samples)
        
        acc_data = [] # Randomly generated accelerometer data
        gyro_data = [] # Randomly generated gyroscope data
        for i in range(num_channels):
            for data_list in [acc_data, gyro_data]:
                if np.random.rand() < 0.5:
                    data_list.append(np.random.rand() * np.sin(x) + np.random.normal(scale=0.1, size=len(x)))
                else:
                    data_list.append(np.random.rand() * np.cos(x) + np.random.normal(scale=0.1, size=len(x)))
        
        x_r = np.concatenate((acc_data, gyro_data), axis=0).T
        
        # We frame the signals
        x_r = frame_sig(x_r, conf.window_len, conf.hop_len)
        
        # We randomly generate a mask to simulate sections of the data in which unwanted phenomena occurred, such as
        # if the baby was out of screen or if the baby was being carried by a caregiver.
        # 1 = frame is masked, 0 = frame is not masked
        mask = (np.random.rand(len(x_r)) < 0.1).astype(int)
        
        babyData['X'] = x_r
        babyData['Mask'] = mask

        Data.append(babyData)
    
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write('Done!\n\n')

    Nbabies = len(Data)
    num_babies_per_fold = int(np.floor(Nbabies // conf.num_folds))

    if conf.randomize_order_kfolds:
        perm = np.random.RandomState(seed=42).permutation(Nbabies)
    else:
        perm = np.arange(Nbabies)
    
    fold_test_accuracies = []
    fold_conf_mats = []
    for fold in range(conf.num_folds):
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write(f'Training model for fold {fold+1}\n\n')
        
        if fold == conf.num_folds - 1:
            test_inds = perm[fold*num_babies_per_fold:]
        else:
            test_inds = perm[fold*num_babies_per_fold:(fold+1)*num_babies_per_fold]

        D_train = []
        D_test = []
    
        # Select test subject(s)
        for i in range(len(Data)):
            if i in test_inds:
                D_test.append(Data[i])
            else:
                D_train.append(Data[i])
        
        # Initialize the data loaders
        if conf.train_model:
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('Initializing training set...\n')
            training_set = dataset(D_train, train_val_test='train', **conf.params_train_dataset)
            train_data_loader = DataLoader(training_set, **conf.params_train)
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('Done!\n')
                f.write('Initializing validation set...\n')
            validation_set = dataset(D_train, train_val_test='validation', **conf.params_validation_dataset)
            validation_data_loader = DataLoader(validation_set, **conf.params_train)
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('Done!\n')
        if conf.test_model:
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('Initializing test set...\n')
            test_set = dataset(D_test, train_val_test='test', **conf.params_test_dataset)
            test_data_loader = DataLoader(test_set, **conf.params_test)
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('Done!\n\n')
        
        for finetuning_stage in [1, 2]:
            
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write(f'######### Fine-tuning stage {finetuning_stage} #########\n\n')
        
            # Initialize our models, pass the models to the available device
            Encoder_model = encoder_model(**conf.encoder_model_params).to(device)
            Timeseries_model = timeseries_model(**conf.timeseries_model_params).to(device)
            
            # Give the parameters of our models to an optimizer
            model_parameters = list(Encoder_model.parameters()) + list(Timeseries_model.parameters())
            optimizer = optimization_algorithm(params=model_parameters, **conf.optimization_algorithm_params)
            
            # Get our learning rate for later use
            learning_rate = optimizer.param_groups[0]['lr']
            
            # Give the optimizer to the learning rate scheduler
            if conf.use_lr_scheduler:
                if finetuning_stage == 1:
                    scheduler = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler_stage_1]), conf.lr_scheduler_stage_1)
                    lr_scheduler = scheduler(optimizer, **conf.lr_scheduler_params_stage_1)
                else:
                    scheduler_1 = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler_stage_2_part_1]), conf.lr_scheduler_stage_2_part_1)
                    scheduler_2 = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler_stage_2_part_2]), conf.lr_scheduler_stage_2_part_2)
                    scheduler_part_1 = scheduler_1(optimizer, **conf.lr_scheduler_params_stage_2_part_1)
                    scheduler_part_2 = scheduler_2(optimizer, **conf.lr_scheduler_params_stage_2_part_2)
        
            # Initialize our loss function as a class
            loss_function = maiju_loss(**conf.loss_params)
        
            # Variables for early stopping
            highest_validation_accuracy = -1e10
            best_validation_epoch = 0
            patience_counter = 0
            
            
            # Load our model weights
            try:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('Loading models from file...\n')
                    if finetuning_stage == 1:
                        f.write(f'Loading model {conf.best_model_encoder_name_pretrained}...\n')
                        f.write(f'Loading model {conf.best_model_timeseries_name_pretrained}...\n')
                        Encoder_model.load_state_dict(load(f'{conf.best_model_encoder_name_pretrained}', map_location=device))
                        
                        # The pre-trained model does not have classification layers at the end of the network so
                        # we need to filter those out from the parameter dict before we can load the weights
                        timeseries_model_state_dict = load(f'{conf.best_model_timeseries_name_pretrained}', map_location=device)
                        filtered_dict = {k: v for k, v in timeseries_model_state_dict.items() if 'classification' not in k}
                        model_updated_state_dict = Timeseries_model.state_dict()
                        model_updated_state_dict.update(filtered_dict)
                        Timeseries_model.load_state_dict(model_updated_state_dict)
                    else:
                        f.write(f'Loading model {conf.result_dir}/fold_{fold+1}/{conf.best_model_encoder_name_finetuned}...\n')
                        f.write(f'Loading model {conf.result_dir}/fold_{fold+1}/{conf.best_model_timeseries_name_finetuned}...\n')
                        Encoder_model.load_state_dict(load(f'{conf.result_dir}/fold_{fold+1}/{conf.best_model_encoder_name_finetuned}',
                                                           map_location=device))
                        Timeseries_model.load_state_dict(load(f'{conf.result_dir}/fold_{fold+1}/{conf.best_model_timeseries_name_finetuned}',
                                                           map_location=device))
                
                best_model_encoder = deepcopy(Encoder_model.state_dict())
                best_model_timeseries = deepcopy(Timeseries_model.state_dict())
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('Done!\n\n')
            except (FileNotFoundError, RuntimeError):
                best_model_encoder = None
                best_model_timeseries = None
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('An error occurred while loading the files! Training without loaded model weights...\n\n')
            
            if finetuning_stage == 1:
                # We only fine-tune the classification layers in the first fine-tuning stage
                for name, param in Encoder_model.named_parameters():
                    param.requires_grad = False
                for name, param in Timeseries_model.named_parameters():
                    if not 'classification' in name:
                        param.requires_grad = False
            
            # Flag for indicating if max epochs are reached
            max_epochs_reached = 1
            
            # Start training our model
            if conf.train_model:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('Starting training...\n')
                
                if conf.print_training_updates_to_command_line:
                    print(f'Starting training (fold {fold+1}, fine-tuning stage {finetuning_stage})...\n')
                
                for epoch in range(1, conf.max_epochs + 1):
                    
                    start_time = time.time()
            
                    # Lists containing the losses of each epoch
                    epoch_loss_training = []
                    epoch_loss_validation = []
                    epoch_true_Y_training = np.array([])
                    epoch_pred_Y_training = np.array([])
                    epoch_true_Y_validation = np.array([])
                    epoch_pred_Y_validation = np.array([])
            
                    # Indicate that we are in training mode, so e.g. dropout will function
                    Encoder_model.train()
                    Timeseries_model.train()
                    
                    # Loop through every minibatch of our training data
                    for train_data in train_data_loader:
                        
                        # Get the minibatches
                        X, Y, data_mask = [element.to(device) for element in train_data]
                        
                        # Zero the gradient of the optimizer
                        optimizer.zero_grad()
                        
                        # We go through each sequence one at a time
                        loss_batch = []
                        for sequence_index in range(len(X)):
                            if data_mask[sequence_index].sum() > 0.95 * X.size()[1]:
                                continue
                            
                            X_input = X[sequence_index]
                        
                            # Pass our data through the encoder
                            sensor_encoding = Encoder_model(X_input.float())
                            
                            # Pass our encodings to the timeseries model
                            if 'transformer' in Timeseries_model.__class__.__name__:
                                padding_mask = unsqueeze(data_mask[sequence_index].bool(), dim=0)
                                sensor_encoding = unsqueeze(sensor_encoding, dim=0)
                                Y_pred, _, _, _ = Timeseries_model(sensor_encoding, src_key_padding_mask=padding_mask)
                                Y_pred = squeeze(Y_pred)
                            else:
                                Y_pred, _ = Timeseries_model(sensor_encoding)
                            
                            # We only take the unmasked frames into account
                            masked_frames = data_mask[sequence_index].bool()
                            Y_pred_unmasked = Y_pred[~masked_frames, :]
                            Y_unmasked = Y[sequence_index, ~masked_frames, :].max(dim=1)[1]
                            
                            # Compute the weighted loss and add it to the losses of the minibatch
                            loss = loss_function(input=Y_pred_unmasked, target=Y_unmasked)
                            loss_batch.append(loss)
                            
                            # Compute the prediction accuracy of the model (unweighted mean F1 score)
                            smax = Softmax(dim=1)
                            Y_pred_unmasked_smax_np = smax(Y_pred_unmasked).detach().cpu().numpy()
                            predictions = np.argmax(Y_pred_unmasked_smax_np, axis=1)
                            epoch_true_Y_training = np.concatenate((epoch_true_Y_training, Y_unmasked.detach().cpu().numpy()))
                            epoch_pred_Y_training = np.concatenate((epoch_pred_Y_training, predictions))
                        
                        if len(loss_batch) > 0:
                            total_loss = sum(loss_batch) / len(loss_batch)
                            
                            # Perform the backward pass
                            total_loss.backward()
                            
                            # Update the weights
                            optimizer.step()
                            
                            # Add the loss to the total loss of the batch
                            epoch_loss_training.append(total_loss.item())
                    
                    
                    # Indicate that we are in evaluation mode, so e.g. dropout will not function
                    Encoder_model.eval()
                    Timeseries_model.eval()
            
                    # Make PyTorch not calculate the gradients, so everything will be much faster.
                    with no_grad():
                        
                        # Loop through every batch of our validation data and perform a similar process
                        # as for the training data
                        for validation_data in validation_data_loader:
                            X, Y, data_mask = [element.to(device) for element in validation_data]
                            loss_batch = []
                            for sequence_index in range(len(X)):
                                if data_mask[sequence_index].sum() > 0.95 * X.size()[1]:
                                    continue
                                X_input = X[sequence_index]
                                sensor_encoding = Encoder_model(X_input.float())
                                if 'transformer' in Timeseries_model.__class__.__name__:
                                    padding_mask = unsqueeze(data_mask[sequence_index].bool(), dim=0)
                                    sensor_encoding = unsqueeze(sensor_encoding, dim=0)
                                    Y_pred, _, _, _ = Timeseries_model(sensor_encoding, src_key_padding_mask=padding_mask)
                                    Y_pred = squeeze(Y_pred)
                                else:
                                    Y_pred, _ = Timeseries_model(sensor_encoding)
                                masked_frames = data_mask[sequence_index].bool()
                                Y_pred_unmasked = Y_pred[~masked_frames, :]
                                Y_unmasked = Y[sequence_index, ~masked_frames, :].max(dim=1)[1]
                                loss = loss_function(input=Y_pred_unmasked, target=Y_unmasked)
                                loss_batch.append(loss)
                                smax = Softmax(dim=1)
                                Y_pred_unmasked_smax_np = smax(Y_pred_unmasked).detach().cpu().numpy()
                                predictions = np.argmax(Y_pred_unmasked_smax_np, axis=1)
                                epoch_true_Y_validation = np.concatenate((epoch_true_Y_validation, Y_unmasked.detach().cpu().numpy()))
                                epoch_pred_Y_validation = np.concatenate((epoch_pred_Y_validation, predictions))
                            if len(loss_batch) > 0:
                                total_loss = sum(loss_batch) / len(loss_batch)
                                epoch_loss_validation.append(total_loss.item())
                    
                    # Calculate mean losses
                    epoch_loss_training = np.array(epoch_loss_training).mean()
                    epoch_loss_validation = np.array(epoch_loss_validation).mean()
                    if conf.train_criterion == 'f1':
                        epoch_accuracy_training = f1_score(epoch_true_Y_training, epoch_pred_Y_training, average='macro')
                        epoch_accuracy_validation = f1_score(epoch_true_Y_validation, epoch_pred_Y_validation, average='macro')
                    elif conf.train_criterion == 'recall':
                        epoch_accuracy_training = recall_score(epoch_true_Y_training, epoch_pred_Y_training, average='macro')
                        epoch_accuracy_validation = recall_score(epoch_true_Y_validation, epoch_pred_Y_validation, average='macro')
                    else:
                        sys.exit(f'The training criterion {conf.train_criterion} not implemented!')
                    
                    # Check early stopping conditions
                    if epoch_accuracy_validation > highest_validation_accuracy:
                        highest_validation_accuracy = epoch_accuracy_validation
                        patience_counter = 0
                        best_model_encoder = deepcopy(Encoder_model.state_dict())
                        best_model_timeseries = deepcopy(Timeseries_model.state_dict())
                        best_validation_epoch = epoch
                        if conf.save_best_model:
                            # We first make sure that we are able to write the files
                            save_names = [f'{conf.result_dir}/fold_{fold+1}/{conf.best_model_encoder_name_finetuned}',
                                          f'{conf.result_dir}/fold_{fold+1}/{conf.best_model_timeseries_name_finetuned}']
                            for model_save_name in save_names:
                                model_path, model_filename = os.path.split(model_save_name)
                                if not os.path.exists(model_path):
                                    if model_path != '':
                                        os.makedirs(model_path)
                            
                            save(best_model_encoder, f'{conf.result_dir}/fold_{fold+1}/{conf.best_model_encoder_name_finetuned}')
                            save(best_model_timeseries, f'{conf.result_dir}/fold_{fold+1}/{conf.best_model_timeseries_name_finetuned}')
                    else:
                        patience_counter += 1
                    
                    end_time = time.time()
                    epoch_time = end_time - start_time
                    
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write(f'Epoch: {epoch:04d} | Mean training loss: {epoch_loss_training:6.4f} | '
                          f'Mean validation loss: {epoch_loss_validation:6.4f} | '
                          f'Mean training accuracy: {epoch_accuracy_training:6.4f} | '
                          f'Mean validation accuracy: {epoch_accuracy_validation:6.4f} (highest: {highest_validation_accuracy:6.4f}) | '
                          f'Duration: {epoch_time:4.2f} seconds\n')
                    
                    # We check that do we need to update the learning rate based on the validation loss
                    if conf.use_lr_scheduler:
                        if finetuning_stage == 1:
                            if conf.lr_scheduler_stage_1 == 'ReduceLROnPlateau':
                                lr_scheduler.step(epoch_accuracy_validation)
                            else:
                                lr_scheduler.step()
                        else:
                            if epoch < conf.lr_scheduler_stage_2_part_1_epochs + 1:
                                lr_scheduler = scheduler_part_1
                                if conf.lr_scheduler_stage_2_part_1 == 'ReduceLROnPlateau':
                                    lr_scheduler.step(epoch_accuracy_validation)
                                else:
                                    lr_scheduler.step()
                            else:
                                lr_scheduler = scheduler_part_2
                                if conf.lr_scheduler_stage_2_part_2 == 'ReduceLROnPlateau':
                                    lr_scheduler.step(epoch_accuracy_validation)
                                else:
                                    lr_scheduler.step()
                        current_learning_rate = optimizer.param_groups[0]['lr']
                        if current_learning_rate != learning_rate:
                            learning_rate = current_learning_rate
                            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                                f.write(f'Updated learning rate after epoch {epoch} based on learning rate scheduler, now lr={learning_rate}\n')
                    
                    if conf.print_training_updates_to_command_line:
                        print(f'Finished training epoch {epoch:04d} (see log file {conf.result_dir}/{conf.name_of_log_textfile} for further information)')
                    
                    # If patience counter is fulfilled, stop the training
                    if patience_counter >= conf.patience:
                        max_epochs_reached = 0
                        break
                
                if max_epochs_reached:
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('\nMax number of epochs reached, stopping training\n\n')
                else:
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('\nExiting due to early stopping\n\n')
                
                if best_model_encoder is None:
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('\nNo best model. The criteria for the lowest acceptable validation accuracy not satisfied!\n\n')
                    sys.exit('No best model, exiting...')
                else:
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write(f'\nBest epoch {best_validation_epoch} with validation accuracy {highest_validation_accuracy}\n\n')
        
                if conf.print_training_updates_to_command_line:
                    print(f'\nFinished training process (see log file {conf.result_dir}/{conf.name_of_log_textfile} for further information)\n')
        
        if conf.test_model:
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('\nStarting testing... => ')
            
            if conf.print_training_updates_to_command_line:
                print(f'Testing trained model (fold {fold+1})...\n')
            
            # Load the best version of the model
            try:
                Encoder_model.load_state_dict(load(f'{conf.result_dir}/fold_{fold+1}/{conf.best_model_encoder_name_finetuned}',
                                                   map_location=device))
                Timeseries_model.load_state_dict(load(f'{conf.result_dir}/fold_{fold+1}/{conf.best_model_timeseries_name_finetuned}',
                                                      map_location=device))
            except (FileNotFoundError, RuntimeError):
                Encoder_model.load_state_dict(best_model_encoder)
                Timeseries_model.load_state_dict(best_model_timeseries)
                    
            testing_loss = []
            epoch_true_Y_testing = np.array([])
            epoch_pred_Y_testing = np.array([])
            Encoder_model.eval()
            Timeseries_model.eval()
            with no_grad():
                for test_data in test_data_loader:
                    X, Y, data_mask = [element.to(device) for element in test_data]
                    loss_batch = []
                    for sequence_index in range(len(X)):
                        if data_mask[sequence_index].sum() > 0.95 * X.size()[1]:
                            continue
                        X_input = X[sequence_index]
                        sensor_encoding = Encoder_model(X_input.float())
                        if 'transformer' in Timeseries_model.__class__.__name__:
                            padding_mask = unsqueeze(data_mask[sequence_index].bool(), dim=0)
                            sensor_encoding = unsqueeze(sensor_encoding, dim=0)
                            Y_pred, _, _, _ = Timeseries_model(sensor_encoding, src_key_padding_mask=padding_mask)
                            Y_pred = squeeze(Y_pred)
                        else:
                            Y_pred, _ = Timeseries_model(sensor_encoding)
                        masked_frames = data_mask[sequence_index].bool()
                        Y_pred_unmasked = Y_pred[~masked_frames, :]
                        Y_unmasked = Y[sequence_index, ~masked_frames, :].max(dim=1)[1]
                        loss = loss_function(input=Y_pred_unmasked, target=Y_unmasked)
                        loss_batch.append(loss)
                        smax = Softmax(dim=1)
                        Y_pred_unmasked_smax_np = smax(Y_pred_unmasked).detach().cpu().numpy()
                        predictions = np.argmax(Y_pred_unmasked_smax_np, axis=1)
                        epoch_true_Y_testing = np.concatenate((epoch_true_Y_testing, Y_unmasked.detach().cpu().numpy()))
                        epoch_pred_Y_testing = np.concatenate((epoch_pred_Y_testing, predictions))
                    if len(loss_batch) > 0:
                        total_loss = sum(loss_batch) / len(loss_batch)
                        testing_loss.append(total_loss.item())
                testing_loss = np.array(testing_loss).mean()
                
                conf_mat = confusion_matrix(epoch_true_Y_testing, epoch_pred_Y_testing, labels=np.arange(9))
                confmat_path, confmat_filename = os.path.split(f'{conf.result_dir}/fold_{fold+1}/fold_{fold+1}_confmat.npy')
                if not os.path.exists(confmat_path):
                    if confmat_path != '':
                        os.makedirs(confmat_path)
                np.save(f'{conf.result_dir}/fold_{fold+1}/fold_{fold+1}_confmat.npy', conf_mat)
                np.save(f'{conf.result_dir}/fold_{fold+1}/fold_{fold+1}_true_Y.npy', epoch_true_Y_testing)
                np.save(f'{conf.result_dir}/fold_{fold+1}/fold_{fold+1}_pred_Y.npy', epoch_pred_Y_testing)
                
                if conf.train_criterion == 'f1':
                    testing_accuracy = f1_score(epoch_true_Y_testing, epoch_pred_Y_testing, average='macro')
                elif conf.train_criterion == 'recall':
                    testing_accuracy = recall_score(epoch_true_Y_testing, epoch_pred_Y_testing, average='macro')
                else:
                    sys.exit(f'The training criterion {conf.train_criterion} not implemented!')
                
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write(f'Testing loss: {testing_loss:7.4f}, testing accuracy: {testing_accuracy:7.4f}\n\n\n\n\n\n')
            fold_test_accuracies.append(testing_accuracy)
            fold_conf_mats.append(conf_mat)
    
    fold_test_accuracies_mean = np.array(fold_test_accuracies).mean()
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write('\n########################################################################################\n')
        f.write(f'Mean test accuracies of {conf.num_folds}-folds cross-validation: {fold_test_accuracies_mean}')
        f.write('\n########################################################################################\n\n\n\n')
    
    combined_conf_mat = sum(fold_conf_mats)
    num_output_categories = len(combined_conf_mat)
    
    prec = np.zeros(num_output_categories)
    rec = np.zeros(num_output_categories)
    f1 = np.zeros(num_output_categories)
    acc = np.float32(np.sum(combined_conf_mat.diagonal()))/np.float32(np.sum(combined_conf_mat))
    for i in range(combined_conf_mat.shape[0]):
        # Check if target contains current category
        containsCat = np.sum(combined_conf_mat[:,i]) > 0
        if containsCat:
            prec[i] = np.float32(combined_conf_mat[i,i])/np.float32(np.sum(combined_conf_mat[i,:]))
            rec[i] = np.float32(combined_conf_mat[i,i])/np.float32(np.sum(combined_conf_mat[:,i]))
            if np.isnan(prec[i]):
                prec[i] = 0.0
            if np.isnan(rec[i]):
                rec[i] = 0.0    
            f1[i] = 2.0*prec[i]*rec[i]/(prec[i] + rec[i])
   
            if np.isnan(f1[i]):
                f1[i] = 0.0        
        else:
            prec[i] = np.nan; rec[i] = np.nan; f1[i] = np.nan

    prec_mean = np.nanmean(prec)
    rec_mean = np.nanmean(rec)
    f1_mean = np.nanmean(f1)
    
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write('\n########################################################################################\n')
        f.write(f'Mean precision for {conf.num_folds}-folds cross-validation: {prec_mean}\n')
        f.write(f'Mean recall for {conf.num_folds}-folds cross-validation: {rec_mean}\n')
        f.write(f'Mean F1 score for {conf.num_folds}-folds cross-validation: {f1_mean}')
        f.write('\n########################################################################################\n\n\n\n')


    
    
    

