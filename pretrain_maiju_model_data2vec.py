# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

Code for running data2vec pre-training and/or using a pre-trained model to extract features.
This implementation contains simulated (= randomly generated) MAIJU data.

"""

import numpy as np
import time
import os
import sys
import pickle
import torch.nn.functional as F

from importlib.machinery import SourceFileLoader
from copy import deepcopy
from torch import cuda, no_grad, save, load, stack, from_numpy, cat
from torch.utils.data import DataLoader

from py_conf_file_into_text import convert_py_conf_file_to_text
from data2vec_ema import update_teacher_weights


# Read the configuration file
if len(sys.argv) > 2:
    sys.exit('\nUsage: \n1) python pretrain_maiju_model_data2vec.py \nOR \n2) python pretrain_maiju_model_data2vec.py <configuration_file>')
if len(sys.argv) == 2:
    conf = SourceFileLoader('', sys.argv[1]).load_module()
    conf_file_name = sys.argv[1]
else:
    try:
        import conf_pretrain_maiju_model_data2vec as conf
        conf_file_name = 'conf_pretrain_maiju_model_data2vec.py'
    except ModuleNotFoundError:
        sys.exit('\nUsage: \n1) python pretrain_maiju_model_data2vec.py \nOR \n2) python pretrain_maiju_model_data2vec.py <configuration_file>\n\n' \
        'By using the first option, you need to have a configuration file named "conf_pretrain_maiju_model_data2vec.py" in the same ' \
        'directory as "pretrain_maiju_model_data2vec.py"')


# Import our models
data2vec_encoder = getattr(__import__('maiju_nn_model', fromlist=[conf.encoder_name]), conf.encoder_name)
data2vec_transformer = getattr(__import__('maiju_nn_model', fromlist=[conf.transformer_name]), conf.transformer_name)

# Import our dataset for our data loader
data2vec_dataset = getattr(__import__('maiju_data_loader', fromlist=[conf.dataset_name]), conf.dataset_name)

# Import our loss function
data2vec_loss = getattr(__import__('torch.nn', fromlist=[conf.loss_name]), conf.loss_name)

# Import our optimization algorithm
optimization_algorithm = getattr(__import__('torch.optim', fromlist=[conf.optimization_algorithm]), conf.optimization_algorithm)

# Import our learning rate scheduler
if conf.use_lr_scheduler:
    scheduler = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler]), conf.lr_scheduler)




def frame_sig(X, winlen, hop):
    Nframes = int(np.floor(((X.shape[0] - winlen)/hop) + 1))
    numchans = X.shape[1]
    X_framed = np.zeros([Nframes, numchans, winlen], dtype=np.float32) # [Nframes, Nchans, winlen]
    for i in range(0, Nframes):
        start = i * hop
        stop = start + winlen
        X_framed[i,:,:] = np.transpose(X[start:stop,:])

    return X_framed







if __name__ == '__main__':
    
    # We make sure that we are able to write the logging file
    textfile_path, textfile_name = os.path.split(conf.name_of_log_textfile)
    if not os.path.exists(textfile_path):
        if textfile_path != '':
            os.makedirs(textfile_path)
    file = open(conf.name_of_log_textfile, 'w')
    file.close()
    
    # Read the text in the configuration file and add it to the logging file
    if conf.print_conf_contents:
        conf_file_lines = convert_py_conf_file_to_text(conf_file_name)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write(f'The configuration settings in the file {conf_file_name}:\n\n')
            for line in conf_file_lines:
                f.write(f'{line}\n')
            f.write('\n########################################################################################\n\n\n\n')
        
    
    # Use CUDA if it is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    with open(conf.name_of_log_textfile, 'a') as f:
        f.write(f'Process on {device}\n\n')
    
    # Read data
    with open(conf.name_of_log_textfile, 'a') as f:
        f.write('Reading & processing raw data...\n')
    
    if conf.print_training_updates_to_command_line:
        print(f'Initializing data... (see log file {conf.name_of_log_textfile} for further information)\n')
    
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
    
    with open(conf.name_of_log_textfile, 'a') as f:
        f.write('Done!\n\n')

    # Initialize our models, pass the models to the available device
    Encoder_student = data2vec_encoder(**conf.encoder_model_params).to(device)
    Encoder_teacher = data2vec_encoder(**conf.encoder_model_params).to(device)
    Transformer_student = data2vec_transformer(**conf.transformer_params_student).to(device)
    Transformer_teacher = data2vec_transformer(**conf.transformer_params_teacher).to(device)
    
    if conf.teacher_updater_params['student_teacher_share_encoder_weights']:
        Encoder_teacher.load_state_dict(Encoder_student.state_dict())
    
    # Give the parameters of our models to an optimizer
    model_parameters = list(Encoder_student.parameters()) + list(Transformer_student.parameters())
    optimizer = optimization_algorithm(params=model_parameters, **conf.optimization_algorithm_params)
    
    # Get our learning rate for later use
    learning_rate = optimizer.param_groups[0]['lr']
    
    # Give the optimizer to the learning rate scheduler
    if conf.use_lr_scheduler:
        lr_scheduler = scheduler(optimizer, **conf.lr_scheduler_params)

    # Initialize our loss function as a class
    loss_function = data2vec_loss(**conf.loss_params)

    # Variables for early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience_counter = 0
    
    if conf.load_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Loading model from file...\n')
            f.write(f'Loading model {conf.encoder_student_best_model_name}\n')
            f.write(f'Loading model {conf.encoder_teacher_best_model_name}\n')
            f.write(f'Loading model {conf.transformer_student_best_model_name}\n')
            f.write(f'Loading model {conf.transformer_teacher_best_model_name}\n')
            f.write(f'Initializing teacher model updater from file {conf.teacher_updater_state_dict_savename}\n')
        Encoder_student.load_state_dict(load(conf.encoder_student_best_model_name, map_location=device))
        Encoder_teacher.load_state_dict(load(conf.encoder_teacher_best_model_name, map_location=device))
        Transformer_student.load_state_dict(load(conf.transformer_student_best_model_name, map_location=device))
        Transformer_teacher.load_state_dict(load(conf.transformer_teacher_best_model_name, map_location=device))
        best_model_student_encoder = deepcopy(Encoder_student.state_dict())
        best_model_teacher_encoder = deepcopy(Encoder_teacher.state_dict())
        best_model_student_transformer = deepcopy(Transformer_student.state_dict())
        best_model_teacher_transformer = deepcopy(Transformer_teacher.state_dict())
        
        # Initialize the teacher model weight updater based on where the training was stopped during the last training
        with open(conf.teacher_updater_state_dict_savename, 'rb') as fp:
            teacher_updater_state_dict_fromfile = pickle.load(fp)
        teacher_updater = update_teacher_weights(Encoder_teacher, Transformer_teacher, **teacher_updater_state_dict_fromfile)
        
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n\n')
    else:
        best_model_student_encoder = None
        best_model_teacher_encoder = None
        best_model_student_transformer = None
        best_model_teacher_transformer = None
        
        # Initialize the teacher model weight updater based on the configuration file
        teacher_updater = update_teacher_weights(Encoder_teacher, Transformer_teacher, **conf.teacher_updater_params)
    
    
    # Initialize the data loaders
    if conf.train_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Initializing training set...\n')
        training_set = data2vec_dataset(Data, train_val_test='train', **conf.params_train_dataset)
        train_data_loader = DataLoader(training_set, **conf.params_train)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n')
            f.write('Initializing validation set...\n')
        validation_set = data2vec_dataset(Data, train_val_test='validation', **conf.params_validation_dataset)
        validation_data_loader = DataLoader(validation_set, **conf.params_train)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n')
    if conf.extract_features:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Initializing full dataset for feature extraction...\n')
        test_set = data2vec_dataset(Data, train_val_test='test', **conf.params_feature_extraction_dataset)
        test_data_loader = DataLoader(test_set, **conf.params_feature_extraction)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n\n')
    
    # Check that the number of averaged layers does not exceed the number of Transformer encoder layers
    if conf.num_top_averaged_teacher_layers > conf.num_transformer_encoder_layers:
        sys.exit('num_top_averaged_teacher_layers cannot be larger than num_transformer_encoder_layers ' \
                 f'({conf.num_top_averaged_teacher_layers} > {conf.num_transformer_encoder_layers})')
    
    if conf.max_epochs < conf.min_train_epochs:
        sys.exit(f'max_epochs cannot be smaller than min_train_epochs ({conf.max_epochs} < {conf.min_train_epochs})')
    
    # Flag for indicating if max epochs are reached
    max_epochs_reached = 1
    
    # Start training our model
    if conf.train_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Starting training...\n')
        
        if conf.print_training_updates_to_command_line:
            print('Starting training...\n')
        
        for epoch in range(1, conf.max_epochs + 1):
            
            start_time = time.time()
    
            # Lists containing the losses of each epoch
            epoch_loss_training = []
            epoch_loss_validation = []
            epoch_prediction_loss = []
            
            # Lists containing the variances (over the time dimension) of our embeddings, predicted outputs,
            # and training targets
            epoch_variance_embeddings = []
            epoch_variance_predictions = []
            epoch_variance_targets = []
    
            # Indicate that we are in training mode, so e.g. dropout will function
            Encoder_student.train()
            Transformer_student.train()
            
            # The teacher network is constantly on evaluation mode
            Encoder_teacher.eval()
            Transformer_teacher.eval()
            
            # Loop through every batch of our training data
            for train_data in train_data_loader:
                
                # Get the minibatches. We remove each input sequence that contains over 80% of masked data
                X_input_initial, data_masks_initial = [element.to(device) for element in train_data]
                X_input = []
                data_masks = []
                for sequence_index in range(len(X_input_initial)):
                    if not data_masks_initial[sequence_index].sum() > 0.80 * X_input_initial.size()[1]:
                        X_input.append(X_input_initial[sequence_index])
                        data_masks.append(data_masks_initial[sequence_index])
                if len(X_input) == 0:
                    continue
                X_input = stack(X_input, dim=0)
                data_masks = stack(data_masks, dim=0)
                padding_masks = data_masks.bool()
                
                # We add one additional padding value due to the CLS token
                if conf.include_cls_token:
                    padding_masks = cat((from_numpy(np.array([False])).repeat(X_input.size()[0], 1).to(X_input.device), padding_masks), dim=1)
                
                # Zero the gradient of the optimizer
                optimizer.zero_grad()
                
                # Pass our data through the student encoder, one sequence at a time
                Embedding = []
                for i in range(len(X_input)):
                    seq_embedding = Encoder_student(X_input[i].float())
                    Embedding.append(seq_embedding)
                Embedding = stack(Embedding, dim=0)
                
                # Pass our embeddings to the student Transformer encoder
                X_output, embedding_mask_indices = Transformer_student(Embedding, src_key_padding_mask=padding_masks)
                embedding_mask_indices = embedding_mask_indices.to(device)
                
                # Do the same thing with our teacher network
                with no_grad():
                    Embedding_teacher = []
                    for i in range(len(X_input)):
                        seq_embedding = Encoder_teacher(X_input[i].float())
                        Embedding_teacher.append(seq_embedding)
                    Embedding_teacher = stack(Embedding_teacher, dim=0)
                    Transformer_teacher_layer_outputs = Transformer_teacher(Embedding_teacher, src_key_padding_mask=padding_masks)
                
                    # We construct the training targets by:
                    #    1. Taking the output of the top K Transformer encoder blocks
                    #    2. Normalizing the output of each block
                    #    3. Averaging the normalized output of each block
                    top_k_layer_outputs = Transformer_teacher_layer_outputs[-conf.num_top_averaged_teacher_layers:]
                    del Transformer_teacher_layer_outputs
                    top_k_layer_outputs = [F.instance_norm(layer_output.permute(0, 2, 1)).permute(0, 2, 1) for layer_output in top_k_layer_outputs]
                    Y = sum(top_k_layer_outputs) / len(top_k_layer_outputs)
                
                # Compute the variance (over the time dimension, ignoring paddings) of our inputs, predicted outputs, and training targets.
                X_input_variance = []
                Embedding_variance = []
                X_output_variance = []
                Y_variance = []
                for i in range(X_output.size()[0]):
                    if conf.compute_variance_for_unmasked_parts:
                        # Since the embedding masks can bring additional variance to the outputs, we ignore them to e.g. avoid learning a
                        # high-variance mask to make X_output to artificially contain a higher variance then it truly has without a mask.
                        X_input_variance.append(X_input[i, ~padding_masks[i], :][~embedding_mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                        Embedding_variance.append(Embedding[i, ~padding_masks[i], :][~embedding_mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                        X_output_variance.append(X_output[i, ~padding_masks[i], :][~embedding_mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                        Y_variance.append(Y[i, ~padding_masks[i], :][~embedding_mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                    else:
                        X_input_variance.append(X_input[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                        Embedding_variance.append(Embedding[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                        X_output_variance.append(X_output[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                        Y_variance.append(Y[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                
                X_input_variance = stack(X_input_variance, dim=0).mean()
                Embedding_variance = stack(Embedding_variance, dim=0).mean()
                X_output_variance = stack(X_output_variance, dim=0).mean()
                Y_variance = stack(Y_variance, dim=0).mean()
                epoch_variance_embeddings.append(Embedding_variance.item())
                epoch_variance_predictions.append(X_output_variance.item())
                epoch_variance_targets.append(Y_variance.item())
                
                if conf.compute_loss_for_masked_embeddings:
                    # We compute the loss only for the masked parts of the embeddings
                    X_outputs_masked_section = X_output[embedding_mask_indices]
                    Y_masked_section = Y[embedding_mask_indices]
                    loss = loss_function(X_outputs_masked_section, Y_masked_section) * conf.loss_scaler
                else:
                    # We compute the loss for the non-padded parts of the embeddings (excluding possible CLS tokens)
                    loss = []
                    for i in range(X_output.size()[0]):
                        X_non_padded = X_output[i, ~padding_masks[i], :]
                        Y_non_padded = Y[i, ~padding_masks[i], :]
                        if conf.include_cls_token:
                            # We exclude the CLS token
                            X_non_padded = X_non_padded[:, 1:, :]
                            Y_non_padded = Y_non_padded[:, 1:, :]
                        loss.append(loss_function(X_non_padded, Y_non_padded))
                    loss = (sum(loss) / len(loss)) * conf.loss_scaler
                
                epoch_prediction_loss.append(loss.item())
                
                # Perform the backward pass
                loss.backward()
                
                # Update the weights
                optimizer.step()

                # Add the loss to the total loss of the batch
                epoch_loss_training.append(loss.item())
                
                # Update our teacher model
                teacher_encoder_weights, teacher_transformer_weights, tau = teacher_updater.update(Encoder_student,
                                                                                                  Transformer_student)
                Encoder_teacher.load_state_dict(teacher_encoder_weights, strict=True)
                Transformer_teacher.load_state_dict(teacher_transformer_weights, strict=True)
                
            
            # Indicate that we are in evaluation mode, so e.g. dropout will not function
            Encoder_student.eval()
            Transformer_student.eval()
    
            # Make PyTorch not calculate the gradients, so everything will be much faster.
            with no_grad():
                
                # Loop through every batch of our validation data and perform a similar process as for the training data
                for validation_data in validation_data_loader:
                    X_input_initial, data_masks_initial = [element.to(device) for element in validation_data]
                    X_input = []
                    data_masks = []
                    for sequence_index in range(len(X_input_initial)):
                        if not data_masks_initial[sequence_index].sum() > 0.80 * X_input_initial.size()[1]:
                            X_input.append(X_input_initial[sequence_index])
                            data_masks.append(data_masks_initial[sequence_index])
                    if len(X_input) == 0:
                        continue
                    X_input = stack(X_input, dim=0)
                    data_masks = stack(data_masks, dim=0)
                    padding_masks = data_masks.bool()
                    if conf.include_cls_token:
                        padding_masks = cat((from_numpy(np.array([False])).repeat(X_input.size()[0], 1).to(X_input.device), padding_masks), dim=1)
                    Embedding = []
                    for i in range(len(X_input)):
                        seq_embedding = Encoder_student(X_input[i].float())
                        Embedding.append(seq_embedding)
                    Embedding = stack(Embedding, dim=0)
                    X_output, embedding_mask_indices = Transformer_student(Embedding, src_key_padding_mask=padding_masks)
                    embedding_mask_indices = embedding_mask_indices.to(device)
                    with no_grad():
                        Embedding_teacher = []
                        for i in range(len(X_input)):
                            seq_embedding = Encoder_teacher(X_input[i].float())
                            Embedding_teacher.append(seq_embedding)
                        Embedding_teacher = stack(Embedding_teacher, dim=0)
                        Transformer_teacher_layer_outputs = Transformer_teacher(Embedding_teacher, src_key_padding_mask=padding_masks)
                        top_k_layer_outputs = Transformer_teacher_layer_outputs[-conf.num_top_averaged_teacher_layers:]
                        del Transformer_teacher_layer_outputs
                        top_k_layer_outputs = [F.instance_norm(layer_output.permute(0, 2, 1)).permute(0, 2, 1) for layer_output in top_k_layer_outputs]
                        Y = sum(top_k_layer_outputs) / len(top_k_layer_outputs)
                    X_input_variance = []
                    Embedding_variance = []
                    X_output_variance = []
                    Y_variance = []
                    for i in range(X_output.size()[0]):
                        if conf.compute_variance_for_unmasked_parts:
                            X_input_variance.append(X_input[i, ~padding_masks[i], :][~embedding_mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                            Embedding_variance.append(Embedding[i, ~padding_masks[i], :][~embedding_mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                            X_output_variance.append(X_output[i, ~padding_masks[i], :][~embedding_mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                            Y_variance.append(Y[i, ~padding_masks[i], :][~embedding_mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                        else:
                            X_input_variance.append(X_input[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                            Embedding_variance.append(Embedding[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                            X_output_variance.append(X_output[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                            Y_variance.append(Y[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                    X_input_variance = stack(X_input_variance, dim=0).mean()
                    Embedding_variance = stack(Embedding_variance, dim=0).mean()
                    X_output_variance = stack(X_output_variance, dim=0).mean()
                    Y_variance = stack(Y_variance, dim=0).mean()
                    epoch_variance_embeddings.append(Embedding_variance.item())
                    epoch_variance_predictions.append(X_output_variance.item())
                    epoch_variance_targets.append(Y_variance.item())
                    if conf.compute_loss_for_masked_embeddings:
                        X_outputs_masked_section = X_output[embedding_mask_indices]
                        Y_masked_section = Y[embedding_mask_indices]
                        loss = loss_function(X_outputs_masked_section, Y_masked_section) * conf.loss_scaler
                    else:
                        loss = []
                        for i in range(X_output.size()[0]):
                            X_non_padded = X_output[i, ~padding_masks[i], :]
                            Y_non_padded = Y[i, ~padding_masks[i], :]
                            if conf.include_cls_token:
                                X_non_padded = X_non_padded[:, 1:, :]
                                Y_non_padded = Y_non_padded[:, 1:, :]
                            loss.append(loss_function(X_non_padded, Y_non_padded))
                        loss = (sum(loss) / len(loss)) * conf.loss_scaler
                    epoch_prediction_loss.append(loss.item())
                    epoch_loss_validation.append(loss.item())
    
            # Calculate mean losses and variances
            epoch_loss_training = np.array(epoch_loss_training).mean()
            epoch_loss_validation = np.array(epoch_loss_validation).mean()
            epoch_prediction_loss = np.array(epoch_prediction_loss).mean()
            epoch_variance_embeddings = np.array(epoch_variance_embeddings).mean()
            epoch_variance_predictions = np.nanmean(np.array(epoch_variance_predictions))
            epoch_variance_targets = np.nanmean(np.array(epoch_variance_targets))
    
            # Check early stopping conditions
            if epoch_loss_validation < lowest_validation_loss and epoch_loss_validation > 0.0001:
                lowest_validation_loss = epoch_loss_validation
                patience_counter = 0
                best_model_student_encoder = deepcopy(Encoder_student.state_dict())
                best_model_teacher_encoder = deepcopy(Encoder_teacher.state_dict())
                best_model_student_transformer = deepcopy(Transformer_student.state_dict())
                best_model_teacher_transformer = deepcopy(Transformer_teacher.state_dict())
                best_validation_epoch = epoch
                if conf.save_best_model:
                    # We first make sure that we are able to write the files
                    save_names = [conf.encoder_student_best_model_name, conf.encoder_teacher_best_model_name,
                                  conf.transformer_student_best_model_name, conf.transformer_teacher_best_model_name,
                                  conf.teacher_updater_state_dict_savename]
                    for model_save_name in save_names:
                        model_path, model_filename = os.path.split(model_save_name)
                        if not os.path.exists(model_path):
                            if model_path != '':
                                os.makedirs(model_path)
                    
                    save(best_model_student_encoder, conf.encoder_student_best_model_name)
                    save(best_model_teacher_encoder, conf.encoder_teacher_best_model_name)
                    save(best_model_student_transformer, conf.transformer_student_best_model_name)
                    save(best_model_teacher_transformer, conf.transformer_teacher_best_model_name)
                    teacher_updater_state_dict = teacher_updater.get_teacher_updater_state()
                    with open(conf.teacher_updater_state_dict_savename, 'wb') as sv:
                        pickle.dump(teacher_updater_state_dict, sv)
            else:
                patience_counter += 1
            
            end_time = time.time()
            epoch_time = end_time - start_time
            
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write(f'Epoch: {epoch:04d} | Mean training loss: {epoch_loss_training:6.4f} | '
                  f'Mean validation loss: {epoch_loss_validation:6.4f} (lowest: {lowest_validation_loss:6.4f}) | '
                  f'Mean prediction loss: {epoch_prediction_loss:6.4f} | Mean Emb variance: {epoch_variance_embeddings:6.4f} | '
                  f'Mean X variance: {epoch_variance_predictions:6.4f} | Mean Y variance: {epoch_variance_targets:6.4f} | '
                  f'Latest tau value: {tau:7.6f} | Duration: {epoch_time:4.2f} seconds\n')
                
            # We check that do we need to update the learning rate based on the validation loss
            if conf.use_lr_scheduler:
                if conf.lr_scheduler == 'ReduceLROnPlateau':
                    lr_scheduler.step(epoch_loss_validation)
                else:
                    lr_scheduler.step()
                current_learning_rate = optimizer.param_groups[0]['lr']
                if current_learning_rate != learning_rate:
                    learning_rate = current_learning_rate
                    with open(conf.name_of_log_textfile, 'a') as f:
                        f.write(f'Updated learning rate after epoch {epoch} based on learning rate scheduler, now lr={learning_rate}\n')
            
            if conf.print_training_updates_to_command_line:
                print(f'Finished training epoch {epoch:04d} (see log file {conf.name_of_log_textfile} for further information)')
            
            # If patience counter is fulfilled, stop the training
            if patience_counter >= conf.patience:
                max_epochs_reached = 0
                break
            
            # If the variance of our predictions or targets falls too low, we stop training
            if epoch >= conf.min_train_epochs and (epoch_variance_predictions < conf.min_prediction_variance or epoch_variance_targets < conf.min_target_variance):
                max_epochs_reached = 0
                break
            
        if max_epochs_reached:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('\nMax number of epochs reached, stopping training\n\n')
        else:
            if patience_counter >= conf.patience:
                with open(conf.name_of_log_textfile, 'a') as f:
                    f.write('\nExiting due to early stopping\n\n')
            else:
                with open(conf.name_of_log_textfile, 'a') as f:
                    f.write('\nThe variance of our predictions or targets fell too low, stopping training\n\n')
        
        if best_model_student_encoder is None:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('\nNo best model. The criteria for the lowest acceptable validation loss not satisfied!\n\n')
            sys.exit('No best model, exiting...')
        else:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write(f'\nBest epoch {best_validation_epoch} with validation loss {lowest_validation_loss}\n\n')
        
        if conf.print_training_updates_to_command_line:
            print(f'\nFinished training process (see log file {conf.name_of_log_textfile} for further information)\n')
        
        
    # Perform feature extraction using a trained model
    if conf.extract_features:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('\n\nExtracting features using a trained data2vec model...\n\n')
        
        if conf.print_training_updates_to_command_line:
            print('Extracting features using a trained data2vec model...\n')
        
        # Initialize the best version of our model, leave embedding masks out
        conf.transformer_params_student['use_embedding_mask'] = False
        Transformer_student = data2vec_transformer(**conf.transformer_params_student).to(device)
        try:
            Encoder_student.load_state_dict(load(conf.encoder_student_best_model_name, map_location=device))
            Transformer_student.load_state_dict(load(conf.transformer_student_best_model_name, map_location=device))
        except (FileNotFoundError, RuntimeError):
            Encoder_student.load_state_dict(best_model_student_encoder)
            Transformer_student.load_state_dict(best_model_student_transformer)
                
        Encoder_student.eval()
        Transformer_student.eval()
        with no_grad():
            X_outputs_array = []
            padding_mask_indices_array = []
            if conf.include_cls_token:
                CLS_token_outputs = []
            for test_data in test_data_loader:
                X_input, data_masks = [element.to(device) for element in test_data]
                padding_masks = data_masks.bool()
                if conf.include_cls_token:
                    padding_masks = cat((from_numpy(np.array([False])).repeat(X_input.size()[0], 1).to(X_input.device), padding_masks), dim=1)
                Embedding = []
                for i in range(len(X_input)):
                    seq_embedding = Encoder_student(X_input[i].float())
                    Embedding.append(seq_embedding)
                Embedding = stack(Embedding, dim=0)
                X_output = Transformer_student(Embedding, src_key_padding_mask=padding_masks)
                if conf.include_cls_token:
                    CLS_token_outputs.append(X_output[:, 0, :].cpu().numpy())
                    X_output = X_output[:, 1:, :]
                    padding_masks = padding_masks[:, 1:]
                X_outputs_array.append(X_output.cpu().numpy())
                padding_mask_indices_array.append(padding_masks.cpu().numpy())
        
        X_outputs_array = np.vstack(X_outputs_array)
        padding_mask_indices_array = np.vstack(padding_mask_indices_array)
        if conf.include_cls_token:
            CLS_token_outputs = np.vstack(CLS_token_outputs)
        
        # We make sure that we are able to write the files before we save the files
        if conf.include_cls_token:
            save_names = [conf.feature_extraction_model_output_savefile, conf.feature_extraction_padding_mask_indices_savefile, conf.cls_token_output_savefile]
        else:
            save_names = [conf.feature_extraction_model_output_savefile, conf.feature_extraction_padding_mask_indices_savefile]
        for file_save_name in save_names:
            file_path, filename = os.path.split(file_save_name)
            if not os.path.exists(file_path):
                if file_path != '':
                    os.makedirs(file_path)
        np.save(conf.feature_extraction_model_output_savefile, X_outputs_array)
        np.save(conf.feature_extraction_padding_mask_indices_savefile, padding_mask_indices_array)
        if conf.include_cls_token:
            np.save(conf.cls_token_output_savefile, CLS_token_outputs)
        
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write(f'Done! Data written to files {conf.feature_extraction_model_output_savefile}')
            f.write(f' and {conf.feature_extraction_padding_mask_indices_savefile}\n')
        
        if conf.include_cls_token:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write(f'CLS token output written to file {conf.cls_token_output_savefile}')
        
