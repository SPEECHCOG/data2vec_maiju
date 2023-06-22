#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The configuration file for finetune_pretrained_maiju_model.py.

"""

experiment_num = 1

# A flag for determining whether we want to print the contents of the configuration file into the
# logging file
print_conf_contents = True

# A flag for determining whether we want to print updates of our training process to the command line
print_training_updates_to_command_line = True

# The directory where the experimental results are saved (the models and the logging output)
result_dir = f'finetuning_results_{experiment_num}'

# The name of the text file into which we log the output of the training process. Please note that this
# file (and its directory) will be saved under the directory result_dir.
name_of_log_textfile = f'maiju_random_data_finetuning_trainlog_{experiment_num}.txt'

# Flag for saving the best model (according to validation accuracy) after each training epoch where the
# validation accuracy is higher than before
save_best_model = True

# The names of the model weight files of the best models (according to validation accuracy) for
# loading/saving model weights. Please note that the fine-tuned model files (and directories) will be saved
# under the directory result_dir under training fold-specific directories.
best_model_encoder_name_pretrained = f'pretrained_models/data2vec_Encoder_best_student_model_maiju_{experiment_num}.pt'
best_model_timeseries_name_pretrained = f'pretrained_models/data2vec_Transformer_best_student_model_maiju_{experiment_num}.pt'
best_model_encoder_name_finetuned = f'finetuned_models/maiju_best_encoder_model_finetuned_{experiment_num}.pt'
best_model_timeseries_name_finetuned = f'finetuned_models/maiju_best_timeseries_model_finetuned_{experiment_num}.pt'


"""The hyperparameters for our training process"""

# Flag for training our model(s)
train_model = True

# Flag for testing our trained model(s)
test_model = True

# The maximum number of training epochs
max_epochs = 800

# The learning rate of our model training
learning_rate = 4e-5

# The number of frames in each input sequence for our model (Fs=52 Hz, 60-sample hop length --> 260 frames is 5 minutes)
train_sequence_length = 260

# The number of input sequences that we feed into our model before computing the mean loss (and performing backpropagation
# during training).
batch_size = 1

# The patience counter for early stopping
patience = 100

# Dropout rate of the encoder model
dropout_encoder_model = 0.3

# Dropout rate of the timeseries model
dropout_timeseries_model = 0.4

# Window length (in samples)
window_len = 120

# Hop length (in samples)
hop_len = 60

# Select the training criterion
train_criterion = 'f1' # Options: 'f1' / 'recall'

# The number of folds for k-folds cross-validation
num_folds = 10

# A flag whether we want to randomize the order of the babies before applying k-folds cross-validation
randomize_order_kfolds = True

# Define our loss function that we want to use from torch.nn
loss_name = 'CrossEntropyLoss'

# The hyperparameters for the loss function
loss_params = {}

# Define the optimization algorithm we want to use from torch.optim
optimization_algorithm = 'Adam'

# The hyperparameters for our optimization algorithm
optimization_algorithm_params = {'lr': learning_rate}

# A flag to determine if we want to use a learning rate scheduler
use_lr_scheduler = True

# Define our learning rate schedulers for the fine-tuning stages 1 and 2 (from torch.optim.lr_scheduler)
lr_scheduler_stage_1 = 'ReduceLROnPlateau'
lr_scheduler_params_stage_1 = {'mode': 'max',
                               'factor': 0.5,
                               'patience': 30}

lr_scheduler_stage_2_part_1_epochs = 20
lr_scheduler_stage_2_part_1 = 'LinearLR'
lr_scheduler_params_stage_2_part_1 = {'start_factor': 0.001,
                                      'total_iters': lr_scheduler_stage_2_part_1_epochs}
lr_scheduler_stage_2_part_2 = 'ReduceLROnPlateau'
lr_scheduler_params_stage_2_part_2 = {'mode': 'max',
                                      'factor': 0.5,
                                      'patience': 30}


"""The hyperparameters for training data augmentation"""

# Select whether we want to use data augmentation for our training data or not
use_augmentation = True

# Probability for additive noise augmentation
aug_p_noise = 0.0

# If we perform additive noise augmentation, the probability for adding noise to samples
aug_p_dropout = 0.0

# Probability for performing a random rotation
aug_p_rotation = 0.0

# Probability for sensor dropout
aug_p_chandropout = 0.3

# Probability for time warping
aug_p_time_warping = 0.0


"""The hyperparameters for our dataset and data loaders"""

# The number of randomly generated MAIJU recordings of babies
num_randomly_generated_babydata = 20

# Define our dataset for our data loader that we want to use from the file maiju_data_loader.py
dataset_name = 'random_maiju_data_dataset'

# The ratio in which we split our training data into training and validation sets. For example, a ratio
# of 0.8 will result in 80% of our training data being in the training set and 20% in the validation set.
train_val_ratio = 0.8

# Select whether we want to shuffle our training data
shuffle_training_data = True

# Select if we want to split our training and validation data so that baby-specific data is included
# in both sets.
mix_train_val_babies = False

# The hyperparameters for our data loaders
params_train_dataset = {'train_sequence_length': train_sequence_length,
                        'train_val_ratio': train_val_ratio,
                        'window_len': window_len,
                        'hop_len': hop_len,
                        'mix_train_val_babies': mix_train_val_babies,
                        'augment_train_data': use_augmentation,
                        'aug_p_noise': aug_p_noise,
                        'aug_p_dropout': aug_p_dropout,
                        'aug_p_rotation': aug_p_rotation,
                        'aug_p_chandropout': aug_p_chandropout,
                        'aug_p_time_warping': aug_p_time_warping,
                        'include_artificial_labels': True}

params_validation_dataset = {'train_sequence_length': train_sequence_length,
                             'train_val_ratio': train_val_ratio,
                             'window_len': window_len,
                             'hop_len': hop_len,
                             'mix_train_val_babies': mix_train_val_babies,
                             'include_artificial_labels': True}

params_test_dataset = {'train_sequence_length': train_sequence_length,
                       'window_len': window_len,
                       'hop_len': hop_len,
                       'include_artificial_labels': True}

# The hyperparameters for training and validation (arguments for torch.utils.data.DataLoader object)
params_train = {'batch_size': batch_size,
                'shuffle': shuffle_training_data,
                'drop_last': False}

# The hyperparameters for using our trained data2vec model to extract features (arguments for
# torch.utils.data.DataLoader object)
params_test = {'batch_size': batch_size,
               'shuffle': False,
               'drop_last': False}


"""The neural network hyperparameters"""

# Define our models that we want to use from the file maiju_nn_model.py
encoder_model = 'SENSOR_MODULE'
timeseries_model = 'data2vec_transformer_finetuning'

# Hyperparameters for our encoder model
num_input_channels = 24
encoder_num_latent_channels = 80
encoder_num_output_channels = 160

encoder_model_params = {'s_channels': num_input_channels,
                        'input_channels': window_len,
                        'latent_channels': encoder_num_latent_channels,
                        'output_channels': encoder_num_output_channels,
                        'dropout': dropout_encoder_model,
                        'conv_1_kernel_size': (3,11),
                        'conv_2_kernel_size': (4,5),
                        'conv_3_kernel_size': 4,
                        'conv_4_kernel_size': 4,
                        'conv_1_stride': (3,5),
                        'conv_2_stride': (1,2),
                        'conv_3_stride': 1,
                        'conv_4_stride': 1,
                        'conv_1_zero_padding': 'valid',
                        'conv_2_zero_padding': (0,2),
                        'conv_3_zero_padding': 'same',
                        'conv_4_zero_padding': 'valid',
                        'pooling_1_zero_padding': (0,0),
                        'pooling_2_zero_padding': 0,
                        'pooling_2_kernel_size': 4,
                        'normalization_type': 'layernorm'}


# Hyperparameters for our timeseries model

# The size of the hidden dimension of the feed-forward neural network part of the Transformer encoder blocks
transformer_hidden_dim = 640

# The number of attention heads for each multi-head self-attention
num_attention_heads = 10

# The number of Transformer encoder blocks
num_transformer_encoder_layers = 12

# Defines whether we want to use absolute positional encodings (using sinusoids) or relative positional
# encodings (using a CNN layer) for our embeddings. Relative positional encoding was used in the
# data2vec paper, whereas absolute positional encodings were used in the original Transformer paper.
#     Options: 'absolute' or 'relative'
positional_encoding_type = 'relative'

timeseries_model_params = {'dim_model': encoder_num_output_channels,
                           'dim_feedforward': transformer_hidden_dim,
                           'classification_layer_latent_dim': encoder_num_output_channels,
                           'output_channels': 9,
                           'num_heads': num_attention_heads,
                           'num_encoder_layers': num_transformer_encoder_layers,
                           'dropout': dropout_timeseries_model,
                           'transformer_activation_function': 'gelu',
                           'only_attend_to_previous_context': False,
                           'use_sqrt': False,
                           'use_embedding_projection': True,
                           'include_cls_token': False,
                           'is_cls_token_random': False,
                           'positional_encoding_type': positional_encoding_type,
                           'dropout_pos_encoding': 0.0,
                           'rel_pos_encoding_conv_in_dim': encoder_num_output_channels,
                           'rel_pos_encoding_conv_out_dim': encoder_num_output_channels,
                           'rel_pos_encoding_conv_kernel_size': 13,
                           'rel_pos_encoding_conv_padding': 6}
