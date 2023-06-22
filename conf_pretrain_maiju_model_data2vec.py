#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The configuration file for pretrain_maiju_model_data2vec.py.

"""

experiment_num = 1

"""The hyperparameters for our training and feature extraction processes"""

# The maximum number of training epochs
max_epochs = 10000

# The patience counter for early stopping
patience = 25

# Dropout rate of the encoder model
dropout_encoder_model = 0.1

# The learning rate of our model training
learning_rate = 1e-4

# The number of frames in each input sequence for our model (Fs=52 Hz, 60-sample hop length --> 260 frames is 5 minutes)
train_sequence_length = 260

# The number of input sequences that we feed into our model before computing the mean loss (and performing backpropagation
# during training).
batch_size = 32

# Window length (in samples)
window_len = 120

# Hop length (in samples)
hop_len = 60

# Flag for training our data2vec model
train_model = True

# Flag for using our trained data2vec model to extract features
extract_features = False

# Flag for loading the weights for our model, i.e. flag for continuing a previous training process
load_model = False

# Flag for saving the best model (according to validation loss) after each training epoch where the
# validation loss is lower than before
save_best_model = True

# The name of the text file into which we log the output of the training process
name_of_log_textfile = f'trainlogs/data2vec_trainlog_maiju_random_data_{experiment_num}.txt'

# A flag for determining whether we want to print the contents of the configuration file into the
# logging file
print_conf_contents = True

# A flag for determining whether we want to print updates of our training process to the command line
print_training_updates_to_command_line = True

# Define our models that we want to use from the file maiju_nn_model.py
encoder_name = 'SENSOR_MODULE'
transformer_name = 'data2vec_transformer_encoder'

# A flag for computing the loss only for the masked parts of the embeddings (True) as was did in the
# data2vec paper. If set to False, the loss is computed for the non-padded parts of the embeddings,
# including the non-masked parts.
compute_loss_for_masked_embeddings = True

# Define our loss function that we want to use from torch.nn
loss_name = 'MSELoss'

# The hyperparameters for the loss function
loss_params = {}

# The scaling multiplier for the loss value, a value of 1.0 means no scaling
loss_scaler = 1.0

# A flag for defining whether we want to compute the variance over the time dimension only for non-padded
# and unmasked parts of our predicted outputs and training targets (True), or only for the non-padded
# parts (False).
compute_variance_for_unmasked_parts = True

# Defines the minimum number of training epochs, just to make sure that we don't stop training too soon if
# the variance of the predictions or targets is too low in the beginning of the training process
min_train_epochs = 3

# The minimum acceptable variances of our predictions and targets. If we don't care about the minimum
# acceptable variance, we can set these variables to some negative value (variance cannot be less than 0).
# In the original data2vec implementation in Fairseq, training was stopped if variance fell below a
# given threshold.
min_prediction_variance = -9999.0
min_target_variance = -9999.0

# Define the optimization algorithm we want to use from torch.optim
optimization_algorithm = 'Adam'

# The hyperparameters for our optimization algorithm
optimization_algorithm_params = {'lr': learning_rate}

# A flag to determine if we want to use a learning rate scheduler
use_lr_scheduler = True

# Define which learning rate scheduler we want to use from torch.optim.lr_scheduler
lr_scheduler = 'ReduceLROnPlateau'

# The hyperparameters for the learning rate scheduler
lr_scheduler_params = {'mode': 'min',
                       'factor': 0.5,
                       'patience': 15}

# The names of the model weight files of the best models (according to validation loss) for
# loading/saving model weights
encoder_student_best_model_name = f'pretrained_models/data2vec_Encoder_best_student_model_maiju_{experiment_num}.pt'
encoder_teacher_best_model_name = f'pretrained_models/data2vec_Encoder_best_teacher_model_maiju_{experiment_num}.pt'
transformer_student_best_model_name = f'pretrained_models/data2vec_Transformer_best_student_model_maiju_{experiment_num}.pt'
transformer_teacher_best_model_name = f'pretrained_models/data2vec_Transformer_best_teacher_model_maiju_{experiment_num}.pt'

# The names of the files containing the output of the feature extraction
feature_extraction_model_output_savefile = f'feature_extraction_output/data2vec_fe_model_output_maiju_{experiment_num}.npy'
feature_extraction_padding_mask_indices_savefile = f'feature_extraction_output/data2vec_fe_padding_mask_indices_maiju_{experiment_num}.npy'
cls_token_output_savefile = f'feature_extraction_output/data2vec_cls_token_output_maiju_{experiment_num}.npy' # Applies only if "include_cls_token = True"


"""The hyperparameters for training data augmentation"""

# Select whether we want to use data augmentation for our training data or not
use_augmentation = True

# Probability for additive noise augmentation
aug_p_noise = 0.0

# If we perform additive noise augmentation, the probability for adding noise to samples
aug_p_dropout = 0.0

# Probability for performing a random rotation
aug_p_rotation = 0.3

# Probability for sensor dropout
aug_p_chandropout = 0.1

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
                        'aug_p_time_warping': aug_p_time_warping}

params_validation_dataset = {'train_sequence_length': train_sequence_length,
                             'train_val_ratio': train_val_ratio,
                             'window_len': window_len,
                             'hop_len': hop_len,
                             'mix_train_val_babies': mix_train_val_babies}

params_feature_extraction_dataset = {'train_sequence_length': train_sequence_length,
                                     'window_len': window_len,
                                     'hop_len': hop_len}

# The hyperparameters for training and validation (arguments for torch.utils.data.DataLoader object)
params_train = {'batch_size': batch_size,
                'shuffle': shuffle_training_data,
                'drop_last': False}

# The hyperparameters for using our trained data2vec model to extract features (arguments for
# torch.utils.data.DataLoader object)
params_feature_extraction = {'batch_size': batch_size,
                             'shuffle': False,
                             'drop_last': False}


"""The hyperparameters for our encoder model"""

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


"""The hyperparameters for our Transformer model"""

# The dimensionality of the input embedding sequences for the Transformer encoder
embedding_dim = encoder_num_output_channels

# The size of the hidden dimension of the feed-forward neural network part of the Transformer encoder blocks
transformer_hidden_dim = 640

# The number of attention heads for each multi-head self-attention
num_attention_heads = 10

# The number of Transformer encoder blocks
num_transformer_encoder_layers = 12

# The dropout of the Transformer encoder blocks
dropout_transformer = 0.0

# The activation function for the Transformer feed-forward neural network part. Options: 'relu' and 'gelu'.
# ReLU was used in the original Transformer paper, whereas GELU was used in e.g. wav2vec 2.0 and data2vec.
transformer_activation_function = 'gelu'

# Defines whether we want to have the same number of embedding masks in each batch element (as was used in the
# original data2vec implementation in Fairseq). If set to True: After computing the embedding mask indices, the
# minimum number of embedding masks in a batch element is first defined. Then, for the rest of the batch elements
# containing more embedding masks, mask indices are randomly removed until each batch element has the same number
# of embedding masks. Please note that this might be problematic if there are large differences between the lengths
# of the batch elements (e.g. a long sample might have very few masks compared to the length of the sample).
require_same_num_embedding_masks = False

# The probability of a frame being the start of an embedding mask when masking embeddings
# for the student network
prob_frame_is_start_of_embedding_mask = 0.15

# The length of the embedding masks (in frames) when masking the embeddings for the student network
embedding_mask_length_frames = 3

# The minimum number of embedding mask starting frames in each embedding (the embedding mask start indices are
# chosen randomly, so without this parameter there is a chance that there might be # no masked frames at all
min_num_mask_start_frames = 1

# Defines whether we want to use a learnable mask embedding (as in the data2vec paper). If set to False,
# the masked parts of the embeddings are replaced with zeros
learnable_mask_embedding = False

# Defines what output of the teacher network Transformer encoder blocks we want to use as our training targets
# (the targets are then normalized and averaged afterwards). There are three possible options, all present in
# the data2vec paper (Table 4):
#   'ff_outputs': The output of the feed-forward (FFN) part of the Transformer encoder
#    'ff_residual_outputs': The output of the FFN of the Transformer encoder after adding the residual
#    'end_of_block': The output of the FFN of the Transformer encoder after the residual connection and LayerNorm
#
# PLEASE NOTE that this hyperparameter applies only to the teacher network (student has value None by default)
target_output_type = 'ff_outputs'

# The number of top Transformer encoder blocks of the teacher network from which the outputs are
# taken, followed by normalization and averaging to construct the training targets
num_top_averaged_teacher_layers = 12

# Defines whether our Transformer encoder is bidirectional (False) or left-to-right (True). In data2vec and
# e.g. BERT, a bidirectional version was used.
only_attend_to_previous_context = False

# Defines whether we want to multiply the embeddings with the square root of the model dimensionality
# before we compute the positional encodings. In the original Transformer paper, this was done to make
# the positional encodings less dominant compared to the embeddings.
use_sqrt = False

# Defines whether we want to apply a linear projection to the embeddings after the positional encoding.
# This projection was used in the original data2vec implementation in Fairseq.
use_embedding_projection = True

# Defines whether we want to add a CLS token to the beginning of the embedding sequence (to be used for
# classification purposes)
include_cls_token = False

# Defines whether we want to use absolute positional encodings (using sinusoids) or relative positional
# encodings (using a CNN layer) for our embeddings. Relative positional encoding was used in the
# data2vec paper, whereas absolute positional encodings were used in the original Transformer paper.
#     Options: 'absolute' or 'relative'
positional_encoding_type = 'relative'

# Defines the dropout of our positional encodings (applies to both the absolute and relative positional
# encodings). In the original Transformer paper, a dropout of 0.1 was used as a regularization technique
dropout_pos_encoding = 0.0

# (Only related to absolute positional encodings) Defines the maximum sequence length in frames
abs_pos_encoding_max_sequence_length = train_sequence_length

# (Only related to relative positional encodings)
rel_pos_encoding_conv_in_dim = encoder_num_output_channels # The input dimensionality of the positional encodings
rel_pos_encoding_conv_out_dim = encoder_num_output_channels # The output dimensionality of the positional encodings
rel_pos_encoding_conv_kernel_size = 13 # The CNN kernel size of the positional encodings
rel_pos_encoding_conv_stride = 1 # The CNN stride of the positional encodings
rel_pos_encoding_conv_padding = 6 # The CNN padding of the positional encodings
rel_pos_encoding_conv_bias = False # The CNN bias of the pos. encodings (not used in wav2vec 2.0 and data2vec papers)
rel_pos_encoding_use_layernorm = True # Defines whether we want to apply LayerNorm after the positional encoding


"""The hyperparameters for our teacher model updater"""

teacher_updater_params = {'initial_tau': 0.9998,
                          'final_tau': 0.99999,
                          'n_tau_updates': 10000,
                          'student_teacher_share_encoder_weights': True,
                          'student_teacher_share_positional_encoder_weights': True}

# The name of the saved dictionary file containing the state of the teacher model updater
teacher_updater_state_dict_savename = 'intermediate_files/teacher_updater_state_dict.p'



"""Other hyperparameters"""

# The hyperparameter dicts for constructing the Transformer models. An empty dictionary will make the model
# to use only default hyperparameters, i.e., the hyperparameters of the original data2vec paper. The teacher
# model is constantly in evaluation mode so we don't care about dropout rates in the hyperparameters. Also,
# since there are no embedding masks used in the teacher model, all hyperparameters related to embedding
# masks have been left out.
transformer_params_student = {'dim_model': embedding_dim,
                              'dim_feedforward': transformer_hidden_dim,
                              'num_heads': num_attention_heads,
                              'num_encoder_layers': num_transformer_encoder_layers,
                              'dropout': dropout_transformer,
                              'transformer_activation_function': transformer_activation_function,
                              'use_embedding_mask': True,
                              'require_same_num_embedding_masks': require_same_num_embedding_masks,
                              'prob_frame_is_start_of_embedding_mask': prob_frame_is_start_of_embedding_mask,
                              'embedding_mask_length_frames': embedding_mask_length_frames,
                              'min_num_mask_start_frames': min_num_mask_start_frames,
                              'learnable_mask_embedding': learnable_mask_embedding,
                              'target_output_type': None,
                              'only_attend_to_previous_context': only_attend_to_previous_context,
                              'use_sqrt': use_sqrt,
                              'use_embedding_projection': use_embedding_projection,
                              'include_cls_token': include_cls_token,
                              'positional_encoding_type': positional_encoding_type,
                              'dropout_pos_encoding': dropout_pos_encoding,
                              'abs_pos_encoding_max_sequence_length': abs_pos_encoding_max_sequence_length,
                              'rel_pos_encoding_conv_in_dim': rel_pos_encoding_conv_in_dim,
                              'rel_pos_encoding_conv_out_dim': rel_pos_encoding_conv_out_dim,
                              'rel_pos_encoding_conv_kernel_size': rel_pos_encoding_conv_kernel_size,
                              'rel_pos_encoding_conv_stride': rel_pos_encoding_conv_stride,
                              'rel_pos_encoding_conv_padding': rel_pos_encoding_conv_padding,
                              'rel_pos_encoding_conv_bias': rel_pos_encoding_conv_bias,
                              'rel_pos_encoding_use_layernorm': rel_pos_encoding_use_layernorm}

transformer_params_teacher = {'dim_model': embedding_dim,
                              'dim_feedforward': transformer_hidden_dim,
                              'num_heads': num_attention_heads,
                              'num_encoder_layers': num_transformer_encoder_layers,
                              'dropout': dropout_transformer,
                              'transformer_activation_function': transformer_activation_function,
                              'use_embedding_mask': False,
                              'learnable_mask_embedding': learnable_mask_embedding,
                              'target_output_type': target_output_type,
                              'only_attend_to_previous_context': only_attend_to_previous_context,
                              'use_sqrt': use_sqrt,
                              'use_embedding_projection': use_embedding_projection,
                              'include_cls_token': include_cls_token,
                              'positional_encoding_type': positional_encoding_type,
                              'dropout_pos_encoding': dropout_pos_encoding,
                              'abs_pos_encoding_max_sequence_length': abs_pos_encoding_max_sequence_length,
                              'rel_pos_encoding_conv_in_dim': rel_pos_encoding_conv_in_dim,
                              'rel_pos_encoding_conv_out_dim': rel_pos_encoding_conv_out_dim,
                              'rel_pos_encoding_conv_kernel_size': rel_pos_encoding_conv_kernel_size,
                              'rel_pos_encoding_conv_stride': rel_pos_encoding_conv_stride,
                              'rel_pos_encoding_conv_padding': rel_pos_encoding_conv_padding,
                              'rel_pos_encoding_conv_bias': rel_pos_encoding_conv_bias,
                              'rel_pos_encoding_use_layernorm': rel_pos_encoding_use_layernorm}


