# A PyTorch implementation of the data2vec algorithm for MAIJU smart jumpsuit data

This repository contains code for pre-training models using the [data2vec algorithm](https://proceedings.mlr.press/v162/baevski22a/baevski22a.pdf) for MAIJU smart jumpsuit data, and also code for fine-tuning the pre-trained models using labeled data. The code has been implemented using PyTorch. For a thorough description of the MAIJU smart jumpsuit, see e.g. https://www.nature.com/articles/s43856-022-00131-6.

**The present data2vec implementation has been used in the following publication:**
[E. Vaaras, M. Airaksinen, S. Vanhatalo, and O. Räsänen, "Evaluation of self-supervised pre-training for automatic infant movement classification using wearable movement sensors," in _Proc. IEEE EMBC 2023_](https://ieeexplore.ieee.org/document/10340118).

If you use the present code or its derivatives, please cite the [repository URL](https://github.com/SPEECHCOG/data2vec_maiju) and/or the [aforementioned publication](https://ieeexplore.ieee.org/document/10340118).

## Requirements
Any `PyTorch` version newer than version 1.9.0 should work fine. You can find out how to install PyTorch here: https://pytorch.org/get-started/locally/. You also need to have `Numpy` and `scikit-learn` installed.

## Repository contents
- `conf_pretrain_maiju_model_data2vec.py`: Example configuration file for data2vec pre-training for MAIJU data, using the same configuration settings that were used in the [present paper](https://arxiv.org/abs/2305.09366).
- `conf_finetune_pretrained_maiju_model.py`: Example configuration file for fine-tuning pre-trained models, using the same configuration settings that were used in the [present paper](https://arxiv.org/abs/2305.09366).
- `pretrain_maiju_model_data2vec.py`: A script for running a single data2vec pre-training and/or using a pre-trained model to extract features.
- `finetune_pretrained_maiju_model.py`: A script for fine-tuning a pre-trained model.
- `maiju_data_loader.py`: A file containing an example data loader for simulated MAIJU data and functions for MAIJU data augmentation.
- `maiju_nn_model.py`: A file containing the neural network model implementations of the present paper.
- `maiju_data2vec_pretrain_test_bench.py`: A superscript that can be used for running and/or evaluating a number of different data2vec pre-trainings.
- `data2vec_ema.py`: A file containing the exponential moving average (EMA) implementation for the data2vec algorithm.
- `transformer_encoder_pytorch.py`: A file containing a slightly modified version of PyTorch's Transformer encoder implementation.
- `py_conf_file_into_text.py`: An auxiliary script for converting _.py_ configuration files into lists of text that can be used for printing or writing the configuration file contents into a text file.

## Examples of how to use the code


### How to run a single data2vec pre-training with one hyperparameter configuration:
You can either use the command
```
python pretrain_maiju_model_data2vec.py
```
or
```
python pretrain_maiju_model_data2vec.py <configuration_file>
```
in order to run a single data2vec pre-training with one hyperparameter configuration. Using the former of these options requires having a configuration file named _conf_pretrain_maiju_model_data2vec.py_ in the same directory as the file _pretrain_maiju_model_data2vec.py_. In the latter option, _<configuration_file>_ is a _.py_ configuration file containing the hyperparameters you want to use during pre-training.

### How to run multiple data2vec pre-trainings with different hyperparameter configurations:
In order to run multiple data2vec pre-trainings with different hyperparameter configurations, you can use the command
```
python maiju_data2vec_pretrain_test_bench.py <dir_to_configuration_files>
```
where _<dir_to_configuration_files>_ is a directory containing at least one _.py_ configuration file. This script runs through the data2vec model pre-training (_pretrain_maiju_model_data2vec.py_) with each of the configuration files located in _<dir_to_configuration_files>_, one at a time.

### How to fine-tune pre-trained models:
You can either use the command
```
python finetune_pretrained_maiju_model.py
```
or
```
python finetune_pretrained_maiju_model.py <configuration_file>
```
in order to fine-tune pre-trained models. Using the former of these options requires having a configuration file named _conf_finetune_pretrained_maiju_model.py_ in the same directory as the file _finetune_pretrained_maiju_model.py_. In the latter option, _<configuration_file>_ is a _.py_ configuration file containing the hyperparameters you want to use during fine-tuning.
