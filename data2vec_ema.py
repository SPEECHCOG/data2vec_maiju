# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The exponential moving average (EMA) implementation of the present paper. The code
is partially adapted from the original Fairseq implementation:
  https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/ema_module.py

"""

import numpy as np
from copy import deepcopy
import sys


class update_teacher_weights:
    """Exponential Moving Average of Fairseq Models"""

    def __init__(self, initial_teacher_model_encoder, initial_teacher_model_transformer, initial_tau=0.999,
                 final_tau=0.9999, n_tau_updates=30000, tau_start_index = 0, additional_skip_keys = [],
                 student_teacher_share_encoder_weights=True, student_teacher_share_positional_encoder_weights=True):

        self.teacher_model_encoder_weights = deepcopy(initial_teacher_model_encoder.state_dict())
        self.teacher_model_transformer_weights = deepcopy(initial_teacher_model_transformer.state_dict())
        
        self.student_teacher_share_encoder_weights = student_teacher_share_encoder_weights
        self.student_teacher_share_positional_encoder_weights = student_teacher_share_positional_encoder_weights
        
        # self.skip_keys is a list containing keys whose associated weights are taken straight from the
        # student network to the teacher network without any modification at all
        if student_teacher_share_positional_encoder_weights:
            self.skip_keys = ['positional_encoder', 'mask_embedding']
        else:
            self.skip_keys = ['mask_embedding']
        
        self.additional_skip_keys = additional_skip_keys
        
        if isinstance(additional_skip_keys, list):
            for key in additional_skip_keys:
                self.skip_keys.append(key)
        elif isinstance(additional_skip_keys, str):
            self.skip_keys.append(additional_skip_keys)
        else:
            sys.exit('The argument "additional_skip_keys" can only be a string or a list containing strings!')
        
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.n_tau_updates = n_tau_updates
        self.tau_index = tau_start_index
        self.taus = np.linspace(initial_tau, final_tau, num=n_tau_updates)
        

    
    def add_skip_keys(self, new_skip_keys):
        if isinstance(new_skip_keys, list):
            for key in new_skip_keys:
                self.skip_keys.append(key)
        elif isinstance(new_skip_keys, str):
            self.skip_keys.append(new_skip_keys)
        else:
            sys.exit('You can only add a string or a list containing strings to self.skip_keys!')
    
    def get_teacher_updater_state(self):
        updater_state = {'initial_tau': self.initial_tau,
                         'final_tau': self.final_tau,
                         'n_tau_updates': self.n_tau_updates,
                         'tau_start_index': self.tau_index,
                         'additional_skip_keys': self.additional_skip_keys,
                         'student_teacher_share_encoder_weights': self.student_teacher_share_encoder_weights,
                         'student_teacher_share_positional_encoder_weights': self.student_teacher_share_positional_encoder_weights}
            
        return updater_state
    
    def check_if_key_in_skip_keys(self, key):
        for skip_key in self.skip_keys:
            if skip_key in key:
                return True
        return False
    
    def update_weights(self, student_model, tau, mode):
        """Code adapted from Fairseq:
            https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/ema_module.py
        
        """
        
        teacher_state_dict = {}
        if mode == 'encoder':
            teacher_params = self.teacher_model_encoder_weights
        else:
            teacher_params = self.teacher_model_transformer_weights
        
        for key, param in student_model.named_parameters():
            if isinstance(param, dict):
                continue
            try:
                teacher_param = teacher_params[key]
            except KeyError:
                teacher_param = (param.float().clone() if param.ndim == 1 else deepcopy(param))
                teacher_params[key] = teacher_param

            if param.shape != teacher_param.shape:
                raise ValueError(f'Incompatible tensor shapes between param and teacher_param {param.shape} vs. {teacher_param.shape}')
            
            if self.check_if_key_in_skip_keys(key) or not param.requires_grad:
                teacher_params[key].copy_(param.data)
                teacher_param = teacher_params[key]
            else:
                teacher_param.mul_(tau)
                
                # teacher_param = teacher_param + (1 - tau) * param.data
                teacher_param.add_(param.data, alpha = (1 - tau))

            teacher_state_dict[key] = teacher_param

        for key, param in student_model.named_buffers():
            teacher_state_dict[key] = param
        
        return teacher_state_dict

    def update(self, student_model_encoder, student_model_transformer):
        "Update the weights of the teacher model."
        
        if self.tau_index >= self.n_tau_updates:
            tau = self.taus[-1]
        else:
            tau = self.taus[self.tau_index]
            self.tau_index += 1
        
        if self.student_teacher_share_encoder_weights:
            teacher_encoder_weight_dict = student_model_encoder.state_dict()
        else:
            teacher_encoder_weight_dict = self.update_weights(student_model_encoder, tau, 'encoder')
        teacher_transformer_weight_dict = self.update_weights(student_model_transformer, tau, 'transformer')
        
        self.teacher_model_encoder_weights = deepcopy(teacher_encoder_weight_dict)
        self.teacher_model_transformer_weights = deepcopy(teacher_transformer_weight_dict)
        
        return teacher_encoder_weight_dict, teacher_transformer_weight_dict, tau
        