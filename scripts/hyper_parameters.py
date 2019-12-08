# -*- coding: utf-8 -*-
from bunch import Bunch

hyp = {
 'accumulation_steps': 3.0,                                                                                   # TODO
 'batch_size': 128,
 'beam_sizes': [2, 3, 4],        # Used only during inference                                                 #TODO for training
 'dropout_rate': 0.1,
 'epochs': 2,
 'epsilon_ls': 0.1,              # label_smoothing hyper parameter
 'grad_clipnorm':None,
 'l2_norm':0,
 'learning_rate': 3e-4,          # set learning rate decay
 'length_penalty' : 0.6,
 'mean_attention_heads':True,    # if False then the attention weight of the last head will be used
 }                                    

h_parms = Bunch(hyp)
