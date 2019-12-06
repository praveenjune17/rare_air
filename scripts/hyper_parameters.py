# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:09:43 2019

@author: pravech3
"""
from bunch import Bunch


hyp = {
 'accumulation_steps': 3.0,
 'batch_size': 128,
 'beam_size': 2,
 'copy_gen': False,
 'grad_clipnorm':None,
 'd_model': 512,                 # the projected word vector dimension
 'decay_lr': False,              # set learning rate decay
 'dff': 512,                     # feed forward network hidden parameters
 'doc_length': 8096,
 'dropout_rate': 0.1,
 'early_stop' : False,
 'epochs': 50,
 'epsilon_ls': 0.1,              # label_smoothing hyper parameter
 'examples_to_train' : None,
 'from_scratch': False,          # train from scratch
 'input_vocab_size': 8167,       # total vocab size + start and end token
 'l2_norm':0,
 'last_validation_loss' : float('inf'),
 'look_only_after': 10,          # check for decreasing validation loss only after this epoch
 'max_tokens_per_batch': 16618,
 'mean_attention_heads':True,
 'minimum_train_loss': 0.1,
 'num_examples': 20,
 'num_heads': 8,                  # the number of heads in the multi-headed attention unit
 'num_layers': 3,                 # number of transformer blocks
 'print_chks': 50,                 # print training progress per number of batches specified
 'run_tensorboard': False,
 'summ_length': 1340,
 'target_vocab_size': 8167,       # total vocab size + start and end token
 'test_size': 0.05,
 'tolerance_threshold': 20,       # counter which does early stopping
 'verbose': True,
 'write_per_epoch': 5,            # write summary for every specified epoch
 'write_summary_op': True         # write valdiaiton summary to hardisk
 }                                    

config = Bunch(hyp)
