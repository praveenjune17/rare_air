# -*- coding: utf-8 -*-
from bunch import Bunch


hyp = {
 'copy_gen':True,
 'doc_length': 8096,
 'd_model': 512,                  # the projected word vector dimension
 'dff': 512,                      # feed forward network hidden parameters
 'early_stop' : False,
 'init_tolerance':0,
 'input_vocab_size': 8167,        # total vocab size + start and end token
 'last_validation_loss' : float('inf'),
 'monitor_only_after': 10,        # monitor the validation loss only after this epoch                                         # Generalaise monitor metric #TODO
 'max_tokens_per_batch': 16618,
 'minimum_train_loss': 0.1,
 'num_examples_to_train': None,   #If None then all the examples in the dataset will be used to train
 'num_examples_to_infer': None,
 'num_heads': 8,                  # the number of heads in the multi-headed attention unit
 'num_layers': 3,                 # number of transformer blocks
 'print_chks': 50,                # print training progress per number of batches specified
 'run_tensorboard': False,
 'show_detokenized_samples' : False,
 'summ_length': 1340,
 'target_vocab_size': 8167,       # total vocab size + start and end token
 'test_size': 0.05,               # used when the input is supplied as a csv file
 'tolerance_threshold': 20,       # counter which does early stopping
 'use_tfds' : True,               # use tfds datasets as input to the model (default :- Gigaword )
 'verbose': True,
 'write_per_epoch': 5,            # write summary for every specified epoch
 'write_summary_op': True         # write valdiaiton summary to hardisk
 }                                    

config = Bunch(hyp)
