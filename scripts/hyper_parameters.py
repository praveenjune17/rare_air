# -*- coding: utf-8 -*-
from bunch import Bunch
import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

hyp = {
 'accumulation_steps': 3.0,                                                                                   # TODO
 'batch_size': 64,
 'beam_sizes': [2, 3, 4],        # Used only during inference                                                 #TODO for training
 'combined_metric_weights': [0.4, 0.3, 0.3], #(bert_score, rouge, validation accuracy)
 'dropout_rate': 0.0,
 'epochs': 20,
 'epsilon_ls': 0.0,              # label_smoothing hyper parameter
 'grad_clipnorm':None,
 'l2_norm':0,
 'learning_rate': 3e-4,          # set learning rate decay
 'length_penalty' : 1,
 'mean_attention_heads':True,    # if False then the attention weight of the last head will be used
 }                                    

h_parms = Bunch(hyp)
