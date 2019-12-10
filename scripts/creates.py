# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf
import os
import logging
from hyper_parameters import h_parms
from configuration import config
from input_path import file_path


# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(file_path.log_path)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)
log.propagate = False

if not tf.test.is_gpu_available():
    log.info("GPU Not available so Running in CPU")

def check_and_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if config.run_tensorboard:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = file_path.tensorboard_log + current_time + '/train'
    validation_log_dir = file_path.tensorboard_log + current_time + '/validation'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(validation_log_dir)
else:
    train_summary_writer = None
    valid_summary_writer = None
    
# check for folders in the input_path script and create them if they do not exist
# creat vocab file if not exist
for key in file_path.keys():
  if key == 'subword_vocab_path':
    if not os.path.exists(file_path[key]+'.subwords'):
      os.system(os.path.join(os.getcwd(), 'create_tokenizer.py'))
  elif key in ['document', 'summary', 'new_checkpoint_path', 'infer_ckpt_path' ,'log_path']:
    pass
  else:
    if not os.path.exists(file_path[key]):
      check_and_create_dir(file_path[key])
      log.info(f'{key} directory created')

# create metrics dict

monitor_metrics = dict()
monitor_metrics['validation_loss'] = None
monitor_metrics['BERT_f1'] = None
monitor_metrics['ROUGE_f1'] = None
monitor_metrics['validation_accuracy'] = None
monitor_metrics['combined_metric'] = (
                                      monitor_metrics['BERT_f1'], 
                                      monitor_metrics['ROUGE_f1'], 
                                      monitor_metrics['validation_accuracy']
                                      )
assert (config.monitor_metric in monitor_metrics.keys()), f'Available metrics to monitor are {monitor_metrics.keys()}'
assert (tf.reduce_sum(h_parms.combined_metric_weights) == 1), 'weights should sum to 1'
