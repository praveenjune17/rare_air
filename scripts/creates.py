# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf
import os
import shutil
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

assert(str(input('set the last_validation_loss parameter,reply "ok" if set ')) == 'ok'), \
            'Please change the hyper prameters and proceed with training model'
            
if input('Remove summaries dir and tensorboard_logs ? reply "yes or no" ') == 'yes':
  try:
    shutil.rmtree(file_path.summary_write_path)
    shutil.rmtree(file_path.tensorboard_log)
  except FileNotFoundError:
    pass

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
