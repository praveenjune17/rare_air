# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:28:12 2019

@author: pravech3
"""
from bunch import Bunch
from hyper_parameters import config
import datetime
import tensorflow as tf
import os

    
def check_and_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

file_path = {
        'csv_path' : "/content/drive/My Drive/Client_demo/input_files/Azure_dataset/Train.csv",
        'subword_vocab_path' : "/content/drive/My Drive/Client_demo/input_files/Vocab_files/vocab_file_summarization_giga",
        'infer_csv_path' : "/content/drive/My Drive/Client_demo/input_files/Azure_dataset/Test.csv",
        'tensorboard_log' : "/content/drive/My Drive/Client_demo/created_files/tensorboard_logs/giga/",
        'summary_write_path' : "/content/drive/My Drive/Client_demo/created_files/summaries/giga/",
        'best_ckpt_path' : "/content/drive/My Drive/Client_demo/created_files/training_summarization_model_ckpts/giga/best_checkpoints/",
        #'old_checkpoint_path' : "/content/drive/My Drive/Client_demo/created_files/training_summarization_model_ckpts/giga/",
        'old_checkpoint_path' : "/content/giga_with_copy_gen/",
        'document' : "Document",
        'summary' :  "Summary"
        
}
file_path = Bunch(file_path)

if config.from_scratch:
    fol_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path['new_checkpoint_path'] = "/content/giga/"+fol_name

if config.run_tensorboard:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = file_path.tensorboard_log + current_time + '/train'
    validation_log_dir = file_path.tensorboard_log + current_time + '/validation'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(validation_log_dir)




check_and_create_dir(file_path.summary_write_path)
check_and_create_dir(file_path.best_ckpt_path)

