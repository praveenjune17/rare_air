# -*- coding: utf-8 -*-
from bunch import Bunch
import os

#core_path = '/content/drive/My Drive/Client_demo/'  #G_drive
core_path = '/content/'  
dataset_name = 'gigaword'
file_path = {
        'best_ckpt_path' : os.path.join(core_path, "created_files/training_summarization_model_ckpts/"+dataset_name+"/best_checkpoints/"),  
        'infer_csv_path' : os.path.join(core_path, "input_files/Azure_dataset/Test.csv"),
        'infer_ckpt_path' : os.path.join(core_path, "created_files/training_summarization_model_ckpts/"+dataset_name+"/best_checkpoints/ckpt-37"),
        'log_path' : os.path.join(core_path, "created_files/tensorflow.log"),
        'checkpoint_path' : os.path.join(core_path, dataset_name+"_checkpoints"),
        'subword_vocab_path' : os.path.join(core_path, "input_files/vocab_file_summarization_"+dataset_name),
        'summary_write_path' : os.path.join(core_path, "created_files/summaries/"+dataset_name+"/"),
        'tensorboard_log' : os.path.join(core_path, "created_files/tensorboard_logs/"+dataset_name+"/"),
        'train_csv_path' : os.path.join(core_path, "input_files/Azure_dataset/Train.csv"),
        
}
file_path = Bunch(file_path)

