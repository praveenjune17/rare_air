# -*- coding: utf-8 -*-
from bunch import Bunch

file_path = {
        'best_ckpt_path' : "/content/drive/My Drive/Client_demo/created_files/training_summarization_model_ckpts/giga/best_checkpoints/",  
        'document' : "Document",
        'infer_csv_path' : "/content/drive/My Drive/Client_demo/input_files/Azure_dataset/Test.csv",
        'infer_ckpt_path' : "/content/drive/My Drive/Client_demo/created_files/training_summarization_model_ckpts/giga/best_checkpoints/ckpt-37",
        'log_path' : "/content/drive/My Drive/Client_demo/created_files/tensorflow.log",
        'checkpoint_path' : "/content/giga_with_copy_gen/",
        'subword_vocab_path' : "/content/drive/My Drive/Client_demo/input_files/Vocab_files/vocab_file_summarization_giga",
        'summary_write_path' : "/content/drive/My Drive/Client_demo/created_files/summaries/giga/",
        'summary' :  "Summary",
        'tensorboard_log' : "/content/drive/My Drive/Client_demo/created_files/tensorboard_logs/giga/",
        'train_csv_path' : "/content/drive/My Drive/Client_demo/input_files/Azure_dataset/Train.csv",
        
}
file_path = Bunch(file_path)

