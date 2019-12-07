# -*- coding: utf-8 -*-

import pandas as pd
import os
import tensorflow_datasets as tfds
from input_path import file_path 
from creates import log

def create_dataframe(path, num_examples):
    df = pd.read_csv(path)
    assert len([i.capitalize() for i in df.columns if i.lower() in ['document', 'summary']]) == 2, 'Incorrect column names'
    df.columns = [i.capitalize() for i in df.columns if i.lower() in ['document', 'summary']]
    df = df[:num_examples]
    assert not df.isnull().any().any(), 'dataset contains  nans'
    return (df[file_path.document].values, df[file_path.summary].values)

if os.path.exists(file_path.subword_vocab_path+'.subwords'):
  tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(file_path.subword_vocab_path)
  log.info('Subword vocab file loaded')
  log.info(f'Set the vocab size in hyperparameters file as {tokenizer_en.vocab_size+2}')
else:
  log.info('Vocab file not available so building it from the training set')
  doc, summ = create_dataframe(file_path.train_csv_path, None)
  tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
              (doc for doc, _ in zip(doc, summ)), target_vocab_size=2**13)
  tokenizer_en.save_to_file(file_path.subword_vocab_path)
  log.info('subword vocab file created')
  log.info(f'Set the vocab size in hyperparameters file as {tokenizer_en.vocab_size+2}')