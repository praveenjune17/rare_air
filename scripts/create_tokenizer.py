import sys
sys.path.insert(0, '/content/drive/My Drive/Client_demo/scripts')

from input_path import file_path 
import pandas as pd
import os
import tensorflow_datasets as tfds

def create_dataset(path, num_examples):
    df = pd.read_csv(path)
    assert len([i.capitalize() for i in df.columns if i.lower() in ['document', 'summary']]) == 2, 'Incorrect column names'
    df.columns = [i.capitalize() for i in df.columns if i.lower() in ['document', 'summary']]
    df = df[:num_examples]
    assert not df.isnull().any().any(), f'examples contains  nans please check the input dataset'
    return (df[file_path.document].values, df[file_path.summary].values)

if os.path.exists(file_path.subword_vocab_path+'.subwords'):
  tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(file_path.subword_vocab_path)
  print('Subword vocab file loaded')
  print(f'Set the vocab size in hyperparameters file as {tokenizer_en.vocab_size+2}')
else:
  print('Vocab file not available so building from the training set it')
  doc, summ = create_dataset(file_path.csv_path, None)
  tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
              (doc for doc, _ in zip(doc, summ)), target_vocab_size=2**13)
  tokenizer_en.save_to_file(file_path.subword_vocab_path)
  print('subword vocab file created')
  print(f'Set the vocab size in hyperparameters file as {tokenizer_en.vocab_size+2}')