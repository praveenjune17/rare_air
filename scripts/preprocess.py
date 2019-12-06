# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:28:40 2019

@author: pravech3
"""
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
from hyper_parameters import config
from input_path import file_path
from create_tokenizer import tokenizer_en,create_dataset

#tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(file_path.subword_vocab_path)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def encode(doc, summary):
    lang1 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
    doc.numpy()) + [tokenizer_en.vocab_size+1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
    summary.numpy()) + [tokenizer_en.vocab_size+1]
    return lang1, lang2

    # Set threshold for document and  summary length
def filter_max_length(x, y):
    return tf.logical_and(tf.size(x) <= config.doc_length,
                        tf.size(y) <= config.summ_length)
    
def filter_token_size(x, y):
    return tf.math.less_equal(config.batch_size*(tf.size(x) + tf.size(y)), config.max_tokens_per_batch)


def tf_encode(doc, summary):
    return tf.py_function(encode, [doc, summary], [tf.int64, tf.int64])

def create_train_data(num_examples=None):
    doc, summ = create_dataset(file_path.csv_path, num_examples)
    X_train, X_test, y_train, y_test = train_test_split(doc, summ, test_size=config.test_size, random_state=42)
    print('Training and Test set created')    
    train_examples = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_examples = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    BUFFER_SIZE = len(X_train) 
    train_dataset = train_examples.map(tf_encode, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.filter(filter_token_size)
    total_train_records = sum(1 for l in train_dataset)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE, seed = 100).padded_batch(
        config.batch_size, padded_shapes=([-1], [-1]))
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_examples.map(tf_encode, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.filter(filter_token_size)
    total_val_records = sum(1 for l in val_dataset)
    val_dataset = val_dataset.padded_batch(
        config.batch_size, padded_shapes=([-1], [-1]))
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    print(f'Number of records filtered {BUFFER_SIZE - total_train_records}')
    print(f'Number of records to be trained {total_train_records}')
    print(f'Number of records to be validated {total_val_records}')
    return train_dataset, val_dataset
