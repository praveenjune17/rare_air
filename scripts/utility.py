# -*- coding: utf-8 -*-

import tempfile
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocess import create_train_data

train_dataset, val_dataset, num_of_train_examples = create_train_data(shuffle=False)
  
def create_temp_file( text):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with tf.io.gfile.GFile(temp_file.name, "w") as w:
      w.write(text)
    return temp_file.name


# histogram of tokens per batch_size
# arg1 :- must be a padded_batch dataset
def hist_tokens_per_batch(split=='valid'):
    x=[]
    tf_dataset = train_dataset if split == 'train' else val_dataset
    for (_, (i, j)) in enumerate(tf_dataset):
        x.append(tf.size(i) + tf.size(j))

    plt.hist(x, bins=20)
    plt.xlabel('Total tokens per batch')
    plt.ylabel('No of times')
    plt.show()

# histogram of Summary_lengths
# arg1 :- must be a padded_batch dataset
def hist_summary_length(split=='valid'):
    x=[]
    tf_dataset = train_dataset if split == 'train' else val_dataset
    for (_, (i, j)) in enumerate(tf_dataset.unbatch()):
        x.append(tf.shape(j)[1])

    plt.hist(x, bins=20)
    plt.xlabel('Summary_lengths')
    plt.ylabel('No of times')
    plt.show()
    
def beam_search_train(inp_sentences, beam_size):
  
  start = [tokenizer_en.vocab_size] * inp_sentences.shape[0]
  end = [tokenizer_en.vocab_size+1]
  encoder_input = tf.tile(inp_sentences, multiples=[beam_size, 1])
  def transformer_query(output):

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
          encoder_input, output)
    predictions, attention_weights, dec_output = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask
                                                   )
    if predictions:
      predictions = generator(dec_output, predictions, attention_weights, encoder_input, 
                                    inp_sentences.shape[1], output.shape[1], inp_sentences.shape[0], beam_size, False)

    # select the last sequence
    # (batch_size, 1, target_vocab_size)
    return (predictions[:,-1:,:]) 
  return (beam_search(transformer_query, start, beam_size, summ_length, 
                      target_vocab_size, 0.6, stop_early=True, eos_id=[end]))
  
  
  
