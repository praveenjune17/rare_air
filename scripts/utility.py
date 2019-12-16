# -*- coding: utf-8 -*-

import tempfile
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocess import create_train_data
from hyper_parameters import h_parms
from create_tokenizer import tokenizer_en
from configuration import config

  
def create_temp_file( text):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with tf.io.gfile.GFile(temp_file.name, "w") as w:
      w.write(text)
    return temp_file.name


# histogram of tokens per batch_size
# arg1 :- must be a padded_batch dataset
def hist_tokens_per_batch(tf_dataset, split='valid'):
    x=[]
    count=0
    samples = int(np.ceil(config.samples_to_use/h_parms.batch_size))
    for (_, (i, j)) in enumerate(tf_dataset):
        count+=1
        x.append(tf.size(i) + tf.size(j))
        if count==samples:
          break
    plt.hist(x, bins=20)
    plt.xlabel('Total tokens per batch')
    plt.ylabel('No of times')
    plt.savefig('#_of_tokens per batch in '+split+' set.png')
    plt.close() 

# histogram of Summary_lengths
# arg1 :- must be a padded_batch dataset
def hist_summary_length(train_dataset, val_dataset, split='valid'):
    x=[]
    count=0
    tf_dataset = train_dataset if split == 'train' else val_dataset
    for (doc, summ) in tf_dataset.unbatch():
        count+=1
        # don't count padded zeros as part of summary length
        x.append(len([i for i in summ if i]))
        if count==config.samples_to_use:
          break
    plt.hist(x, bins=20)
    plt.xlabel('Summary_lengths')
    plt.ylabel('No of times')
    plt.savefig(split+'_Summary_lengths.png')
    plt.close() 

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
 


if config.create_hist:
  train_dataset, val_dataset, num_of_train_examples, num_of_valid_examples = create_train_data(shuffle=False)
  #create histogram for summary_lengths and token 
  hist_summary_length(train_dataset, val_dataset, 'valid')
  hist_summary_length(train_dataset, val_dataset, 'train')
  hist_tokens_per_batch(train_dataset, val_dataset, 'valid')
  hist_tokens_per_batch(train_dataset, val_dataset, 'train')
  
  if config.show_detokenized_samples:
    inp, tar = next(iter(train_dataset))
    for ip,ta in zip(inp.numpy(), tar.numpy()):
      print(tokenizer_en.decode([i for i in ta if i < tokenizer_en.vocab_size]))
      print(tokenizer_en.decode([i for i in ip if i < tokenizer_en.vocab_size]))
      break


