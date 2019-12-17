# -*- coding: utf-8 -*-

import tempfile
import tensorflow as tf
import matplotlib
import tensorflow_datasets as tfds
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
import numpy as np
import pandas as pd
import time
from hyper_parameters import h_parms
from create_tokenizer import tokenizer_en
from configuration import config
from preprocess import tf_encode
  
def create_temp_file( text):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with tf.io.gfile.GFile(temp_file.name, "w") as w:
      w.write(text)
    return temp_file.name


# histogram of tokens per batch_size
# arg1 :- must be a padded_batch dataset
def hist_tokens_per_batch(tf_dataset, num_of_examples, samples_to_try=0.1, split='valid'):
    x=[]
    samples_per_batch = int((samples_to_try*(num_of_examples))//h_parms.batch_size)
    tf_dataset = tf_dataset.padded_batch(h_parms.batch_size, padded_shapes=([-1], [-1]))
    tf_dataset = tf_dataset.take(samples_per_batch).cache()
    tf_dataset = tf_dataset.prefetch(buffer_size=samples_per_batch)
    print(f'creating histogram for {samples_per_batch} samples')
    for (_, (i, j)) in (enumerate(tf_dataset)):
        x.append(tf.size(i) + tf.size(j))
    print(f'Recommendeded tokens per batch based on {samples_per_batch*h_parms.batch_size} samples is {tf.math.reduce_mean(x)}')
    print(f'Min tokens per batch {tf.math.reduce_min(x)}')
    print(f'Max tokens per batch {tf.math.reduce_max(x)}')
    plt.hist(x, bins=20)
    plt.xlabel('Total tokens per batch')
    plt.ylabel('No of times')
    plt.savefig('#_of_tokens per batch in '+split+' set.png')
    plt.close() 

# histogram of Summary_lengths
# arg1 :- must be a padded_batch dataset
def hist_summary_length(tf_dataset, num_of_examples, samples_to_try=0.1, split='valid'):
    summary=[]
    document=[]
    samples = int((samples_to_try*(num_of_examples)))
    tf_dataset = tf_dataset.take(samples).cache()
    tf_dataset = tf_dataset.prefetch(buffer_size=samples)
    print(f'creating histogram for {samples} samples')
    for (doc, summ) in (tf_dataset):
        summary.append(summ.shape[0])
        document.append(doc.shape[0])
    print(f'Recommendeded summary length based on {samples} samples is {tf.math.reduce_mean(summary)}')
    print(f'Recommendeded document length based on {samples} samples is {tf.math.reduce_mean(document)}')
    print(f'Min summary length {tf.math.reduce_min(summary)}')
    print(f'Max summary length {tf.math.reduce_max(summary)}')
    print(f'Min document length {tf.math.reduce_min(document)}')
    print(f'Max document length {tf.math.reduce_max(document)}')
    plt.hist([summary, document], alpha=0.5, bins=20, label=['summary', 'document'] )
    plt.xlabel('lengths of document and summary')
    plt.ylabel('Counts')
    plt.legend(loc='upper right')
    plt.savefig(split+'_lengths of document and summary.png')
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
  #create histogram for summary_lengths and token 
  examples, metadata = tfds.load(config.tfds_name, with_info=True, as_supervised=True)
  val_dataset = examples['validation'].map(tf_encode, num_parallel_calls=2)
  train_dataset = examples['train'].map(tf_encode, num_parallel_calls=2)
  test_dataset = examples['test'].map(tf_encode, num_parallel_calls=2)
  valid_buffer_size = metadata.splits['validation'].num_examples
  test_buffer_size = metadata.splits['test'].num_examples
  train_buffer_size = metadata.splits['train'].num_examples
  datasets = [train_dataset, val_dataset, test_dataset]
  counts   = [train_buffer_size, valid_buffer_size, test_buffer_size]
  splits   = ['train', 'valid', 'test']
  percentage_of_samples = 0.1
  start = time.time()
  for dataset,count,split in zip(datasets, counts, split):
    hist_summary_length(dataset, count, percentage_of_samples, split)  
    print(f'time taken to calculate length of {split} is {time.time() - start}')
    hist_tokens_per_batch(dataset, count, percentage_of_samples, split)
    print(f'time taken to calculate tokens_per_batch of {split} is {time.time() - start}')
  
if config.show_detokenized_samples:
  inp, tar = next(iter(train_dataset))
  for ip,ta in zip(inp.numpy(), tar.numpy()):
    print(tokenizer_en.decode([i for i in ta if i < tokenizer_en.vocab_size]))
    print(tokenizer_en.decode([i for i in ip if i < tokenizer_en.vocab_size]))
    break
