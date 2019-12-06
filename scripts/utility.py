# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:54:23 2019

@author: pravech3
"""
import tempfile
import tensorflow as tf

def create_temp_file( text):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with tf.io.gfile.GFile(temp_file.name, "w") as w:
      w.write(text)
    return temp_file.name


def beam_search_train(inp_sentences, beam_size):
  
  start = [tokenizer_en.vocab_size] * inp_sentences.shape[0]
  end = [tokenizer_en.vocab_size+1]
  #pad_token = 0
  #new = []
  #inp_sentences = [tokenizer_en.encode(i) for i in inp_sentences]
  #N = doc_length                                                         # set N as length of the sentence with max len
  #inp_sentences = [[i]*beam_size for i  in inp_sentences]
  encoder_input = tf.tile(inp_sentences, multiples=[beam_size, 1])
  #print(tf.convert_to_tensor(inp_sentences).shape)
  #encoder_input = tf.reshape(tf.convert_to_tensor(inp_sentences), [-1, N]) # (batch_size * beam_size, doc_length)
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
    return (predictions[:,-1:,:]) # (batch_size, 1, target_vocab_size)
  return (beam_search(transformer_query, start, beam_size, summ_length, 
                      target_vocab_size, 0.6, stop_early=True, eos_id=[end]))
  
  
  
