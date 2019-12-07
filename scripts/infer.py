# -*- coding: utf-8 -*-

import tensorflow as tf
tf.random.set_seed(100)
import time
import os
import shutil
import tensorflow_datasets as tfds
from create_tokenizer import tokenizer_en
from transformer import Transformer, Generator, create_masks
from hyper_parameters import h_parms
from configuration import config
from metrics import optimizer, loss_function, get_loss_and_accuracy, tf_write_summary
from input_path import file_path
from beam_search import beam_search
from create_tokenizer import create_dataframe
from preprocess import infer_data_from_df

def restore_chkpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(
                               transformer=transformer,
                               optimizer=optimizer,
                               generator=generator
                               )
    
    
    assert tf.train.latest_checkpoint(os.path.split(checkpoint_path)[0]), 'checkpoint not available'
    ckpt.restore(checkpoint_path)
    print (checkpoint_path, 'checkpoint restored!!')

def beam_search_eval(document, beam_size):
  
  start = [tokenizer_en.vocab_size] 
  end = [tokenizer_en.vocab_size+1]
  encoder_input = tf.tile(document, multiples=[beam_size, 1])
  batch, inp_shape = encoder_input.shape
  def transformer_query(output):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                                                                     encoder_input, 
                                                                     output
                                                                     )
    predictions, attention_weights, dec_output = transformer(
                                                             encoder_input, 
                                                             output,
                                                             False,
                                                             enc_padding_mask,
                                                             combined_mask,
                                                             dec_padding_mask
                                                            )
    
    if config.copy_gen:	
      predictions = generator(
                              dec_output, 
                              predictions, 
                              attention_weights, 	
                              encoder_input, 
                              inp_shape, 
                              output.shape[-1], 	
                              batch, 
                              False
                              )

    # (batch_size, 1, target_vocab_size)
    return (predictions[:,-1:,:])  
  return beam_search(
                     transformer_query, 
                     start, 
                     beam_size, 
                     config.summ_length, 
                     config.input_vocab_size, 
                     h_parms.length_penalty, 
                     stop_early=True, 
                     eos_id=[end]
                    )

transformer = Transformer(
                          num_layers=config.num_layers, 
                          d_model=config.d_model, 
                          num_heads=config.num_heads, 
                          dff=config.dff, 
                          input_vocab_size=config.input_vocab_size, 
                          target_vocab_size=config.target_vocab_size, 
                          rate=config.dropout_rate
                          )
generator   = Generator()

#Restore the model's checkpoints
restore_chkpt(file_path.infer_ckpt_path)
infer_dataset = infer_data_from_df()

def run_infer(beam_sizes_to_try = h_parms.beam_sizes):
    for beam_size in beam_sizes_to_try:
      for doc_id, document in enumerate(infer_dataset, 1):
        start_time = time.time()
        # translated_output_temp[0] (batch, beam_size, summ_length+1)
        translated_output_temp = beam_search_eval(document, beam_size)
        for top_sentence_ids in (translated_output_temp[0][:,0,:]):
          log.info('Original document: {}'.format(sample_document))
          log.info('Predicted summary: {}'.format(tokenizer_en.decode([j for j in top_sentence_ids if j < tokenizer_en.vocab_size])))
        log.info(f'time to process document {doc_id} : {time.time()-start_time}')
      log.info(f'############ Beam size {beam_size} completed #########')


