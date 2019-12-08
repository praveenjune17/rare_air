# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.random.set_seed(100)
import time
import os
import shutil
import tensorflow_datasets as tfds
from preprocess import create_train_data
from transformer import transformer, generator, create_masks
from hyper_parameters import h_parms
from configuration import config
from metrics import optimizer, loss_function, get_loss_and_accuracy, tf_write_summary, calc_validation_loss
from input_path import file_path
from creates import log, train_summary_writer, valid_summary_writer
from create_tokenizer import tokenizer_en
from local_tf_ops import train_step, val_step_with_summary, check_ckpt_dir

model_metrics = 'Epoch {}, Train Loss: {:.4f}, Train_Accuracy: {:.4f}, \
                     Valid Loss: {:.4f},                                   \
                     Valid Accuracy: {:4f},                                \
                     ROUGE_score {},                                       \
                     BERT_SCORE {}'
epoch_timing  = 'Time taken for {} epoch : {} secs' 
checkpoint_details = 'Saving checkpoint for epoch {} at {}'
batch_zero = 'Time taken to feed the input data to the model {} seconds'

train_dataset, val_dataset, num_of_train_examples = create_train_data()
calc_loss, calc_accuracy = get_loss_and_accuracy()

if config.show_detokenized_samples:
  inp, tar = next(iter(train_dataset))
  for ip,ta in zip(inp.numpy(), tar.numpy()):
    log.info(tokenizer_en.decode([i for i in ta if i < tokenizer_en.vocab_size]))
    log.info(tokenizer_en.decode([i for i in ip if i < tokenizer_en.vocab_size]))
    break
  
# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr, latest_ckpt = check_ckpt_dir(file_path.checkpoint_path)


for epoch in range(h_parms.epochs):
  start = time.time()  
  calc_loss.reset_states()
  calc_accuracy.reset_states()
  for (batch, (inp, tar)) in enumerate(train_dataset):
  # the target is shifted right during training hence its shape is subtracted by 1
  # not able to do this inside tf.function since it doesn't allow this operation
    train_step(inp, tar, inp.shape[1], tar.shape[1]-1, inp.shape[0], True)        
    batch_train_loss = calc_loss.result()
    batch_train_acc = calc_accuracy.result()
    if batch==0 and epoch ==0:
      log.info(transformer.summary())
      if config.copy_gen:
        log.info(generator.summary())
      log.info(batch_zero.format(time.time()-start))
    if batch % config.print_chks == 0:
      log.info('Epoch {} Batch {} Train_Loss {:.4f} Train_Accuracy {:.4f}'.format(
                                                                                  epoch + 1, 
                                                                                  batch, 
                                                                                  batch_train_loss, 
                                                                                  batch_train_acc)
                                                                                  )
    if config.run_tensorboard:
        with train_summary_writer.as_default():
          tf.summary.scalar('train_loss', batch_train_loss, step=epoch)
          tf.summary.scalar('train_accuracy', batch_train_acc, step=epoch)
  data_after_filter = ((batch-1)*h_parms.batch_size)/num_of_train_examples
  log.info(f'Atleast {data_after_filter*100}% of training data was used')
  (val_acc, val_loss, rouge_score, bert_score) = calc_validation_loss(val_dataset, epoch+1)
  if config.run_tensorboard:
    with valid_summary_writer.as_default():  
      tf.summary.scalar('total_validation_loss', val_acc, step=epoch)
      tf.summary.scalar('total_validation_total', val_loss, step=epoch)
  ckpt_save_path = ck_pt_mgr.save()
  ckpt_fold, ckpt_string = os.path.split(ckpt_save_path)
  latest_ckpt+=1
  if config.verbose:
    log.info(model_metrics.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result(),
                         val_loss, 
                         val_acc,
                         rouge_score,
                         bert_score))
    log.info(epoch_timing.format(epoch + 1, time.time() - start))
    log.info(checkpoint_details.format(epoch+1, ckpt_save_path))

  if (latest_ckpt > config.monitor_only_after) and (config.last_validation_loss > val_loss):
    
    # reset tolerance to zero if the validation loss decreases before the tolerance threshold
    config.init_tolerance=0
    config.last_validation_loss =  val_loss
    ckpt_files_tocopy = [files for files in os.listdir(os.path.split(ckpt_save_path)[0]) \
                         if ckpt_string in files]
    log.info(f'Validation loss is {val_loss} so checkpoint files {ckpt_string}           \
             will be copied to best checkpoint directory')
    shutil.copy2(os.path.join(ckpt_fold, 'checkpoint'), file_path.best_ckpt_path)
    for files in ckpt_files_tocopy:
        shutil.copy2(os.path.join(ckpt_fold, files), file_path.best_ckpt_path)
  else:
    config.init_tolerance+=1

  if config.init_tolerance > config.tolerance_threshold:
    log.warning('Tolerance exceeded')
  if config.early_stop and config.init_tolerance > config.tolerance_threshold:
    log.info(f'Early stopping since the validation loss exceeded the tolerance threshold')
    break
  if train_loss.result() == 0:
    log.info('Train loss reached zero so stopping training')
    break