# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
##########################################################################################
#                       import the below packages
#!pip install tensorflow
#!pip install tensorflow-datasets
#!pip install tensorflow-gan
#!pip install tensorflow-probability
#!pip install tensor2tensor
#!pip install rouge==0.3.2
#!pip install bunch
#!tf_upgrade_v2 --infile c:/Users/pravech3/Summarization/beam_search.py --outfile c:/Users/pravech3/Summarization/beam_search.py
###########################################################################################


import sys
sys.path.insert(0, '/content/drive/My Drive/Client_demo/scripts')


import tensorflow as tf
tf.random.set_seed(100)
import time
import os
import shutil
import tensorflow_datasets as tfds
from preprocess_v2 import create_train_data
from transformer import Transformer, Generator, create_masks
from hyper_parameters import config
from metrics import optimizer, loss_function, get_loss_and_accuracy, write_summary
from input_path import file_path
from create_tokenizer import tokenizer_en
import logging


assert  (str(input('check the log file and set the last validation loss parameter ')) == 'ok'), 'Please change the hyper prameters and proceed with training model'
if input('Remove summaries dir and tensorboard_logs ? reply "yes or no" ') == 'yes':
  try:
    shutil.rmtree(file_path.summary_write_path)
    shutil.rmtree(file_path.tensorboard_log)
  except FileNotFoundError:
    pass


#check for folders in the input_path script and create them if not exisist

for key in file_path.keys():
  if key == 'subword_vocab_path':
    if not os.path.exists(file_path[key]+'.subwords'):
      os.system('/content/drive/My\ Drive/Client_demo/scripts/create_tokenizer.py')
  elif key in ['document', 'summary', 'new_checkpoint_path']:
    pass
  else:
    if not os.path.exists(file_path[key]):
      os.makedirs(file_path[key])
      print(f'{key} directory created')

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('/content/drive/My Drive/Client_demo/created_files/tensorflow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)


if config.run_tensorboard:
    from input_path  import train_summary_writer, valid_summary_writer
else:
    train_summary_writer = None
    valid_summary_writer = None
    
train_dataset, val_dataset = create_train_data()
inp, tar = next(iter(train_dataset))

print()
print(' Input samples to be trained')
print()
for ip,ta in zip(inp.numpy(), tar.numpy()):
  print(tokenizer_en.decode([i for i in ta if i < tokenizer_en.vocab_size]))
  print(tokenizer_en.decode([i for i in ip if i < tokenizer_en.vocab_size]))
  break
  
print()
print()

train_loss, train_accuracy = get_loss_and_accuracy()
validation_loss, validation_accuracy = get_loss_and_accuracy()

transformer = Transformer(
        num_layers=config.num_layers, 
        d_model=config.d_model, 
        num_heads=config.num_heads, 
        dff=config.dff, 
        input_vocab_size=config.input_vocab_size, 
        target_vocab_size=config.target_vocab_size, 
        rate=config.dropout_rate)
generator   = Generator()



# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
]

val_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
]

val_step_with_summary_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar, inp_shape, tar_shape, batch):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  with tf.GradientTape() as tape:
    predictions, attention_weights, dec_output = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    train_variables = transformer.trainable_variables
    tf.debugging.check_numerics(
                                predictions,
                                "Nan's in the transformer predictions"
                                )
    if config.copy_gen:
      predictions = generator(dec_output, predictions, attention_weights, inp, 
                            inp_shape, tar_shape, batch, training=True)
      tf.debugging.check_numerics(
                                predictions,
                                "Nan's in the generator predictions"
                                )
      
    train_variables = train_variables + generator.trainable_variables
    
    loss = loss_function(tar_real, predictions)
  gradients = tape.gradient(loss, train_variables)    
  optimizer.apply_gradients(zip(gradients, train_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, predictions)  

@tf.function(input_signature=val_step_signature)
def val_step(inp, tar, inp_shape, tar_shape, batch):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  
  predictions, attention_weights, dec_output = transformer(inp, tar_inp, 
                               False, 
                               enc_padding_mask, 
                               combined_mask, 
                               dec_padding_mask)
  if config.copy_gen:
    predictions = generator(dec_output, predictions, attention_weights, 
                            inp, inp_shape, tar_shape, batch, training=False)
  loss = loss_function(tar_real, predictions)
  
  validation_loss(loss)
  validation_accuracy(tar_real, predictions)
  

def tf_write_summary(tar_real, predictions, inp, epoch):
  
  return tf.py_function(write_summary, [tar_real, predictions, inp, epoch], Tout=[tf.float32, tf.float32])


@tf.function(input_signature=val_step_with_summary_signature)
def val_step_with_summary(inp, tar, epoch, inp_shape, tar_shape, batch):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  
  predictions, attention_weights, dec_output = transformer(inp, tar_inp, 
                               False, 
                               enc_padding_mask, 
                               combined_mask, 
                               dec_padding_mask)
  if config.copy_gen:
    predictions = generator(dec_output, predictions, attention_weights, 
                            inp, inp_shape, tar_shape, batch, training=False)
  loss = loss_function(tar_real, predictions)
  
  validation_loss(loss)
  validation_accuracy(tar_real, predictions)
  return tf_write_summary(tar_real, predictions, inp[:, 1:], epoch)
  

def calc_validation_loss(validation_dataset, epoch):
  validation_loss.reset_states()
  validation_accuracy.reset_states()
  val_acc = 0
  val_loss = 0
  
  for (batch, (inp, tar)) in enumerate(validation_dataset):
    # calculate rouge for only the first batch
    if batch == 0:
        rouge_score, bert_score = val_step_with_summary(inp, tar, epoch, inp.shape[1], tar.shape[1]-1, inp.shape[0])
    else:
        #rouge = 'Calculated only for first batch'
        val_step(inp, tar, inp.shape[1], tar.shape[1]-1, inp.shape[0])
    val_loss += validation_loss.result()
    val_acc += validation_accuracy.result()
  return (val_acc.numpy()/(batch+1), val_loss.numpy()/(batch+1), rouge_score, bert_score)


def check_ckpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer,
                           generator=generator)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)
    
    # if a checkpoint exists, restore the latest checkpoint.
    if tf.train.latest_checkpoint(checkpoint_path) and not config.from_scratch:
      ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
      print (ckpt_manager.latest_checkpoint, 'checkpoint restored!!')
    else:
        ckpt_manager = tf.train.CheckpointManager(ckpt, file_path.new_checkpoint_path, max_to_keep=20)
        print('Training from scratch')
    return ckpt_manager

checkpoint_path=file_path.old_checkpoint_path
epochs=config.epochs
batch_size=config.batch_size
ckpt_manager = check_ckpt(checkpoint_path)

# get the latest checkpoint from the save directory
if not config.from_scratch:
  latest_ckpt = int(tf.train.latest_checkpoint(checkpoint_path)[-2:]) 
# initialise tolerance to zero
tolerance=0 
for epoch in range(epochs):
  start = time.time()  
  train_loss.reset_states()
  train_accuracy.reset_states()
  validation_loss.reset_states()
  validation_accuracy.reset_states()
  #print(f'Total parameters in this model {train_variables}')
  # inp -> document, tar -> summary
  for (batch, (inp, tar)) in enumerate(train_dataset):
  # the target is shifted right during training hence its shape is subtracted by 1
    #not able to do this inside tf.function since it doesn't allow this operation
    train_step(inp, tar, inp.shape[1], tar.shape[1]-1, inp.shape[0])        
    if batch==0 and epoch ==0:
      log.info(transformer.summary())
      if config.copy_gen:
        log.info(generator.summary())
      print('Time taken to feed the input data to the model {} seconds'.format(time.time()-start))
    if batch % config.print_chks == 0:
      print ('Epoch {} Batch {} Train_Loss {:.4f} Train_Accuracy {:.4f}'.format(
        epoch + 1, batch, train_loss.result(), train_accuracy.result()))
  if config.from_scratch:
    latest_ckpt=epoch+1
  (val_acc, val_loss, rouge_score, bert_score) = calc_validation_loss(val_dataset, epoch+1)
  ckpt_save_path = ckpt_manager.save()
  ckpt_fold, ckpt_string = os.path.split(ckpt_save_path)

  if config.run_tensorboard:
    with train_summary_writer.as_default():
      tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
      tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)

    with valid_summary_writer.as_default():
      tf.summary.scalar('validation_loss', validation_loss.result(), step=epoch)
      tf.summary.scalar('validation_accuracy', validation_accuracy.result(), step=epoch)
      tf.summary.scalar('validation_total_loss', val_acc, step=epoch)
      tf.summary.scalar('validation_total_accuracy', val_loss, step=epoch)
      tf.summary.scalar('ROUGE_score', rouge_score, step=epoch)
      tf.summary.scalar('BERT_score', rouge_score, step=epoch)

  if config.verbose:

    model_metrics = 'Epoch {}, Train Loss: {:.4f}, Train_Accuracy: {:.4f}, \
                    Valid Loss: {:.4f},                   \
                    Valid Accuracy: {:4f},                \
                    ROUGE_score {},                             \
                    BERT_SCORE {}'
    epoch_timing  = 'Time taken for {} epoch : {} secs' 
    checkpoint_details = 'Saving checkpoint for epoch {} at {}'

    print(model_metrics.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result(),
                         val_loss, 
                         val_acc,
                         rouge_score,
                         bert_score))
    print(epoch_timing.format(epoch + 1, time.time() - start))
    print(checkpoint_details.format(epoch+1, ckpt_save_path))

  if (latest_ckpt > config.look_only_after) and (config.last_validation_loss > val_loss):
    
    # reset tolerance to zero if the validation loss decreases before the tolerance threshold
    tolerance=0
    config.last_validation_loss =  val_loss
    ckpt_files_tocopy = [files for files in os.listdir(os.path.split(ckpt_save_path)[0]) if ckpt_string in files]
    log.info(f'Validation loss is {val_loss} so checkpoint files {ckpt_string} will be copied to best checkpoint directory')
    shutil.copy2(os.path.join(ckpt_fold, 'checkpoint'), file_path.best_ckpt_path)
    for files in ckpt_files_tocopy:
        shutil.copy2(os.path.join(ckpt_fold, files), file_path.best_ckpt_path)
  else:
    tolerance+=1

  if config.early_stop and tolerance > config.tolerance_threshold:
    print(f'Early stopping since the validation loss is not decreasing for quite a while and exceeded the tolerance threshold')
    break
  if train_loss.result() == 0:
    print('Train loss reached zero so stopping training')
    break