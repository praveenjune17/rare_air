import tensorflow as tf
from transformer import transformer, generator, create_masks
from metrics import optimizer, loss_function, get_loss_and_accuracy, tf_write_summary
from configuration import config

signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
    tf.TensorSpec(shape=(None), dtype=tf.bool),
]
train_step_signature = signature
val_step_with_summary_signature = signature
val_step_with_summary_signature[-1] = tf.TensorSpec(shape=(None), dtype=tf.int32)

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar, inp_shape, tar_shape, batch, apply_grad):
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
  if apply_grad:
      gradients = tape.gradient(loss, train_variables)    
      optimizer.apply_gradients(zip(gradients, train_variables))
  
  calc_loss(loss)
  calc_accuracy(tar_real, predictions)  
  
@tf.function(input_signature=val_step_with_summary_signature)
def val_step_with_summary(inp, tar, epoch, inp_shape, tar_shape, batch):
  calc_loss.reset_states()
  calc_accuracy.reset_states()
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
  calc_loss(loss)
  calc_accuracy(tar_real, predictions)
  return tf_write_summary(tar_real, predictions, inp[:, 1:], epoch)
  
def check_ckpt_dir(checkpoint_path):
    ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer,
                           generator=generator)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)
    if tf.train.latest_checkpoint(checkpoint_path):
      ckpt.restore(ckpt_manager.latest_checkpoint)
      log.info(ckpt_manager.latest_checkpoint +' restored')
      latest_ckpt = int(ckpt_manager.latest_checkpoint[-2:])
    else:
        latest_ckpt=0
        log.info('Training from scratch')
    return (ckpt_manager, latest_ckpt)

