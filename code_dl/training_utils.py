'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import tensorflow as tf
import numpy as np


class TrainingClass():

  def __init__(self):
    self.log = False

  def save(self, step):
    self.ckpt_manager.save()
    with open(self.ckpt_manager.checkpoints[-1] + '.epoch', 'w') as f:
      f.write(str(step))

  def init_tf_summary_writer(self, log_dir: str):
    self.log = True
    self.summary_writer = tf.summary.create_file_writer(log_dir)

  def init_tf_ckpt_manager(self, log_dir: str, model: tf.Module, save_all_ckpts=False, discriminator=None):
    self.log_dir = log_dir
    #enable saving of model weights
    if discriminator is None:
      ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=model)
    else:
      ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=model, discriminator=discriminator)
    if save_all_ckpts:
      max_to_keep = None
    else:
      max_to_keep = 1
    self.ckpt_manager = tf.train.CheckpointManager(ckpt, log_dir + '/tf_ckpts', max_to_keep=max_to_keep)
    if self.ckpt_manager.latest_checkpoint:
      ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
      print('[Latest checkpoint restored!!]')

  def resolve_optimizer(self, name: str, lr: float):
    if name == 'ADAM':
      optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif name == 'RMSProp':
      optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
      raise AttributeError('Unknown optimizer! ' + name)
    return optimizer

  def resolve_loss(self, name: str):
    if name == 'MAE' or name == 'L1':
      self.loss_str = 'L1'
      loss_function = tf.keras.losses.MAE
    elif name == 'MSE' or name == 'L2':
      self.loss_str = 'L2'
      loss_function = tf.keras.losses.MSE
    elif name == 'MAE+MSE' or name == 'MSE+MAE':
      self.loss_str = 'L1+L2'
      loss_function = lambda y_true, y_pred: tf.keras.losses.mae(y_true, y_pred) + tf.keras.losses.mse(y_true, y_pred)
    elif name == 'MAE+DG' or name == 'L1+DG':
      self.loss_str = 'L1+DG'
      loss_function = lambda y_true, y_pred, amplitudes: tf.keras.losses.mae(y_true, y_pred, amplitudes) + depth_gradient_loss(y_pred, amplitudes)
    else:
      raise AttributeError('Unknown loss function!' + name)
    return loss_function


def total_variation_loss(images):
  """ computed the total variation loss
  Args:
    images: A `float` `Tensor` of shape `[B, H, W, C]`.
  Returns:
    A scalar `float` `Tensor`, the loss.
  """
  return tf.reduce_mean(tf.image.total_variation(images))


def depth_gradient_loss(pred, amplitudes):
  """ depth gradient loss
  Args:
    pred: predicted depths, shape `[B, H , W, 1]`
    amplitudes: amplitudes from correlation inputs, shape `[B, H, W, 1]`
  Returns:
    Scalar loss.
  """
  amplitudes = tf.convert_to_tensor(value=amplitudes, dtype=tf.float32)
  dx_d, dy_d = tf.image.image_gradients(pred)
  dx_w, dy_w = tf.image.image_gradients(amplitudes)
  return tf.reduce_mean(
      tf.abs(dx_d) * tf.math.exp(-tf.abs(dx_w)) + \
      tf.abs(dy_d) * tf.math.exp(-tf.abs(dy_w)))


def adversarial_loss(D_real_logits, D_fake_logits):
    """ Least square GAN loss.
    Args:
      D_real_logits: Logits of discriminator on real data.
      D_fake_logits: Logits of discriminator on fake data.
    Returns:
      G_loss: Scalar loss for Generator
      D_Loss: Scalar loss for Discriminator
    """
    D_loss = 0.5 * tf.reduce_mean(tf.square(D_real_logits - 1)) + 0.5 * tf.reduce_mean(tf.square(D_fake_logits))
    G_loss = 0.5 * tf.reduce_mean(tf.square(D_fake_logits - 1))
    return G_loss, D_loss


def adversarial_loss_G(D_fake_logits):
    """ Least square GAN loss.
    Args:
      D_fake_logits: Logits of discriminator on fake data.
    Returns:
      G_loss: Scalar loss for Generator
    """
    G_loss = 0.5 * tf.reduce_mean(tf.square(D_fake_logits - 1))
    return G_loss

