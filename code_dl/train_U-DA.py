'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf
import numpy as np
import time
import argparse
import tensorflow_addons as tfa

# dynamically allocate GPU memory for memory monitoring
if True:
  physical_devices = tf.config.list_physical_devices('GPU')
  try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    assert tf.config.experimental.get_memory_growth(physical_devices[0])
  except ValueError:
    print('Invalid device or cannot modify virtual devices once initialized.')
    pass
  except IndexError:
    print('No GPU found')
    pass

from code_dl.data_ops.Agresti.data_generator import data_generator_Agresti
from code_dl.models import config
from code_dl.training_utils import TrainingClass

# for debugging
# tf.config.run_functions_eagerly(True)


class Training(TrainingClass):

  def domain_adaption(self, model, gen_train, gen_val, gen_real, gen_real_ordered, num_epochs, loss_function, epoch_save=10):

      train_loss_results = [[] for i in range(num_epochs)]
      train_loss_ref = [[] for i in range(num_epochs)]
      test_loss_results = [[] for i in range(num_epochs)]
      test_loss_ref = [[] for i in range(num_epochs)]
      epoch_loss_avg = tf.keras.metrics.Mean()
      epoch_loss_coarse_avg = tf.keras.metrics.Mean()
      epoch_loss_ref_avg = tf.keras.metrics.Mean()
      if self.loss_str != 'L1':
        epoch_l1_loss_avg = tf.keras.metrics.Mean()
        epoch_l1_loss_coarse_avg = tf.keras.metrics.Mean()
      for epoch in range(num_epochs):
        time_epoch_start = time.time()
        epoch_loss_avg.reset_states()
        epoch_loss_coarse_avg.reset_states()
        epoch_loss_ref_avg.reset_states()

        # get predictions on real data every 20 epochs and save as pseudo labels
        depths_pred = []
        if epoch % args.eval_teacher == 0:
          for points, correlations, depths, tof_depths, masks, rays in gen_real_ordered:
            pred, _ = model(rays, correlations, tof_depths, training=False)
            depths_pred.append(pred.numpy())
          gen_real.depths = np.squeeze(np.concatenate(depths_pred, axis=0))[:, :-1]
          gen_real_ordered.on_epoch_end()

        print()
        print('Epoch {:03d} Start (LR: {:.6f})'.format(
          epoch, model.optimizer._decayed_lr(float)))
        print()
        iter_batch = 0
        for points, correlations, depths, tof_depths, masks, rays in gen_train:
          # randomly train on synthtic or real:
          if np.random.choice([0, 1], p=[1 - args.real_ratio, args.real_ratio]):
            # to account for different sizes of data sets
            if iter_batch >= len(gen_real) - 1:
              gen_real.on_epoch_end()
            points, correlations, depths, tof_depths, masks, rays = gen_real[iter_batch % len(gen_real)]
          if args.rotate:
            # additional data augmentation by rotating
            angles = np.random.uniform(-args.rotate, args.rotate, args.batch_size) * np.pi / 180
            correlations = tfa.image.rotate(correlations, angles, interpolation="BILINEAR", fill_mode='reflect').numpy()
            depths = tfa.image.rotate(depths, angles, interpolation="BILINEAR", fill_mode='reflect').numpy()
            tof_depths = tfa.image.rotate(tof_depths, angles, interpolation="BILINEAR", fill_mode='reflect').numpy()
            # masks = tfa.image.rotate(masks, angles, interpolation="BILINEAR", fill_mode='reflect')
            masks = depths != 0
          # compute loss only on valid pixels
          ratio = gen_train.valid_ratio(masks)
          # reference loss
          loss = loss_function(y_pred=masks * tof_depths, y_true=masks * depths) * ratio
          epoch_loss_ref_avg.update_state(loss)
          with tf.GradientTape() as tape:
            pred, pred_coarse = model(rays, correlations, tof_depths, training=True)
            loss = loss_function(y_true=masks * depths, y_pred=masks * pred) * ratio
            loss_coarse = loss_function(y_true=masks * depths, y_pred=masks * pred_coarse) * ratio
            total_loss = loss + loss_coarse
          grads = tape.gradient(total_loss, model.trainable_variables)
          optimizer.apply_gradients(zip(grads, model.trainable_variables))
          epoch_loss_avg.update_state(loss)
          epoch_loss_coarse_avg.update_state(loss_coarse)
          if self.loss_str != 'L1':
            l1_loss = tf.keras.losses.mae(y_true=masks * depths, y_pred=masks * pred) * ratio
            epoch_l1_loss_avg.update_state(l1_loss)
            l1_loss_coarse = tf.keras.losses.mae(y_true=masks * depths, y_pred=masks * pred_coarse) * ratio
            epoch_l1_loss_coarse_avg.update_state(l1_loss_coarse)
          if iter_batch % 5 == 0:
            print(("\r {:03d} / {:03d} " + self.loss_str + "-Loss: {:.4f}, " + self.loss_str + "-Loss_coarse: {:.4f}      ").format(
                iter_batch, len(gen_train),
                epoch_loss_avg.result(),
                epoch_loss_coarse_avg.result()), end="")
          iter_batch += 1
      # TensorBoard summaries
        if self.log:
          with self.summary_writer.as_default():
            tf.summary.scalar(self.loss_str + '_loss_train', epoch_loss_avg.result(), step=epoch)
            tf.summary.scalar(self.loss_str + '_loss_coarse_train', epoch_loss_coarse_avg.result(), step=epoch)
            tf.summary.scalar('LR', model.optimizer._decayed_lr(float), step=epoch)
            if self.loss_str != 'L1':
              tf.summary.scalar('L1_loss_train', epoch_l1_loss_coarse_avg.result(), step=epoch)
              tf.summary.scalar('L1_loss_coarse_train', epoch_l1_loss_coarse_avg.result(), step=epoch)
        train_loss_results[epoch] = epoch_loss_avg.result()
        train_loss_ref[epoch] = epoch_loss_ref_avg.result()
        gen_train.on_epoch_end()

        # --- Validation ---
        epoch_loss_avg.reset_states()
        epoch_loss_coarse_avg.reset_states()
        epoch_loss_ref_avg.reset_states()
        if self.loss_str != 'L1':
          epoch_l1_loss_avg.reset_states()

        print()
        for points, correlations, depths, tof_depths, masks, rays in gen_val:
          ratio = gen_val.valid_ratio(masks)
          # reference loss
          loss = loss_function(y_pred=masks * tof_depths, y_true=masks * depths) * ratio
          epoch_loss_ref_avg.update_state(loss)

          pred, pred_coarse = model(rays, correlations, tof_depths, training=False)
          loss = loss_function(y_true=masks * depths, y_pred=masks * pred) * ratio
          loss_coarse = loss_function(y_true=masks * depths, y_pred=masks * pred_coarse) * ratio
          epoch_loss_avg.update_state(loss)
          epoch_loss_coarse_avg.update_state(loss_coarse)
          if self.loss_str != 'L1':
            l1_loss = tf.keras.losses.mae(y_true=masks * depths, y_pred=masks * pred) * ratio
            epoch_l1_loss_avg.update_state(l1_loss)
            l1_loss_coarse = tf.keras.losses.mae(y_true=masks * depths, y_pred=masks * pred_coarse) * ratio
            epoch_l1_loss_coarse_avg.update_state(l1_loss_coarse)
      # TensorBoard summaries
        if self.log:
          with self.summary_writer.as_default():
            tf.summary.scalar(self.loss_str + '_loss_val', epoch_loss_avg.result(), step=epoch)
            tf.summary.scalar(self.loss_str + '_loss_coarse_val', epoch_loss_coarse_avg.result(), step=epoch)
            if self.loss_str != 'L1':
              tf.summary.scalar('L1_loss_val', epoch_l1_loss_coarse_avg.result(), step=epoch)
              tf.summary.scalar('L1_loss_coarse_val', epoch_l1_loss_coarse_avg.result(), step=epoch)
          if epoch % epoch_save == 0 or epoch == num_epochs - 1:
            self.save(epoch)
        test_loss_results[epoch] = epoch_loss_avg.result()
        test_loss_ref[epoch] = epoch_loss_ref_avg.result()
        # --- End epoch ---
        time_epoch_end = time.time()
        gen_val.on_epoch_end()
        print('Epoch {:03d} Time: {:.4f}s'.format(
              epoch,
              time_epoch_end - time_epoch_start))
        print('Training:   ' + self.loss_str + '-Loss: {:.4f}   Reference: {:.4f}'.format(
            train_loss_results[epoch], train_loss_ref[epoch]))
        print('Validation:   ' + self.loss_str + '-Loss: {:.4f}   Reference: {:.4f}'.format(
            test_loss_results[epoch], test_loss_ref[epoch]))


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '--d', default='data')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--batch_size', '--bs', default=8, type=int)
parser.add_argument('--layer_type', default='MCConv_radu')
parser.add_argument('--feature_type', default='mf_agresti')
parser.add_argument('--freq', '--f', default=[20], nargs='+', type=int, help='List of frequencies, can be [20, 50, 70]')
parser.add_argument('--learning_rate', '--lr', default=0.001, type=float)
parser.add_argument('--lr_decay', '--lr_d', default=0.3, type=float)
parser.add_argument('--lr_d_steps', default=100, type=int)
parser.add_argument('--static_lr', action='store_true')
parser.add_argument('--noise_level', default=0.02, type=float, help='level of noise used during training (relative)')
parser.add_argument('--rotate', default=5, type=float, help='additional augmentation by rotating by small number of degrees')
parser.add_argument('--log_dir', '--l', default='logs/RADU/', type=str)
parser.add_argument('--optimizer', '--o', default='ADAM', type=str)
parser.add_argument('--update_along_z', action='store_true', help='updates depth in global instead of camera coordinates')
parser.add_argument('--loss', default='MAE', type=str)
parser.add_argument('--save_all', action='store_true', help='keep all model ckpts instead of only the latest.')
parser.add_argument('--params', default='v3_avg', help='loads the respective parameters from the model file for the network architecture')
parser.add_argument('--no_log', action='store_true')
parser.add_argument('--no_project_back', action='store_true', help='adds a projection from global to camera space at the end of the network')
parser.add_argument('--patches', action='store_true', help='Train the network on cropped patches of the data')
parser.add_argument('--patch_size', default=128, type=int, help='size (quadratic) of the patches if patched is enabled.')
parser.add_argument('--norm', default=None, type=str, help='can be None, `LN`, `Int`')
parser.add_argument('--use_BN', action='store_true', help='activates batch normalization on the latent features')
parser.add_argument('--use_BN_3D', action='store_true', help='deactivates batch normalization on the latent features for 3D convs')
parser.add_argument('--skip_3D', action='store_true', help='skip connection past 3D convs block')
parser.add_argument('--skip_all', action='store_true', help='skip connections in encoder_decoder architecture')
parser.add_argument('--skip_to_output', action='store_true', help='skip connection from input depth to output, so network is residual')
parser.add_argument('--real_ratio', default=0.5, type=float, help='probability of choosing a real data point')
parser.add_argument('--eval_teacher', default=20, type=int, help='evaluate teacher every x epochs')
args = parser.parse_args()


""" Feature Type """

if args.feature_type == 'mf_agresti':
  num_input_features = 5

""" Datasets """

if 'agresti' in args.data_dir:
  sets = ['S1', 'S3', 'S2']
  if args.patches:
    size = [args.patch_size, args.patch_size]
  else:
    size = [240, 320]
  gen_train = data_generator_Agresti(
      args.batch_size, sets[0], frequencies=args.freq, height=size[0], width=size[1], keepdims=True,
      aug_crop=True, aug_flip=True, aug_rot=False, aug_noise=True, aug_mpi=False, noise_level=args.noise_level, feature_type=args.feature_type, normalize_corr=(args.norm == 'Int'))
  gen_val = data_generator_Agresti(
      args.batch_size, sets[1], frequencies=args.freq, height=size[0], width=size[1], keepdims=True, pad_batches=True, feature_type=args.feature_type, normalize_corr=(args.norm == 'Int'))
  gen_real = data_generator_Agresti(
      args.batch_size, sets[2], frequencies=args.freq, height=size[0], width=size[1], keepdims=True,
      aug_crop=True, aug_flip=True, aug_rot=False, aug_noise=True, aug_mpi=False, noise_level=args.noise_level, feature_type=args.feature_type, normalize_corr=(args.norm == 'Int'))
  gen_real_ordered = data_generator_Agresti(
      args.batch_size, sets[2], frequencies=args.freq, height=240, width=320, keepdims=True, pad_batches=True, feature_type=args.feature_type, normalize_corr=(args.norm == 'Int'),
      shuffle=False)
  config.INPUT_FEATURES_SHAPE[1:] = [size[0], size[1], num_input_features]
 
""" Network Model"""

from code_dl.models import RADU_NN as nn_model

model_args = nn_model.model_params(args.params)

""" Training Hyperparameters"""

# learning rate
learning_rate = args.learning_rate

decay_steps_LR = args.lr_d_steps  # every n-th epoch
if not args.static_lr:
  decay_rate_LR = args.lr_decay
else:
  decay_rate_LR = 1.0
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=decay_steps_LR * len(gen_train),
    decay_rate=decay_rate_LR,
    staircase=True)

training = Training()
loss_function = training.resolve_loss(args.loss)
optimizer = training.resolve_optimizer(args.optimizer, learning_rate)
model = nn_model.mymodel(
    **model_args, layer_type=args.layer_type, batch_size=args.batch_size,
    update_along_rays=not args.update_along_z, normalize_features=(args.norm == 'LN'), project_back=not args.no_project_back,
    use_BN=args.use_BN, use_BN_3D=args.use_BN_3D,
    skip_3D=args.skip_3D, skip_all=args.skip_all, skip_to_output=args.skip_to_output)
model.optimizer = optimizer

""" Training """

if not args.no_log:
  training.init_tf_summary_writer(args.log_dir)
  training.init_tf_ckpt_manager(args.log_dir, model, save_all_ckpts=args.save_all)

print('\n######################')
print('training with ' + args.layer_type + ' network layers')
print('training frames: ' + str(gen_train.epoch_size))
print('validation_frames: ' + str(gen_val.epoch_size))
print('######################\n')

training.domain_adaptation(model, gen_train, gen_val, gen_real, gen_real_ordered, args.num_epochs, loss_function)
