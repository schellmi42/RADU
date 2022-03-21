'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf
import os
import numpy as np
import time
import argparse

# dynamically allocate GPU memory for memory monitoring
if False:
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

from code_dl.data_ops.data_generator import data_generator
from code_dl.data_ops.Agresti.data_generator import data_generator_Agresti
from code_dl.data_ops.FLAT.data_generator import data_generator_FLAT
from code_dl.models import config
from code_dl.training_utils import TrainingClass

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


tf.config.run_functions_eagerly(True)


class Evaluation(TrainingClass):

  def __init__(self, save_imgs, log_dir, data_dir, save_raw):
    super(Evaluation).__init__()
    self.save_imgs = save_imgs
    self.save_raw = save_raw
    self.imgs_dir = log_dir + '/images' + save_raw * '_raw' + '/'  + data_dir + '/'
    if save_imgs or save_raw:
      print('\n[saving images to ' + self.imgs_dir + ']\n')
      if not os.path.exists(self.imgs_dir):
        os.makedirs(self.imgs_dir)
    if args.save_point_clouds:
      self.points_dir = log_dir + '/points/'  + data_dir + '/'
      print('\n[saving points to ' + self.points_dir + ']\n')
      if not os.path.exists(self.points_dir):
        os.makedirs(self.points_dir)

    self.imgs_id = 0

  def evaluate(self, model, gen_test, loss_function):

      epoch_loss_avg = tf.keras.metrics.Mean()
      epoch_loss_coarse_avg = tf.keras.metrics.Mean()
      epoch_loss_ref_avg = tf.keras.metrics.Mean()

      # --- Testing ---
      epoch_loss_avg.reset_states()
      epoch_loss_coarse_avg.reset_states()
      epoch_loss_ref_avg.reset_states()

      preds_list = []
      depths_list = []
      masks_list = []

      print()
      time_epoch_start = time.time()
      for points, correlations, depths, tof_depths, masks, rays in gen_test:
        # compute loss only on valid pixels
        ratio = gen_test.valid_ratio(masks)
        depths_list.append(depths)
        masks_list.append(masks)
        # reference loss
        loss = loss_function(y_pred=masks * tof_depths, y_true=masks * depths) * ratio
        epoch_loss_ref_avg.update_state(loss)
        # loss on prediction
        pred, pred_coarse = model(rays, correlations, tof_depths, training=False)
        preds_list.append(pred)
        loss = loss_function(y_true=masks * depths, y_pred=masks * pred) * ratio
        loss_coarse = loss_function(y_true=masks * depths, y_pred=masks * pred_coarse) * ratio
        epoch_loss_avg.update_state(loss)
        epoch_loss_coarse_avg.update_state(loss_coarse)
        if self.save_imgs and self.imgs_id < 20:
          if args.feature_type == 'mf_agresti':
            self.save_images_agresti(correlations, depths, tof_depths, pred, pred_coarse, masks)
          else:
            self.save_images_corr(correlations, depths, tof_depths, pred, pred_coarse, masks)
        elif self.save_raw:
            self.save_images_raw(depths, tof_depths, pred, pred_coarse, masks)
        if args.save_point_clouds:
            self.save_point_clouds(depths, pred, masks, rays)

      # --- End epoch ---
      time_epoch_end = time.time()
      print('Time. {:.2f}'.format(time_epoch_end - time_epoch_start))
      print('Testing:   L1-Loss: {:.4f}, L1-Loss coarse: {:.4f},   Reference: {:.4f}'.format(
          100 * epoch_loss_avg.result(), 100 * epoch_loss_coarse_avg.result(), 100 * epoch_loss_ref_avg.result()))
      with open(args.result_f, 'a') as f:
        f.write(args.data_dir + '   {:.4f}'.format(100 * epoch_loss_avg.result()) + '   ' + args.log_dir + '\n')

  def save_images_corr(self, correlations, depths, tof_depths, predictions, predictions_coarse, masks):
    batch_size = len(correlations)

    for b in range(batch_size):
      _ = plt.figure(figsize=(15, 10))
      # correlations
      max_grey = np.max(correlations[b, :, :, :4])
      num_corrs = correlations.shape[-1]
      for i in range(min([num_corrs, 4])):
        ax = plt.subplot(340 + i + 1)
        ax.imshow(correlations[b, :, :, i], cmap=plt.get_cmap('Greys'), vmin=0, vmax=max_grey)
        ax.set_title('correlation_phase_' + str(i))
      # GT depth
      max_depth = np.max(depths[b]) * 2
      cmap = plt.get_cmap('jet')
      ax = plt.subplot(345)
      imshow_1 = ax.imshow(np.squeeze(depths[b]), cmap=cmap, vmin=0, vmax=max_depth)
      divider = make_axes_locatable(ax)
      cax = divider.append_axes("right", size="5%", pad=0.05)
      plt.colorbar(imshow_1, cax=cax)
      ax.set_title('GT depth')
      # ToF depth
      ax = plt.subplot(346)
      ax.imshow(np.squeeze(tof_depths[b]), cmap=cmap, vmin=0, vmax=max_depth)
      ax.set_title('ToF depth | MAE: {:.2f}'.format(np.mean(np.abs(tof_depths[b] - depths[b]))))
      # prediction
      ax = plt.subplot(347)
      ax.imshow(np.squeeze(predictions[b]), cmap=cmap, vmin=0, vmax=max_depth)
      ax.set_title('predicted depth')
      ax = plt.subplot(348)
      error = np.squeeze(predictions[b] * masks[b] - depths[b])
      imshow = ax.imshow(error, cmap=cmap, vmin=-clip_plt_error, vmax=clip_plt_error)
      ax.set_title('error of prediction | MAE: {:.2f}'.format(np.mean(np.abs(error))))
      divider = make_axes_locatable(ax)
      cax = divider.append_axes("right", size="5%", pad=0.05)
      plt.colorbar(imshow, cax=cax)
      # coarse prediction
      ax = plt.subplot(3, 4, 11)
      ax.imshow(np.squeeze(predictions_coarse[b]), cmap=cmap, vmin=0, vmax=max_depth)
      ax.set_title('predicted depth coarse')
      ax = plt.subplot(3, 4, 12)
      error = np.squeeze(predictions_coarse[b] * masks[b] - depths[b])
      imshow = ax.imshow(error, cmap=cmap, vmin=-clip_plt_error, vmax=clip_plt_error)
      ax.set_title('error of prediction coarse | MAE: {:.2f}'.format(np.mean(np.abs(error))))

      # plt.show()

      plt.savefig(self.imgs_dir + str(self.imgs_id).zfill(3))
      plt.close()
      self.imgs_id += 1

  def save_images_agresti(self, features, depths, tof_depths, predictions, predictions_coarse, masks):
    batch_size = len(features)

    for b in range(batch_size):

      valid_pixel_rate = 100 * np.prod(depths[b].shape) / np.sum(masks[b])

      _ = plt.figure(figsize=(15, 10))
      # input features
      titles = ['d_20 - d_60', 'd_50 - d_60', 'A_20 / A_60', 'A_50 / A_60']
      for i in range(4):
        ax = plt.subplot(340 + i + 1)
        ax.imshow(features[b, :, :, i + 1], cmap=plt.get_cmap('Greys'))
        ax.set_title(titles[i])
      # GT depth
      max_depth = np.max(depths[b]) * 2
      cmap = plt.get_cmap('jet')
      ax = plt.subplot(345)
      imshow_1 = ax.imshow(np.squeeze(depths[b]), cmap=cmap, vmin=0, vmax=max_depth)
      divider = make_axes_locatable(ax)
      cax = divider.append_axes("right", size="5%", pad=0.05)
      plt.colorbar(imshow_1, cax=cax)
      ax.set_title('GT depth')
      # ToF depth
      ax = plt.subplot(346)
      ax.imshow(np.squeeze(tof_depths[b]), cmap=cmap, vmin=0, vmax=max_depth)
      ax.set_title('ToF depth | MAE: {:.2f}'.format(valid_pixel_rate * np.mean(masks[b] * np.abs(tof_depths[b] - depths[b]))))
      # prediction
      ax = plt.subplot(347)
      ax.imshow(np.squeeze(predictions[b]), cmap=cmap, vmin=0, vmax=max_depth)
      ax.set_title('predicted depth')
      ax = plt.subplot(348)
      error = (predictions[b] * masks[b] - depths[b]).numpy()
      mae = valid_pixel_rate * np.mean(np.abs(error))
      error[~masks[b]] = -1
      imshow = ax.imshow(np.squeeze(error), cmap=cmap, vmin=-clip_plt_error, vmax=clip_plt_error)
      ax.set_title('error of prediction | MAE: {:.2f}'.format(mae))
      divider = make_axes_locatable(ax)
      cax = divider.append_axes("right", size="5%", pad=0.05)
      plt.colorbar(imshow, cax=cax)
      # coarse prediction
      ax = plt.subplot(3, 4, 11)
      ax.imshow(np.squeeze(predictions_coarse[b]), cmap=cmap, vmin=0, vmax=max_depth)
      ax.set_title('predicted depth coarse')
      ax = plt.subplot(3, 4, 12)
      error = (predictions_coarse[b] * masks[b] - depths[b]).numpy()
      mae = valid_pixel_rate * np.mean(np.abs(error))
      error[~masks[b]]  = -1
      #max_error = np.max(np.abs(error))
      imshow = ax.imshow(np.squeeze(error), cmap=cmap, vmin=-clip_plt_error, vmax=clip_plt_error)
      ax.set_title('error of prediction coarse | MAE: {:.2f}'.format(mae))

      # plt.show()

      plt.savefig(self.imgs_dir + str(self.imgs_id).zfill(3))
      plt.close()
      self.imgs_id += 1

  def save_data_raw(self, features, depths, tof_depths, predictions, predictions_coarse, masks, rays, model):
    batch_size = len(depths)

    for b in range(batch_size):
      max_depth = np.max(depths[b]) * 2
      valid_pixel_rate = 100 * np.prod(depths[b].shape) / np.sum(masks[b])

      error = (predictions[b] * masks[b] - depths[b]).numpy()
      mae = valid_pixel_rate * np.mean(np.abs(error))
      error[~masks[b]] = -1

      error_coarse = (predictions_coarse[b] * masks[b] - depths[b]).numpy()
      mae_coarse = valid_pixel_rate * np.mean(np.abs(error))
      error_coarse[~masks[b]]  = -1

      error_tof = (tof_depths[b] * masks[b] - depths[b])
      mae_tof = valid_pixel_rate * np.mean(np.abs(error_tof))
      error_tof[~masks[b]] = -1

      titles = ['gt', 'tof', 'pred', 'pred_coarse', 'error', 'error_coarse']
      imgs = [depths[b], tof_depths[b], predictions[b], predictions_coarse[b], error, error_coarse]
      mae_strings = ['', 'MAE: {:.2f}'.format(mae_tof), '', '', 'MAE: {:.2f}'.format(mae), 'MAE: {:.2f}'.format(mae_coarse)]
      v_mins = [0, 0, 0, 0, -clip_plt_error, -clip_plt_error]
      v_maxs = [max_depth, max_depth, max_depth, max_depth, clip_plt_error, clip_plt_error]
      for img, title, mae_string, v_min, v_max in zip(imgs, titles, mae_strings, v_mins, v_maxs):
        filename = self.imgs_dir + str(self.imgs_id).zfill(3) + '_' + title + mae_string + '.png'
        plt.imsave(fname=filename, arr=np.squeeze(img), cmap='jet', format='png', vmin=v_min, vmax=v_max)
        plt.close()

      titles = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
      for i in range(4):
        filename = self.imgs_dir + str(self.imgs_id).zfill(3) + '_' + titles[i] + '.png'
        plt.imsave(fname=filename, arr=np.squeeze(features[b, :, :, i + 1]), cmap='Greys', format='png')
      from mpl_toolkits.mplot3d import Axes3D
      sampled_rays, _ = tf.linalg.normalize(model.ray_downscaling(rays), axis=-1)
      sampled_depths = model.downscaling(tof_depths)
      sampled_points = (sampled_rays * sampled_depths).numpy()
      sampled_points = sampled_points.reshape([-1, 3])
      print(len(sampled_points))
      with open(self.imgs_dir + str(self.imgs_id).zfill(3) + '_''points.txt', 'w') as outFile:
        for point in sampled_points:
          outFile.write(str(point[0]) + ',' + str(point[1]) + ',' + str(point[2]) + '\n')

      ax = plt.subplot(111, projection='3d')
      ax.scatter3D(sampled_points[:, 1], sampled_points[:, 2], -sampled_points[:, 0])
      ax.set_title('3D projection')
      ax.view_init(0, 270)
      plt.show()

      self.imgs_id += 1

  def save_point_clouds(self, depths, predictions, masks, rays):
    batch_size = len(depths)

    for b in range(batch_size):

      errors = (predictions[b] * masks[b] - depths[b]).numpy()
      errors[~masks[b]] = -1

      points = (predictions * rays).numpy()
      points = points.reshape([-1, 3])
      errors = errors.reshape([-1])
      with open(self.points_dir + str(self.imgs_id).zfill(3) + '_''points_RADU.txt', 'w') as outFile:
        for (point, error) in zip(points, errors):
          outFile.write(str(point[0]) + ',' + str(point[1]) + ',' + str(point[2]) + ',' + str(error) + '\n')

      # ax = plt.subplot(121, projection='3d')
      # ax.scatter3D(points[::20, 1], points[::20, 2], -points[::20, 0])
      # ax.set_title('3D projection')
      # ax.view_init(0, 270)
      # ax = plt.subplot(122)
      # ax.imshow(np.squeeze(predictions[b]))
      # plt.show()

      points = depths * rays
      points = points.reshape([-1, 3])
      errors = errors.reshape([-1])
      with open(self.points_dir + str(self.imgs_id).zfill(3) + '_''points_gt.txt', 'w') as outFile:
        for (point, error) in zip(points, errors):
          outFile.write(str(point[0]) + ',' + str(point[1]) + ',' + str(point[2]) + '\n')

      self.imgs_id += 1

  def save_images_raw(self, depths, tof_depths, predictions, predictions_coarse, masks):
    batch_size = len(depths)

    for b in range(batch_size):
      max_depth = np.max(depths[b]) * 2
      valid_pixel_rate = 100 * np.prod(depths[b].shape) / np.sum(masks[b])

      error = (predictions[b] * masks[b] - depths[b] * masks[b]).numpy()
      mae = valid_pixel_rate * np.mean(np.abs(error))
      error[~masks[b]] = -1

      error_coarse = (predictions_coarse[b] * masks[b] - depths[b] * masks[b]).numpy()
      mae_coarse = valid_pixel_rate * np.mean(np.abs(error_coarse))
      error_coarse[~masks[b]]  = -1

      error_tof = (tof_depths[b] * masks[b] - depths[b] * masks[b])
      mae_tof = valid_pixel_rate * np.mean(np.abs(error_tof))
      error_tof[~masks[b]] = -1

      titles = ['gt', 'tof', 'pred', 'pred_coarse', 'error', 'error_coarse', 'tof_error']
      imgs = [depths[b], tof_depths[b], predictions[b], predictions_coarse[b], error, error_coarse, error_tof]
      mae_strings = ['', '', '', '', 'MAE_{:.2f}'.format(mae), 'MAE_{:.2f}'.format(mae_coarse), 'MAE_{:.2f}'.format(mae_tof)]
      v_mins = [0, 0, 0, 0, -clip_plt_error, -clip_plt_error, -clip_plt_error]
      v_maxs = [max_depth, max_depth, max_depth, max_depth, clip_plt_error, clip_plt_error, clip_plt_error]
      for img, title, mae_string, v_min, v_max in zip(imgs, titles, mae_strings, v_mins, v_maxs):
        filename = self.imgs_dir + str(self.imgs_id).zfill(3) + '_' + title + mae_string + '.png'
        plt.imsave(fname=filename, arr=np.squeeze(img), cmap='jet', format='png', vmin=v_min, vmax=v_max)
        plt.close()

      self.imgs_id += 1

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '--d', default='data_agresti/S4')
parser.add_argument('--batch_size', '--bs', default=1, type=int)
parser.add_argument('--layer_type', default='MCConv_radu')
parser.add_argument('--feature_type', default='sf_c')
parser.add_argument('--freq', '--f', default=[20], nargs='+', type=int, help='List of frequencies, can be [20, 50, 70]')
parser.add_argument('--noise_level', default=0.02, type=float, help='level of noise used (relative)')
parser.add_argument('--log_dir', '--l', default='logs/RADU/', type=str)
parser.add_argument('--optimizer', '--o', default='ADAM', type=str)
parser.add_argument('--update_along_z', action='store_true')
parser.add_argument('--loss', default='MAE', type=str)
parser.add_argument('--save_imgs', action='store_true')
parser.add_argument('--save_raw_imgs', action='store_true')
parser.add_argument('--save_point_clouds', action='store_true')
parser.add_argument('--params', default='v3_avg')
parser.add_argument('--no_log', action='store_true')
parser.add_argument('--no_project_back', action='store_true', help='adds a projection from global to camera space at the end of the network')
parser.add_argument('--norm', default=None, type=str, help='can be None, `LN`, `Int`')
parser.add_argument('--use_BN', action='store_true', help='deactivates batch normalization on the latent features')
parser.add_argument('--use_BN_3D', action='store_true', help='deactivates batch normalization on the latent features for 3D convs')
parser.add_argument('--skip_3D', action='store_true', help='skip connection past 3D convs block')
parser.add_argument('--skip_to_output', action='store_true', help='skip connection from input depth to output, so network is residual')
parser.add_argument('--skip_all', action='store_true', help='skip connections in encoder_decoder architecture')
parser.add_argument('--unwrap', '--PU', action='store_true', help='phase unwraps tof depths before feeding to network')
parser.add_argument('--result_f', default='results.txt', help='file where the results are saved.')
args = parser.parse_args()

""" Feature Type """

if args.feature_type == 'sf_c':
  num_input_features = 4
elif args.feature_type == 'sf_cai':
  num_input_features = 6
elif args.feature_type == 'mf_c':
  num_input_features = 4 * len(args.freq)
elif args.feature_type == 'mf_agresti':
  num_input_features = 5
elif args.feature_type == 'mf_su_d':
  num_input_features = 4
  
""" Datasets """

if 'agresti' in args.data_dir:
  data_set = args.data_dir.split('/')[-1]
  print(data_set)
  gen_test = data_generator_Agresti(
      args.batch_size, data_set, frequencies=args.freq, height=240, width=320, keepdims=True, shuffle=False, feature_type=args.feature_type, normalize_corr=(args.norm == 'Int'))
  config.INPUT_FEATURES_SHAPE[1:] = [240, 320, num_input_features]
  from code_dl.models import RADU_NN as nn_model
  clip_plt_error = 0.1
elif 'FLAT' in args.data_dir:
  size = [424, 512]
  gen_test = data_generator_FLAT(
      args.batch_size, 'kinect_test', frequencies=args.freq, height=size[0], width=size[1], keepdims=True,
      feature_type=args.feature_type, shuffle=False, unwrap_phases=args.unwrap)
  config.INPUT_FEATURES_SHAPE[1:] = [424, 512, num_input_features]
  from code_dl.models import RADU_NN as nn_model
  clip_plt_error = 0.4
else:  # own data RaspberryPi
  from code_dl.models import RADU_NN as nn_model

  gen_test = data_generator(
      args.batch_size, 'test', frequencies=args.freq, height=512, width=512, keepdims=True,  # full_scene_in_epoch=True,
      aug_noise=True, noise_level=args.noise_level, feature_type=args.feature_type, shuffle=False, unwrap_phases=args.unwrap)
  clip_plt_error = 0.6

  
""" Network Model"""

model_args = nn_model.model_params(args.params)

model = nn_model.mymodel(
    **model_args, layer_type=args.layer_type, batch_size=args.batch_size,
    update_along_rays=not args.update_along_z, normalize_features=(args.norm == 'LN'), project_back=not args.no_project_back,
    use_BN=args.use_BN, use_BN_3D=args.use_BN_3D,
    skip_all=args.skip_all, skip_3D=args.skip_3D, skip_to_output=args.skip_to_output)

""" Testing """
testing = Evaluation(args.save_imgs, args.log_dir, args.data_dir, save_raw=args.save_raw_imgs)
# restore weights
testing.init_tf_ckpt_manager(args.log_dir, model)
optimizer = testing.resolve_optimizer(args.optimizer, 0)
model.optimizer = optimizer
loss_function = testing.resolve_loss(args.loss)

print('\n######################')
print('testing with ' + args.layer_type + ' network layers')
print('testing frames: ' + str(gen_test.epoch_size))
print('######################\n')

testing.evaluate(model, gen_test, loss_function)
