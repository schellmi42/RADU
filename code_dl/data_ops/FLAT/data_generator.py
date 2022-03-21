'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from code_dl.data_ops.phase_unwrapping import phase_unwrapping_two_frequencies_tf
import numpy as np
from code_dl.data_ops.data_generator import data_generator
from code_dl.data_ops.FLAT.data_loader import load_data, load_scene_names
from code_dl.data_ops.FLAT import camera_parameters
from code_dl.data_ops.FLAT.camera_parameters import phase_offsets_neg as phase_offsets


class data_generator_FLAT(data_generator):
  ''' Small generator for batched data.

  Generates in the format [points, correlations, depths, tof_depths, masks, rays]

  Args:
    batch_size: `int` the batch size B,
    data_dir: `str`, can be `'S3', 'S4', 'S5', 'S1_train', 'S1_test'`.
    frequencies: `float`, in MHz.
    height: `int` H, size to crop in x axis.
    input_width: `int` W, size to crop in y axis.
    input_height: `int` , height of images in data set.
    width: `int` , width of images in data set.
    keepdims: `bool`, if `True` return data in shape `[B, H, W, C]`.
      If `False` returns data in shape `[N, C]`. (point cloud format)
    tof_from_aplitudes: `bool`. If `True`, recomputes ToF from augmented amplitude images.
    # projection: `string`, must be `'3D'` `'camera'`. How to project the 2.5D image to 3D.
    noise_level: `float` level of the noise applied to the data in augmentation.
      If `0`, then no noise augmentation is done.
    aug_*: `bool`, to activate augmentation strategies.
      available: crop, flip, rot (rot90), noise
  '''

  def __init__(self,
               batch_size,
               data_set='kinect_train',
               frequencies=[20],
               height=camera_parameters.resolution[0],
               width=camera_parameters.resolution[1],
               fov=camera_parameters.fov_synth,
               keepdims=False,
               aug_noise=False,
               noise_level=0.0,
               aug_crop=False,
               aug_flip=False,
               aug_rot=False,
               aug_mpi=False,
               shuffle=True,
               pad_batches=False,
               normalize_corr=False,
               feature_type='sf_c',
               unwrap_phases=False):
    self.height = height
    self.width = width
    self.keepdims = keepdims
    self.points_per_model = height * width
    self.batch_size = batch_size
    self.fov = fov
    self.feature_type = feature_type
    self.normalize_corr = normalize_corr
    self.data_set = data_set
    if feature_type == 'mf_agresti':
      self.frequencies = camera_parameters.frequencies
    if feature_type == 'mf_su':
      self.frequencies = [camera_parameters.frequencies[0], camera_parameters.frequencies[2]]
    self.unwrap_phases = unwrap_phases

    self.flip_HW = False

    self.scenes = load_scene_names(data_set)
    # if 'test' in data_set:
    #   self.scenes = np.delete(self.scenes, camera_parameters.faulty_test)

    self.input_height = camera_parameters.resolution[0]
    self.input_width = camera_parameters.resolution[1]

    self.aug_noise = aug_noise
    self.noise_level = noise_level
    self.aug_crop = aug_crop
    self.aug_flip = aug_flip
    self.aug_rot = aug_rot
    self.aug_mpi = aug_mpi

    self.epoch_size = len(self.scenes)
    self.num_scenes = len(self.scenes)

    self.sizes = np.ones([batch_size]) * self.points_per_model
    # shuffle data before training
    self.shuffle = shuffle
    self.pad_batches = pad_batches
    self.on_epoch_end()

  def on_epoch_end(self, order=False):
    ''' Shuffles data and resets batch index.
    '''
    if self.shuffle and not order:
      self.order = np.random.permutation(np.arange(0, self.epoch_size))
    else:
      self.order = np.arange(0, self.epoch_size)
    self.index = 0

  def __getitem__(self, index):
    if self.feature_type == 'sf_c':
      return self.get_single_frequency_correlations(index)
    elif self.feature_type == 'sf_d':
      return self.get_single_frequency_depth(index)
    elif self.feature_type == 'sf_cai':
      return self.get_single_frequency_correlations_amplitudes_intensities(index)
    elif self.feature_type == 'mf_c':
      return self.get_multi_frequency_correlations(index)
    elif self.feature_type == 'mf_agresti':
      return self.get_agresti_features(index)
    elif self.feature_type == 'mf_su':
      return self.get_su_features(index)
    elif self.feature_type == 'mf_su_d':
      return self.get_su_features(index)

  def get_agresti_features(self, index):
    """ Returns a batch of data
    """

    # sample points
    indices = \
        self.order[index * self.batch_size:(index + 1) * self.batch_size]
    self.curr_batch_size = len(indices)
    # shape [B, F, H_in, W_in, 1]
    depths, tof_depths, _, amplitudes, _, _ = load_data(self.data_set, self.scenes[indices])
    depths = np.expand_dims(depths, axis=-1)
    # masks = np.expand_dims(masks, axis=-1)
    if self.unwrap_phases:
      tof_depths[:, :, :, 0], _ = phase_unwrapping_two_frequencies_tf(tof_depths[:, :, :, 0], tof_depths[:, :, :, 1], [self.frequencies[0] / 1e3, self.frequencies[1] / 1e3], max_wraps=[3, 0])
      tof_depths[:, :, :, 2], _ = phase_unwrapping_two_frequencies_tf(tof_depths[:, :, :, 2], tof_depths[:, :, :, 1], [self.frequencies[2] / 1e3, self.frequencies[1] / 1e3], max_wraps=[3, 0])

    if self.aug_mpi:
      tof_depths = self.augment_MPI(tof_depths, depths)

    features = np.stack(
      [
        tof_depths[:, :, :, 1],  # tof depth at 30.3MHz
        tof_depths[:, :, :, 2] - tof_depths[:, :, :, 1],  # difference tof depths at 58.8MHz and at 30.3MHz
        tof_depths[:, :, :, 0] - tof_depths[:, :, :, 1],  # difference tof depths at 40MHz and at 30.3MHz
        (amplitudes[:, :, :, 2] / amplitudes[:, :, :, 1]) - 1,  # centered amplitudes ratios between 58.8MHz and 30.3MHz
        (amplitudes[:, :, :, 0] / amplitudes[:, :, :, 1]) - 1,  # centered amplitudes ratios between 40MHz and 30.3MHz
      ], axis=-1)

    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    # 30.3 MHz frequency tof depth
    tof_depths = tof_depths[:, :, :, 1]
    tof_depths = np.expand_dims(tof_depths, axis=-1)
    if self.aug_flip:
      depths, features, tof_depths = self.random_flip_left_right([depths, features, tof_depths])

    if self.aug_rot:
      depths, features, tof_depths  = self.random_rot90([depths, features, tof_depths])

    if self.aug_noise:
      features = self.augment_noise(features, relative=True)
      tof_depths = self.augment_noise(tof_depths, relative=True)

    points, rays = self.project_to_3D(tof_depths)
    # augmentation ##
    if self.aug_crop:
      depths_crop, features_crop, tof_depths_crop, points_crop, rays_crop = self.random_crop([depths, features, tof_depths, points, rays])
      masks = (depths_crop < 1e3) * (depths_crop != 0)
      while np.sum(masks) == 0:
        # avoid cutting only background
        depths_crop, features_crop, tof_depths_crop, points_crop, rays_crop = self.random_crop([depths, features, tof_depths, points, rays])
        masks = (depths_crop < 1e3) * (depths_crop != 0)
      depths, features, tof_depths, points, rays = depths_crop, features_crop, tof_depths_crop, points_crop, rays_crop
    else:
      depths, features, tof_depths, points, rays = self.crop_center([depths, features, tof_depths, points, rays])
      masks = (depths < 1e3) * (depths != 0)

    if self.pad_batches and self.curr_batch_size < self.batch_size:
      bs_diff = self.batch_size - self.curr_batch_size
      points = np.pad(points, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      depths = np.pad(depths, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      features = np.pad(features, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      tof_depths = np.pad(tof_depths, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

    if not self.keepdims:
      points = np.reshape(points, [self.curr_batch_size * self.points_per_model, 3])
      depths = np.reshape(depths, [self.curr_batch_size * self.points_per_model, 1])
      features = np.reshape(features, [self.curr_batch_size * self.points_per_model, 5])
      tof_depths = np.reshape(tof_depths, [self.curr_batch_size * self.points_per_model, 1])

    # clean invalid values
    depths *= masks

    rays = np.repeat(rays, self.batch_size, axis=0)

    return points, features, depths, tof_depths, masks, rays

  def get_single_frequency_depth(self, index):
    """ Returns a batch of data
    """

    # sample points
    indices = \
        self.order[index * self.batch_size:(index + 1) * self.batch_size]
    self.curr_batch_size = len(indices)
    # shape [B, F, H_in, W_in, 1]
    depths, tof_depths, _, _, _, _ = load_data(self.data_set, self.scenes[indices])
    tof_depths = tof_depths[:, :, :, 1]  # get 30.3MHz depth.
    tof_depths = np.expand_dims(tof_depths, axis=-1)
    depths = np.expand_dims(depths, axis=-1)
    # augmentation ##
    if self.aug_crop:
      tof_depths, depths = self.random_crop([tof_depths, depths])
    else:
      tof_depths, depths = self.crop_center([tof_depths, depths])

    if self.aug_flip:
      tof_depths, depths = self.random_flip_left_right([tof_depths, depths])

    if self.aug_rot:
      tof_depths, depths = self.random_rot90([tof_depths, depths])

    if self.aug_noise:
      tof_depths = self.augment_noise(tof_depths)
    points, rays = self.project_to_3D(tof_depths)

    if not self.keepdims:
      points = np.reshape(points, [self.batch_size * self.points_per_model, 3])
      depths = np.reshape(depths, [self.batch_size * self.points_per_model, 1])
      tof_depths = np.reshape(tof_depths, [self.batch_size * self.points_per_model, 1])

    # clean invalid values
    masks = (depths < 1e3) * (depths != 0)
    depths *= masks

    rays = np.repeat(rays, self.batch_size, axis=0)

    return points, tof_depths, depths, tof_depths, masks, rays

  def get_su_features(self, index):
    """ Returns a batch of data
    """

    # sample points
    indices = \
        self.order[index * self.batch_size:(index + 1) * self.batch_size]
    self.curr_batch_size = len(indices)
    # shape [B, F, H_in, W_in, 1]
    depths, tof_depths, correlations, amplitudes, _, _ = load_data(self.data_set, self.scenes[indices])
    depths = np.expand_dims(depths, axis=-1)

    features = np.stack(
      [
        correlations[:, :, :, 0, 0],  # correlations at 40MHz phase 0
        correlations[:, :, :, 0, 1],  # correlations at 40MHz phase 120
        correlations[:, :, :, 2, 0],  # correlations at 58.8MHz phase 0
        correlations[:, :, :, 2, 1],  # correlations at 58.8MHz phase 120
      ], axis=-1)

    # 30.3MHz ToF depth
    tof_depths = tof_depths[:, :, :, 1]
    tof_depths = np.expand_dims(tof_depths, axis=-1)
    if self.aug_flip:
      depths, tof_depths, features, amplitudes = self.random_flip_left_right([depths, tof_depths, features, amplitudes])

    if self.aug_rot:
      depths, tof_depths, features, amplitudes  = self.random_rot90([depths, tof_depths, features, amplitudes])

    if self.aug_noise:
      features = self.augment_noise(features, relative=True)
      tof_depths = self.augment_noise(tof_depths, relative=True)

    points, rays = self.project_to_3D(tof_depths)
    # augmentation ##
    if self.aug_crop:
      depths, tof_depths, features, amplitudes, points, rays = self.random_crop([depths, tof_depths, features, amplitudes, points, rays])
    else:
      depths, tof_depths, features, amplitudes, points, rays = self.crop_center([depths, tof_depths, features, amplitudes, points, rays])

    if self.pad_batches and self.curr_batch_size < self.batch_size:
      bs_diff = self.batch_size - self.curr_batch_size
      points = np.pad(points, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      amplitudes = np.pad(amplitudes, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      depths = np.pad(depths, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      features = np.pad(features, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      tof_depths = np.pad(tof_depths, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

    if not self.keepdims:
      points = np.reshape(points, [self.curr_batch_size * self.points_per_model, 3])
      depths = np.reshape(depths, [self.curr_batch_size * self.points_per_model, 1])
      features = np.reshape(features, [self.curr_batch_size * self.points_per_model, 5])
      tof_depths = np.reshape(tof_depths, [self.curr_batch_size * self.points_per_model, 1])

    # clean invalid values
    masks = (depths < 1e3) * (depths != 0)
    depths *= masks

    rays = np.repeat(rays, self.batch_size, axis=0)
    if self.feature_type == 'mf_su_d':
      # return with tof depth
      return points, features, depths, tof_depths, masks, rays
    else:
      # return with amplitudes for total variation loss
      return points, features, depths, amplitudes, masks, rays
