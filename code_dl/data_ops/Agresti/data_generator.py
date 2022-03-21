'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from code_dl.data_ops.geom_ops_numpy import reconstruct_correlations, correlation2depth
import numpy as np
from code_dl.data_ops.data_generator import data_generator
from code_dl.data_ops.Agresti.data_loader import load_data
from code_dl.data_ops.Agresti import camera_parameters


class data_generator_Agresti(data_generator):
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
               data_set='S3',
               frequencies=[20],
               height=camera_parameters.resolution[0],
               width=camera_parameters.resolution[1],
               fov=camera_parameters.fov,
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
               feature_type='sf_c'):
    self.height = height
    self.width = width
    self.keepdims = keepdims
    self.points_per_model = height * width
    self.batch_size = batch_size
    self.fov = fov
    self.feature_type = feature_type
    self.normalize_corr = normalize_corr

    if self.feature_type == 'mf_agresti':
      # frequencies used by agresti
      self.frequencies = [20, 50, 60]
      self.max_frequency = 60
    elif self.feature_type == 'sf_c' or self.feature_type == 'sf_cai':
      self.max_frequency = frequencies[0]
      self.frequencies = frequencies
    elif self.feature_type == 'mf_c':
      self.frequencies = frequencies
      self.max_frequencies = max(frequencies)

    self.flip_HW = False
    self.depths, self.tof_depths, self.amplitudes, self.intensities, self.PU = load_data(data_set=data_set, frequencies=self.frequencies, squeeze=True)

    if self.feature_type == 'sf_c' or self.feature_type == 'sf_cai':
      self.correlations = reconstruct_correlations(self.amplitudes, self.intensities, self.tof_depths, self.max_frequency / 1e3)
    elif self.feature_type == 'mf_c':
      correlations = []
      for i in range(len(self.frequencies)):
        correlations.append(reconstruct_correlations(self.amplitudes[:, i], self.intensities[:, i], self.tof_depths[:, i], self.frequencies[i] / 1e3))
      self.correlations = np.stack(correlations, axis=1)

    if self.normalize_corr:
      # normalize correlations by intensity
      intensities_non_zeros = np.ones_like(self.intensities)
      intensities_non_zeros[self.intensities != 0] = self.intensities[self.intensities != 0]
      self.correlations = self.correlations / np.expand_dims(intensities_non_zeros, axis=-1)

    # print('WARNING', np.sum(self.amplitudes[:, 2] == 0))
    self.input_height = self.tof_depths.shape[-2]
    self.input_width = self.tof_depths.shape[-1]
    # pad if input is missing some lines (real data misses one line in height)
    self.do_pad = (self.input_height < self.height) or (self.input_width < self.width)

    # self.projection = projection
    self.aug_noise = aug_noise
    self.noise_level = noise_level
    self.aug_crop = aug_crop
    self.aug_flip = aug_flip
    self.aug_rot = aug_rot
    self.aug_mpi = aug_mpi

    self.epoch_size = len(self.tof_depths)

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
    elif self.feature_type == 'sf_cai':
      return self.get_single_frequency_correlations_amplitudes_intensities(index)
    if self.feature_type == 'mf_c':
      return self.get_multi_frequency_correlations(index)
    elif self.feature_type == 'mf_agresti':
      return self.get_agresti_features(index)

  def get_single_frequency_correlations(self, index):
    """ Returns a batch of data
    """

    # sample points
    indices = \
        self.order[index * self.batch_size:(index + 1) * self.batch_size]
    self.curr_batch_size = len(indices)
    # shape [B, H_in, W_in, 4]
    correlations = self.correlations[indices]

    if self.depths is not None:
      depths = np.expand_dims(self.depths[indices], axis=-1)
    else:
      shape = correlations.shape
      depths = np.zeros([*shape[:-1], 1])
    tof_depths = np.expand_dims(self.tof_depths[indices], axis=-1)

    if self.aug_mpi:
      tof_depths = self.augment_MPI(tof_depths, depths)

    if self.aug_flip:
      depths, correlations, tof_depths = self.random_flip_left_right([depths, correlations, tof_depths])

    if self.aug_rot:
      depths, correlations, tof_depths  = self.random_rot90([depths, correlations, tof_depths])

    if self.aug_noise:
      correlations = self.augment_noise(correlations, relative=True)

    points, rays = self.project_to_3D(tof_depths)
    # augmentation ##
    if self.aug_crop:
      depths, correlations, tof_depths, points, rays = self.random_crop([depths, correlations, tof_depths, points, rays])
    elif not self.do_pad:
      depths, correlations, tof_depths, points, rays = self.crop_center([depths, correlations, tof_depths, points, rays])

    # tof_depths = correlation2depth(correlations, frequency=self.frequency / 1e3)

    if self.do_pad:
      self.crop_pos_x, self.crop_pos_y = 0, 0
      # pad with zeros the last row
      # points = np.concatenate((points, np.zeros([batch_size, 1, self.input_width, 3])), axis=1)
      # depths = np.concatenate((depths, np.zeros([batch_size, 1, self.input_width, 1])), axis=1)
      # correlations = np.concatenate((correlations, np.zeros([batch_size, 1, self.input_width, 4])), axis=1)
      # tof_depths = np.concatenate((tof_depths, np.zeros([batch_size, 1, self.input_width, 1])), axis=1)
      # rays = np.concatenate((rays, np.zeros([batch_size, 1, self.input_width, 1])), axis=1)
      # pad with edge value
      h_diff = self.height - self.input_height
      w_diff = self.width - self.input_width

      points = np.pad(points, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      depths = np.pad(depths, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      correlations = np.pad(correlations, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      tof_depths = np.pad(tof_depths, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      rays = np.pad(rays, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
    # fill last batch with zeros to batch size
    if self.pad_batches and self.curr_batch_size < self.batch_size:
      bs_diff = self.batch_size - self.curr_batch_size
      points = np.pad(points, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      depths = np.pad(depths, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      correlations = np.pad(correlations, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      tof_depths = np.pad(tof_depths, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

    if not self.keepdims:
      points = np.reshape(points, [self.curr_batch_size * self.points_per_model, 3])
      depths = np.reshape(depths, [self.curr_batch_size * self.points_per_model, 1])
      correlations = np.reshape(correlations, [self.curr_batch_size * self.points_per_model, 4])
      tof_depths = np.reshape(tof_depths, [self.curr_batch_size * self.points_per_model, 1])

    # clean invalid values
    masks = depths != 0
    # depths *= masks

    rays = np.repeat(rays, self.batch_size, axis=0)

    return points, correlations, depths, tof_depths, masks, rays

  def get_single_frequency_correlations_amplitudes_intensities(self, index):
    """ Returns a batch of data
    """

    # sample points
    indices = \
        self.order[index * self.batch_size:(index + 1) * self.batch_size]
    self.curr_batch_size = len(indices)
    # shape [B, H_in, W_in, 4]
    correlations = self.correlations[indices]
    amplitudes = np.expand_dims(self.amplitudes[indices], axis=-1)
    intensities = np.expand_dims(self.intensities[indices], axis=-1)

    features = np.concatenate((correlations, amplitudes, intensities), axis=-1)

    if self.depths is not None:
      depths = np.expand_dims(self.depths[indices], axis=-1)
    else:
      shape = correlations.shape
      depths = np.zeros([*shape[:-1], 1])
    tof_depths = np.expand_dims(self.tof_depths[indices], axis=-1)

    if self.aug_mpi:
      tof_depths = self.augment_MPI(tof_depths, depths)

    if self.aug_flip:
      depths, features, tof_depths = self.random_flip_left_right([depths, features, tof_depths])

    if self.aug_rot:
      depths, features, tof_depths  = self.random_rot90([depths, features, tof_depths])

    if self.aug_noise:
      features = self.augment_noise(features, relative=True)

    points, rays = self.project_to_3D(tof_depths)
    # augmentation ##
    if self.aug_crop:
      depths, features, tof_depths, points, rays = self.random_crop([depths, features, tof_depths, points, rays])
    elif not self.do_pad:
      depths, features, tof_depths, points, rays = self.crop_center([depths, features, tof_depths, points, rays])

    # tof_depths = correlation2depth(correlations, frequency=self.frequency / 1e3)

    if self.do_pad:
      self.crop_pos_x, self.crop_pos_y = 0, 0
      # pad with zeros the last row
      # points = np.concatenate((points, np.zeros([batch_size, 1, self.input_width, 3])), axis=1)
      # depths = np.concatenate((depths, np.zeros([batch_size, 1, self.input_width, 1])), axis=1)
      # correlations = np.concatenate((correlations, np.zeros([batch_size, 1, self.input_width, 4])), axis=1)
      # tof_depths = np.concatenate((tof_depths, np.zeros([batch_size, 1, self.input_width, 1])), axis=1)
      # rays = np.concatenate((rays, np.zeros([batch_size, 1, self.input_width, 1])), axis=1)
      # pad with edge value
      h_diff = self.height - self.input_height
      w_diff = self.width - self.input_width

      points = np.pad(points, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      depths = np.pad(depths, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      features = np.pad(features, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      tof_depths = np.pad(tof_depths, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      rays = np.pad(rays, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
    # fill last batch with zeros to batch size
    if self.pad_batches and self.curr_batch_size < self.batch_size:
      bs_diff = self.batch_size - self.curr_batch_size
      points = np.pad(points, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      depths = np.pad(depths, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      features = np.pad(features, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      tof_depths = np.pad(tof_depths, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

    if not self.keepdims:
      points = np.reshape(points, [self.curr_batch_size * self.points_per_model, 3])
      depths = np.reshape(depths, [self.curr_batch_size * self.points_per_model, 1])
      features = np.reshape(features, [self.curr_batch_size * self.points_per_model, 4])
      tof_depths = np.reshape(tof_depths, [self.curr_batch_size * self.points_per_model, 1])

    # clean invalid values
    masks = depths != 0
    # depths *= masks

    rays = np.repeat(rays, self.batch_size, axis=0)

    return points, features, depths, tof_depths, masks, rays

  def get_multi_frequency_correlations(self, index):
    """ Returns a batch of data
    """

    # sample points
    indices = \
        self.order[index * self.batch_size:(index + 1) * self.batch_size]
    self.curr_batch_size = len(indices)
    # shape [B, F, H_in, W_in, 4]
    correlations = self.correlations[indices]
    # reshape correlations
    shape = correlations.shape
    correlations = np.transpose(correlations, (0, 2, 3, 4, 1))
    correlations = np.reshape(correlations, [shape[0], shape[2], shape[3], shape[1] * shape[4]])

    if self.depths is not None:
      depths = np.expand_dims(self.depths[indices], axis=-1)
    else:
      shape = correlations.shape
      depths = np.zeros([*shape[:-1], 1])
    tof_depths = np.expand_dims(self.tof_depths[indices], axis=-1)
    # highest frequency tof depth
    tof_depths = tof_depths[:, -1]

    if self.aug_mpi:
      tof_depths = self.augment_MPI(tof_depths, depths)

    if self.aug_flip:
      depths, correlations, tof_depths = self.random_flip_left_right([depths, correlations, tof_depths])

    if self.aug_rot:
      depths, correlations, tof_depths  = self.random_rot90([depths, correlations, tof_depths])

    if self.aug_noise:
      correlations = self.augment_noise(correlations, relative=True)

    points, rays = self.project_to_3D(tof_depths)
    # augmentation ##
    if self.aug_crop:
      depths, correlations, tof_depths, points, rays = self.random_crop([depths, correlations, tof_depths, points, rays])
    elif not self.do_pad:
      depths, correlations, tof_depths, points, rays = self.crop_center([depths, correlations, tof_depths, points, rays])

    # tof_depths = correlation2depth(correlations, frequency=self.frequency / 1e3)

    if self.do_pad:
      self.crop_pos_x, self.crop_pos_y = 0, 0
      # pad with zeros the last row
      # points = np.concatenate((points, np.zeros([batch_size, 1, self.input_width, 3])), axis=1)
      # depths = np.concatenate((depths, np.zeros([batch_size, 1, self.input_width, 1])), axis=1)
      # correlations = np.concatenate((correlations, np.zeros([batch_size, 1, self.input_width, 4])), axis=1)
      # tof_depths = np.concatenate((tof_depths, np.zeros([batch_size, 1, self.input_width, 1])), axis=1)
      # rays = np.concatenate((rays, np.zeros([batch_size, 1, self.input_width, 1])), axis=1)
      # pad with edge value
      h_diff = self.height - self.input_height
      w_diff = self.width - self.input_width

      points = np.pad(points, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      depths = np.pad(depths, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      correlations = np.pad(correlations, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      tof_depths = np.pad(tof_depths, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      rays = np.pad(rays, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
    # fill last batch with zeros to batch size
    if self.pad_batches and self.curr_batch_size < self.batch_size:
      bs_diff = self.batch_size - self.curr_batch_size
      points = np.pad(points, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      depths = np.pad(depths, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      correlations = np.pad(correlations, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      tof_depths = np.pad(tof_depths, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

    if not self.keepdims:
      points = np.reshape(points, [self.curr_batch_size * self.points_per_model, 3])
      depths = np.reshape(depths, [self.curr_batch_size * self.points_per_model, 1])
      correlations = np.reshape(correlations, [self.curr_batch_size * self.points_per_model, 4])
      tof_depths = np.reshape(tof_depths, [self.curr_batch_size * self.points_per_model, 1])

    # clean invalid values
    masks = depths != 0
    # depths *= masks

    rays = np.repeat(rays, self.batch_size, axis=0)

    return points, correlations, depths, tof_depths, masks, rays

  def get_agresti_features(self, index):
    """ Returns a batch of data
    """

    # sample points
    indices = \
        self.order[index * self.batch_size:(index + 1) * self.batch_size]
    self.curr_batch_size = len(indices)

    tof_depths = np.expand_dims(self.tof_depths[indices], axis=-1)
    amplitudes = np.expand_dims(self.amplitudes[indices], axis=-1)

    if self.depths is not None:
      depths = np.expand_dims(self.depths[indices], axis=-1)
    else:
      depths = np.zeros_like(tof_depths[:, -1])

    if self.aug_mpi:
      tof_depths = self.augment_MPI(tof_depths, depths)

    features = np.concatenate(
      [
        tof_depths[:, 2],  # tof depth at 60 MHz
        tof_depths[:, 0] - tof_depths[:, 2],  # difference tof depths at 20MHz and at 60MHz
        tof_depths[:, 1] - tof_depths[:, 2],  # difference tof depths at 50MHz and at 60MHz
        (amplitudes[:, 0] / amplitudes[:, 2]) - 1,  # centered amplitudes ratios between 20MHz and 60 MHz
        (amplitudes[:, 1] / amplitudes[:, 2]) - 1,  # centered amplitudes ratios between 50MHz and 60 MHz
      ], axis=-1)

    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    # highest frequency tof depth
    tof_depths = tof_depths[:, -1]

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
      depths, features, tof_depths, points, rays = self.random_crop([depths, features, tof_depths, points, rays])
    elif not self.do_pad:
      depths, features, tof_depths, points, rays = self.crop_center([depths, features, tof_depths, points, rays])

    # tof_depths = correlation2depth(correlations, frequency=self.frequency / 1e3)

    if self.do_pad:
      self.crop_pos_x, self.crop_pos_y = 0, 0
      # pad with zeros the last row
      # points = np.concatenate((points, np.zeros([batch_size, 1, self.input_width, 3])), axis=1)
      # depths = np.concatenate((depths, np.zeros([batch_size, 1, self.input_width, 1])), axis=1)
      # correlations = np.concatenate((correlations, np.zeros([batch_size, 1, self.input_width, 4])), axis=1)
      # tof_depths = np.concatenate((tof_depths, np.zeros([batch_size, 1, self.input_width, 1])), axis=1)
      # rays = np.concatenate((rays, np.zeros([batch_size, 1, self.input_width, 1])), axis=1)
      # pad with edge value
      h_diff = self.height - self.input_height
      w_diff = self.width - self.input_width

      points = np.pad(points, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      depths = np.pad(depths, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      features = np.pad(features, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      tof_depths = np.pad(tof_depths, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
      rays = np.pad(rays, ((0, 0), (0, h_diff), (0, w_diff), (0, 0)), mode='edge')
    # fill last batch with zeros to batch size
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
    masks = depths != 0
    # depths *= masks

    rays = np.repeat(rays, self.batch_size, axis=0)

    return points, features, depths, tof_depths, masks, rays
