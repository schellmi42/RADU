'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import h5py
from tensorflow.python.ops.check_ops import assert_equal
# from code_dl.data_ops.data_loader import read_files, read_files_h5
from code_dl.data_ops.data_loader import load_filenames, load_batch
from code_dl.data_ops.geom_ops_numpy import compute_points_from_depth, correlation2depth, camera_rays, constants
from code_dl.data_ops.phase_unwrapping import phase_unwrapping_two_frequencies_tf


class data_generator(tf.keras.utils.Sequence):
  ''' Small generator for batched data.

  Generates in the format [points, correlations, depths, tof_depths, masks, rays]

  Args:
    batch_size: `int` the batch size B,
    set: `'train', 'val'`, or `'test'`.
    frequencies: `list` of `floats`, in MHz.
    height: `int` H, size to crop in x axis.
    width: `int` W, size to crop in y axis.
    input_height: `int` , height of images in data set.
    input_width: `int` , width of images in data set.
    keepdims: `bool`, if `True` return data in shape `[B, H, W, C]`.
      If `False` returns data in shape `[N, C]`. (point cloud format)
    noise_level: `float` level of the noise applied to the data in augmentation.
      If `0`, then no noise augmentation is done.
    aug_*: `bool`, to activate augmentation strategies.
      available: crop, flip, rot (rot90), material, noise
  '''

  def __init__(self,
               batch_size,
               set,
               frequencies,
               height=512,
               width=512,
               input_height=600,
               input_width=600,
               fov=[65, 65],
               keepdims=False,
               aug_noise=False,
               noise_level=0.0,
               aug_crop=False,
               aug_flip=False,
               aug_rot=False,
               aug_material=False,
               aug_mpi=False,
               shuffle=True,
               pad_batches=False,
               feature_type='sf_c',
               unwrap_phases=False,
               full_scene_in_epoch=False):
    self.height = height
    self.width = width
    self.input_height = input_height
    self.input_width = input_width
    self.keepdims = keepdims
    self.points_per_model = height * width
    self.batch_size = batch_size
    self.fov = fov
    self.frequencies = frequencies
    self.flip_HW = False
    self.feature_type = feature_type
    self.full_scene_in_epoch = full_scene_in_epoch

    if feature_type == 'mf_agresti':
      self.frequencies = ['20', '50', '70']
    if feature_type == 'mf_c':
      self.frequencies = ['20', '50', '70']
    if feature_type == 'mf_su':
      self.frequencies = ['50', '70']
    self.unwrap_phases = unwrap_phases

    self.frames = load_filenames(set)
    self.frames_flat = np.reshape(self.frames, [-1])

    # self.corr_files = np.array(corr_files, dtype=str)
    # self.tof_depth_files = np.array(corr_files, dtype=str)
    # self.depth_files = np.array(depth_files, dtype=str)
    self.num_scenes, self.frames_per_scene = self.frames.shape
    self.num_frequencies = len(frequencies)

    # self.corr_files_flat = self.corr_files.reshape([-1, self.num_frequencies, 4])
    # self.depth_files_flat = np.array(depth_files, dtype=str).reshape([-1])
    # self.tof_depth_files_flat = np.array(tof_depth_files, dtype=str).reshape([-1])
    # self.projection = projection
    self.aug_noise = aug_noise
    self.noise_level = noise_level
    self.aug_crop = aug_crop
    self.aug_flip = aug_flip
    self.aug_rot = aug_rot
    self.aug_material = aug_material
    self.aug_mpi = aug_mpi

    self.epoch_size = self.num_scenes
    if full_scene_in_epoch:
      self.epoch_size = self.num_scenes * self.frames_per_scene

    self.sizes = np.ones([batch_size]) * self.points_per_model
    # shuffle data before training
    self.shuffle = shuffle
    self.pad_batches = pad_batches
    self.on_epoch_end()

  def __len__(self):
    """ Number of batches in generator.
    """
    if self.pad_batches:
      return int(np.ceil(self.epoch_size / self.batch_size))
    else:
      return self.epoch_size // self.batch_size

  def __call__(self):
    ''' Loads batch and increases batch index.
    '''
    data = self.__getitem__(self.index)
    self.index += 1
    return data

  def ds_call(self, index):
    ''' Loads batch and increases batch index.
    '''
    print(index)
    points, correlations, depths, tof_depths, masks, rays = self.__getitem__(index)
    yield points, correlations, depths, tof_depths, masks, rays

  def init_tfds(self, prefetch_buffer=2):
    """ creates a TFDS dataset for data prefetching"""
    self.tfds = tf.data.Dataset.from_generator(
        self.ds_call,
        output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.bool, tf.float32),
        output_shapes=(
            tf.TensorShape((self.batch_size, None, None, 3)),
            tf.TensorShape((self.batch_size, None, None, 4)),
            tf.TensorShape((self.batch_size, None, None, 1)),
            tf.TensorShape((self.batch_size, None, None, 1)),
            tf.TensorShape((self.batch_size, None, None, 1)),
            tf.TensorShape((self.batch_size, None, None, 3))),
        args=[len(self)]).prefetch(prefetch_buffer)
    print('[TFDS Prefetching with buffer size {:1d}]'.format(prefetch_buffer))

  def __getitem__(self, index):
    if self.feature_type == 'sf_c':
      return self.get_single_frequency_correlations(index)
    elif self.feature_type == 'sf_d':
      return self.get_single_frequency_depth(index)
    elif self.feature_type == 'sf_cai':
      return self.get_single_frequency_correlations_amplitudes_intensities(index)
    if self.feature_type == 'mf_c':
      return self.get_multi_frequency_correlations(index)
    if self.feature_type == 'mf_su':
      return self.get_su_features(index)
    elif self.feature_type == 'mf_agresti':
      return self.get_agresti_features(index)

  def get_batch_indices(self, index):
    if not self.full_scene_in_epoch:
      scene_indices = \
          self.order_scenes[0][index * self.batch_size:(index + 1) * self.batch_size]
      self.curr_batch_size = len(scene_indices)
      if self.shuffle:
        shot_ids = np.random.choice(np.arange(self.frames_per_scene), self.curr_batch_size)
      else:
        shot_ids = 0
      indices = scene_indices * self.frames_per_scene + shot_ids
    else:
      indices = np.arange(index * self.batch_size, (index + 1) * self.batch_size, step=1)

      self.curr_batch_size = len(indices)

    return indices

  def get_single_frequency_depth(self, index):
    """ Returns a batch of data
    """

    # sample points
    indices = self.get_batch_indices(index)
    if self.aug_material:
      material_id = np.random.choice([0, 1, 2])
    else:
      material_id = 0

    # shape [B, H_in, W_in, 1]
    depths, tof_depths, _, _, _ = load_batch(self.frames_flat[indices], self.frequencies, material_id)
    tof_depths = np.expand_dims(np.squeeze(tof_depths, axis=1), axis=-1)
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

  def get_multi_frequency_correlations(self, index):
    """ Returns a batch of data
    """
    indices = self.get_batch_indices(index)

    if self.aug_material:
      material_id = np.random.choice([0, 1, 2])
    else:
      material_id = 0

    depths, tof_depths, correlations, _, _ = load_batch(self.frames_flat[indices], self.frequencies, material_id)
    depths = np.expand_dims(depths, axis=-1)

    if self.unwrap_phases:
      _, tof_depths[:, 2] = phase_unwrapping_two_frequencies_tf(tof_depths[:, 0], tof_depths[:, 2], frequencies=[20 / 1e3, 70 / 1e3], max_wraps=[0, 2])

    if self.aug_mpi:
      tof_depths = self.augment_MPI(tof_depths, depths)

    correlations = np.transpose(correlations, [0, 3, 4, 1, 2]).reshape([self.curr_batch_size, self.input_height, self.input_width, -1])

    # highest frequency tof depth
    tof_depths = tof_depths[:, -1]
    tof_depths = np.expand_dims(tof_depths, axis=-1)

    # augmentation ##
    if self.aug_crop:
      correlations, depths, tof_depths = self.random_crop([correlations, depths, tof_depths])
    else:
      correlations, depths, tof_depths = self.crop_center([correlations, depths, tof_depths])

    if self.aug_flip:
      correlations, depths, tof_depths = self.random_flip_left_right([correlations, depths, tof_depths])

    if self.aug_rot:
      correlations, depths, tof_depths = self.random_rot90([correlations, depths, tof_depths])

    if self.aug_noise:
      correlations = self.augment_noise(correlations)
      tof_depths = self.augment_noise(tof_depths)
    # tof_depths = correlation2depth(correlations, frequency=self.frequencies[freq_id] / 1e3)
    points, rays = self.project_to_3D(tof_depths)

    if self.pad_batches and self.curr_batch_size < self.batch_size:
      bs_diff = self.batch_size - self.curr_batch_size
      points = np.pad(points, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      depths = np.pad(depths, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      correlations = np.pad(correlations, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
      tof_depths = np.pad(tof_depths, ((0, bs_diff), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

    if not self.keepdims:
      points = np.reshape(points, [self.batch_size * self.points_per_model, 3])
      correlations = np.reshape(correlations, [self.batch_size * self.points_per_model, 4])
      depths = np.reshape(depths, [self.batch_size * self.points_per_model, 1])
      tof_depths = np.reshape(tof_depths, [self.batch_size * self.points_per_model, 1])

    # clean invalid values
    masks = (depths < 1e3) * (depths != 0)
    depths *= masks

    rays = np.repeat(rays, self.batch_size, axis=0)

    return points, correlations, depths, tof_depths, masks, rays

  def get_su_features(self, index):
    """ Returns a batch of data
    """

    indices = self.get_batch_indices(index)
    if self.aug_material:
      material_id = np.random.choice([0, 1, 2])
    else:
      material_id = 0

    # shape [B, F, H_in, W_in, 1]
    depths, tof_depths, correlations, amplitudes, _ = load_batch(self.frames_flat[indices], self.frequencies, material_id)
    amplitudes = np.transpose(amplitudes, [0, 2, 3, 1])
    depths = np.expand_dims(depths, axis=-1)

    features = np.stack(
      [
        correlations[:, 0, 0],  # correlations at 50MHz phase 0
        correlations[:, 0, 1],  # correlations at 50MHz phase 90
        correlations[:, 1, 0],  # correlations at 70MHz phase 0
        correlations[:, 1, 1],  # correlations at 70MHz phase 90
      ], axis=-1)

    # highest frequency tof depth
    tof_depths = tof_depths[:, -1]
    tof_depths = np.expand_dims(tof_depths, axis=-1)
    if self.aug_flip:
      depths, features, amplitudes = self.random_flip_left_right([depths, features, amplitudes])

    if self.aug_rot:
      depths, features, amplitudes  = self.random_rot90([depths, features, amplitudes])

    if self.aug_noise:
      features = self.augment_noise(features, relative=True)

    points, rays = self.project_to_3D(tof_depths)
    # augmentation ##
    if self.aug_crop:
      depths, features, amplitudes, points, rays = self.random_crop([depths, features, amplitudes, points, rays])
    else:
      depths, features, amplitudes, points, rays = self.crop_center([depths, features, amplitudes, points, rays])

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

    return points, features, depths, amplitudes, masks, rays

  def get_agresti_features(self, index):
    """ Returns a batch of data
    """

    # sample points
    indices = self.get_batch_indices(index)

    if self.aug_material:
      material_id = np.random.choice([0, 1, 2])
    else:
      material_id = 0

    # shape [B, F, H_in, W_in, 1]
    depths, tof_depths, _, amplitudes, _ = load_batch(self.frames_flat[indices], self.frequencies, material_id)
    depths = np.expand_dims(depths, axis=-1)
    if self.unwrap_phases:
      _, tof_depths[:, 1] = phase_unwrapping_two_frequencies_tf(tof_depths[:, 0], tof_depths[:, 1], frequencies=[20 / 1e3, 50 / 1e3], max_wraps=[0, 1])
      _, tof_depths[:, 2] = phase_unwrapping_two_frequencies_tf(tof_depths[:, 0], tof_depths[:, 2], frequencies=[20 / 1e3, 70 / 1e3], max_wraps=[0, 2])

    if self.aug_mpi:
      tof_depths = self.augment_MPI(tof_depths, depths)

    features = np.stack(
      [
        tof_depths[:, 2],  # tof depth at 70 MHz
        tof_depths[:, 0] - tof_depths[:, 2],  # difference tof depths at 20MHz and at 70MHz
        tof_depths[:, 1] - tof_depths[:, 2],  # difference tof depths at 50MHz and at 70MHz
        (amplitudes[:, 0] / amplitudes[:, 2]) - 1,  # centered amplitudes ratios between 20MHz and 70 MHz
        (amplitudes[:, 1] / amplitudes[:, 2]) - 1,  # centered amplitudes ratios between 50MHz and 70 MHz
      ], axis=-1)

    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    if self.unwrap_phases:
      # highest frequency tof depth
      tof_depths = tof_depths[:, -1]
      tof_depths = np.expand_dims(tof_depths, axis=-1)
    else:
      # lowest frequency tof_depth
      tof_depths = tof_depths[:, 0]
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
      depths, features, tof_depths, points, rays = self.random_crop([depths, features, tof_depths, points, rays])
    else:
      depths, features, tof_depths, points, rays = self.crop_center([depths, features, tof_depths, points, rays])

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
    masks = (depths < 1e3) * (depths != 0)
    depths *= masks

    rays = np.repeat(rays, self.batch_size, axis=0)

    return points, features, depths, tof_depths, masks, rays

  def on_epoch_end(self):
    ''' Shuffles data and resets batch index.
    '''
    if self.shuffle:
      self.order_scenes = np.array([np.random.permutation(np.arange(0, self.num_scenes)) for i in range(self.frames_per_scene)])
      self.order_shots = np.array([np.random.permutation(np.arange(0, self.frames_per_scene)) for i in range(self.num_scenes)]).T
      self.order = np.reshape(self.order_scenes * self.frames_per_scene + self.order_shots, [-1])
    else:
      self.order_scenes = np.array([np.arange(0, self.num_scenes) for i in range(self.frames_per_scene)])
      self.order = np.arange(0, self.num_scenes)
    self.index = 0

  def project_to_3D(self, depths):
    """ projects from 2.5D to 3D.
    Args:
      depths: shape `[B, H, W, 1]
    Returns:
      points: shape `[B, H, W, 3]
      rays: shape `[1, H, W, 3]
    """
    points, rays = compute_points_from_depth(depths, fov=self.fov, return_rays=True)
    return points, rays

  def get_rays(self, shape):
    """ normalized camera ray directions

    Args:
        shape: 3 `ints`, `[B, H, W]`.
    Returns:
        A `float` `np.array`of shape `[1, H, W, 3]`.
    """
    return camera_rays(shape[:3], fov=65)

  #  AUGMENTATIONS

  def crop_center(self, data):
    """ Crops the center of the images.
    """
    self.crop_pos_x = (self.input_height - self.height) // 2
    self.crop_pos_y = (self.input_width - self.width) // 2
    self.flip_HW = False
    data_aug = []
    for d in data:
      data_aug.append(d[:, self.crop_pos_x:self.height + self.crop_pos_x, self.crop_pos_y:self.width + self.crop_pos_y, :])
    return data_aug

  def random_crop(self, data):
    """ Crops a random patch of the images.
    """
    if self.input_height > self.height:
      self.crop_pos_x = np.random.choice(np.arange(0, self.input_height - self.height))
    else:
      self.crop_pos_x = 0
    if self.input_width > self.width:
      self.crop_pos_y = np.random.choice(np.arange(0, self.input_width - self.width))
    else:
      self.crop_pos_y = 0
    if self.flip_HW:
      self.crop_pos_x, self.crop_pos_y = self.crop_pos_y, self.crop_pos_x
    data_aug = []
    for d in data:
      data_aug.append(d[:, self.crop_pos_x:self.height + self.crop_pos_x, self.crop_pos_y:self.width + self.crop_pos_y, :])
    return data_aug

  def random_flip_left_right(self, data):
    """  Randomly flip the images horizontally.
    """
    flip = np.random.choice([True, False])
    if flip:
      data_aug = []
      for d in data:
        data_aug.append(d[:, :, ::-1])
      return data_aug
    else:
      return data

  def random_rot90(self, data):
    """ Random rotation by 0, 90, 180 or 270 degrees.
    """
    k = np.random.choice([0, 1, 2, 3])
    self.flip_HW = (k in [1, 3])
    data_aug = []
    for d in data:
      data_aug.append(np.rot90(d, k=k, axes=(1, 2)))
    return data_aug

  def random_rot(self, data, max_angle):
    """ Random rotation by random angles in given range degrees.
    Different angle per image in batch.
    Args:
      max_angle: `float`, angle in degree.
    """
    batch_size = data[0].shape[0]
    angles = np.random.uniform(-max_angle, max_angle, batch_size) * np.pi / 180

    data_aug = []
    for d in data:
      data_aug.append(tfa.image.rotate(d, angles, interpolation="BILINEAR", fill_mode='reflect').numpy())
    return data_aug

  def augment_noise(self, data, relative=True):
    """ Add random gaussian noise. (shot noise)
    Args:
      data: of shape `[B, H, W, C]`
    """
    if relative:
      noise = np.random.normal(size=data.shape, scale=self.noise_level * np.abs(data))
    else:
      noise = np.random.normal(size=data.shape, scale=self.noise_level)
    return data + noise

  def augment_MPI(self, tof_depths, depths, range=[0.5, 1.5]):
    if len(tof_depths.shape) == 5:
      # for multiple frequencies at once
      depths = np.expand_dims(depths, axis=1)
    mpi = tof_depths - depths
    mpi_amplitude = np.random.uniform(low=range[0], high=range[1])
    return depths  + mpi * mpi_amplitude

  def _plot_image(self, img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

  def valid_ratio(self, masks):
    """ Computes the ratio of valid pixels per image to weight the loss functions.

    Args:
      masks: `bool` of arbitrary shape.
    Returns:
      `float`
    """
    valid_pixels = np.sum(masks)
    total_pixels = np.prod(masks.shape)
    return total_pixels / valid_pixels
