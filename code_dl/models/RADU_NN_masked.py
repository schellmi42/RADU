'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf
from pylib.pc import PointCloud
from pylib.pc.layers import MCConv, KPConv, PointConv

from code_dl.layers import ConvDepthUpdate
from code_dl.models import config
from code_dl.data_ops.geom_ops_tf import global_depth_to_camera_depth_with_rays
from code_dl.layers.utils import mask, demask
print('## BUILDING: RADU model with masking at input resolution: ###')
print(config.INPUT_FEATURES_SHAPE)


def model_params(string):
  """ To create parameters for the below model
  """
  update_scale = [0.1, 0.1, 0.1]
  if string == 'v3_avg':
    feature_sizes = [64, 64, 128, 128, 256, 128, 64, 64, 1]
    conv_radius = [0.1, 0.2, 0.4]
    sampling_rate = 8
  return {'feature_sizes': feature_sizes,
          'sampling_rate': sampling_rate,
          'conv_radius': conv_radius,
          'update_scale': update_scale}


class mymodel(tf.Module):
  ''' Model architecture.

  Args:
    features_sizes: A `list` of `ints`, the feature dimensions. Shape `[L+4]`.
    format: `[2D, 2D, 2D, 3D, ..., 3D, 2D, 2D, 2D]`
    conv_radii: A `list` of `floats`, the radii used by the 3D convolution
      layers. Shape `[L]`.
    layer_type: A `string`, the type of convolution used,
      can be 'MCConv', 'MCConv_du'.
    sampling_method: method to sample the point clouds,
      can be 'posson disk' or 'cell average'
  '''

  def __init__(self,
               feature_sizes,
               conv_radius,
               layer_type='MCConv',
               sampling_rate=8,
               update_along_rays=True,
               batch_size=None,
               pooling_method='avg',
               upscaling_method='bilinear',
               project_back=True,
               normalize_features=False,
               use_BN=False,
               use_BN_3D=False,
               skip_3D=False,
               skip_to_output=False,
               correct_upscaling=False,
               update_scale=[0.1, 0.1, 0.1],
               **kwargs):
    print(config.INPUT_FEATURES_SHAPE)
    super().__init__(name=None)
    self.batch_size = batch_size
    self.layer_type = layer_type
    self.normalize_features = normalize_features
    self.project_back = project_back
    self.pooling_method = pooling_method
    self.upscaling_method = upscaling_method
    self.use_BN = use_BN
    self.use_BN_3D = use_BN_3D
    self.skip_3D = skip_3D
    self.skip_to_output = skip_to_output
    self.correct_upscaling = correct_upscaling
    self.num_levels_3D = len(feature_sizes) - 6
    self.sampling_shape = [config.INPUT_FEATURES_SHAPE[1] // sampling_rate, config.INPUT_FEATURES_SHAPE[2] // sampling_rate]
    self.sampling_rate = sampling_rate
    self.feature_sizes_enc = [config.INPUT_FEATURES_SHAPE[3]] + feature_sizes[:3]

    self.feature_sizes_3D = feature_sizes[2:6]
    self.feature_sizes_dec = feature_sizes[5:]
    if self.layer_type == 'MCConv_du':
      self.feature_sizes_dec[0] += 1
    if self.skip_3D:
      self.feature_sizes_dec[0] += self.feature_sizes_enc[-1]

    self.conv_radius = conv_radius

    if self.normalize_features:
      # pixel-wise normalization to correlation inputs with their corresponding amplitudes
      self.normalization = tf.keras.layers.LayerNormalization(input_shape=config.INPUT_FEATURES_SHAPE[1:], axis=-1)
    # downscaling method
    self.ray_downscaling = tf.keras.layers.AveragePooling2D(pool_size=(self.sampling_rate, self.sampling_rate),
                                                            strides=(self.sampling_rate, self.sampling_rate), padding='valid')
    if pooling_method.lower() == 'max':
      self.downscaling = tf.keras.layers.MaxPooling2D(pool_size=(self.sampling_rate, self.sampling_rate),
                                                      strides=(self.sampling_rate, self.sampling_rate), padding='valid')
    elif pooling_method.lower() == 'avg':
      self.downscaling = self.ray_downscaling

    # Encoder
    self.convs_enc = []
    self.BNs_enc = []
    self.activations_enc = []
    for i in range(3):
      self.convs_enc.append(tf.keras.layers.Conv2D(
          input_shape=[*config.INPUT_FEATURES_SHAPE[1:-1], self.feature_sizes_enc[i]], filters=self.feature_sizes_enc[i + 1], kernel_size=[3, 3], padding='same'))
      if self.use_BN:
        self.BNs_enc.append(tf.keras.layers.BatchNormalization(momentum=0.9))
      self.activations_enc.append(tf.keras.layers.LeakyReLU())

    # -- encoder network
    self.convs_3D = []
    self.BNs_3D = []
    self.activations_3D = []
    for i in range(self.num_levels_3D):
      if layer_type == 'MCConv':
        self.convs_3D.append(MCConv(
          num_features_in=self.feature_sizes_3D[i],
          num_features_out=self.feature_sizes_3D[i + 1],
          num_dims=3,
          num_mlps=1,
          mlp_size=[16]))
      elif layer_type == 'MCConv_du':
        self.convs_3D.append(ConvDepthUpdate(
          num_features_in=self.feature_sizes_3D[i],
          num_features_out=self.feature_sizes_3D[i + 1],
          num_dims=3,
          num_mlps=1,
          mlp_size=[16],
          update_along_rays=update_along_rays,
          update_scale=update_scale[i]))
      elif layer_type == 'KPConv':
        self.convs_3D.append(KPConv(
          num_features_in=self.feature_sizes_3D[i],
          num_features_out=self.feature_sizes_3D[i + 1],
          num_dims=3,
          num_kernel_points=15))
      elif layer_type == 'PointConv':
        self.convs_3D.append(PointConv(
          num_features_in=self.feature_sizes_3D[i],
          num_features_out=self.feature_sizes_3D[i + 1],
          num_dims=3,
          size_hidden=16))
      else:
        raise ValueError("Unknown layer type!")
      if self.use_BN or self.use_BN_3D:
        self.BNs_3D.append(tf.keras.layers.BatchNormalization(momentum=0.9))
      self.activations_3D.append(tf.keras.layers.LeakyReLU())

    # decoder
    self.convs_dec = []
    self.BNs_dec = []
    self.activations_dec = []
    for i in range(3):
      self.convs_dec.append(tf.keras.layers.Conv2D(
          input_shape=(*config.INPUT_FEATURES_SHAPE[1:], self.feature_sizes_dec[i]),
          filters=self.feature_sizes_dec[i + 1],
          kernel_size=(3, 3),
          padding='same'))
      if i != 2:
        if self.use_BN:
          self.BNs_dec.append(tf.keras.layers.BatchNormalization(momentum=0.9))
        self.activations_dec.append(tf.keras.layers.LeakyReLU())

    # print summary
    print('#########################################')
    print('Layer type:              ' + layer_type)
    print('feature_sizes per pixel: ', *(self.feature_sizes_enc + self.feature_sizes_3D + self.feature_sizes_dec))
    print('convolution radius:      ' + str(conv_radius))
    print('number of 3D Convs:      ' + str(self.num_levels_3D))
    print('depth update along rays: ' + str(update_along_rays))
    print('#########################################')

  def depth_upscaling(self, depths):
    if self.upscaling_method == 'bilinear':
      # biliear upsampling
      return tf.image.resize(depths, config.INPUT_FEATURES_SHAPE[1:3], method=tf.image.ResizeMethod.BILINEAR)
    elif self.upscaling_method == 'repeat':
      # upsampling by repeating values
      return tf.repeat(tf.repeat(depths, self.sampling_rate, axis=1), self.sampling_rate, axis=2)

  @tf.function(
    input_signature=[
        tf.TensorSpec(shape=[*config.INPUT_FEATURES_SHAPE[:-1], 3], dtype=tf.float32),  # rays [B, H, W, 3]
        tf.TensorSpec(shape=config.INPUT_FEATURES_SHAPE, dtype=tf.float32),  # features [B, H, W, C]
        tf.TensorSpec(shape=[*config.INPUT_FEATURES_SHAPE[:-1], 1], dtype=tf.float32),  # initial ToF-depths [B, H, W, 1]
        tf.TensorSpec(shape=None, dtype=tf.bool)]  # training
        )
  def __call__(self,
               rays,
               features,
               input_depths,
               training):
    ''' Evaluates network, uses standard quaternions.

    Args:
      rays: The normalized camera ray vectors.
      features: Input features.
      sizes: sizes of the point clouds
      training: A `bool`, passed to the batch norm layers.

    Returns:
      The logits per class.
    '''
    rays = tf.convert_to_tensor(value=rays, dtype=tf.float32)
    features = tf.convert_to_tensor(value=features, dtype=tf.float32)
    input_depths = tf.convert_to_tensor(value=input_depths, dtype=tf.float32)

    # spatial downscaling
    sampled_rays, _ = tf.linalg.normalize(self.ray_downscaling(rays), axis=-1)
    sampled_depths = self.downscaling(input_depths)
    sampled_points = sampled_rays * sampled_depths

    #down / upscaling error on depth
    if self.correct_upscaling:
      depth_scaling_correction = input_depths - self.depth_upscaling(sampled_depths)

    # prepare point cloud
    masks = tf.reduce_any(sampled_points > 1e-4, axis=-1)
    sizes = tf.reduce_sum(tf.cast(masks, dtype=tf.int32), axis=[1, 2])
    # mask zero values
    sampled_points = mask(sampled_points, masks)
    sampled_rays = mask(sampled_rays, masks)
    point_cloud = PointCloud(
        sampled_points, sizes=sizes, batch_size=self.batch_size)
    if self.normalize_features:
      features = self.normalization(features)

    # Encoder
    for i in range(3):
      features = self.convs_enc[i](features)
      if self.use_BN:
        features = self.BNs_enc[i](features, training=training)
      features = self.activations_enc[i](features)
    if self.skip_3D:
      features_skip_3D = features
    # spatial downsampling of the features
    features = self.ray_downscaling(features)

    features = mask(features, masks)

    # 3D network
    if self.layer_type == 'MCConv_du':
      additional_args = {'rays': sampled_rays}
    else:
      additional_args = {}
    for i in range(self.num_levels_3D):
      features = self.convs_3D[i](
          features,
          point_cloud,
          point_cloud,
          self.conv_radius[i],
          **additional_args
      )
      if self.use_BN or self.use_BN_3D:
        features = self.BNs_3D[i](features, training=training)
      features = self.activations_3D[i](features)

    # extract per point depth
    depths_coarse = tf.gather(point_cloud._points, [2], axis=-1)
    # demask zero points
    depths_coarse = demask(depths_coarse, masks)
    features = demask(features, masks)
    # reshape for 2D network
    features = tf.reshape(features, [self.batch_size, *self.sampling_shape, self.feature_sizes_3D[-1]])
    depths_coarse = tf.reshape(depths_coarse, [self.batch_size, *self.sampling_shape, 1])
    # upscale depth with correction factor and project back to camera space
    depths_upscaled = self.depth_upscaling(depths_coarse)
    if self.correct_upscaling:
      depths_upscaled += depth_scaling_correction
    # print(tf.reduce_sum(depths))
    if self.project_back:
      depths_upscaled = global_depth_to_camera_depth_with_rays(depths_upscaled, rays)
    # upscale features
    features = self.depth_upscaling(features)

    # decoder
    if self.skip_3D:
      features = tf.concat((features, features_skip_3D), axis=-1)
    if self.layer_type == 'MCConv_du':
      features = tf.concat((depths_upscaled, features), axis=-1)

    for i in range(3):
      features = self.convs_dec[i](features)
      if i != 2:
        if self.use_BN:
          features = self.BNs_dec[i](features, training=training)
        features = self.activations_dec[i](features)

    if self.skip_to_output:
      return (features + input_depths, depths_upscaled)
    else:
      return (features, depths_upscaled)
