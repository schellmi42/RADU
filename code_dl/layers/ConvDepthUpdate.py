'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf
from pylib.pc.layers import MCConv


class ConvDepthUpdate(MCConv):
  """ Monte-Carlo convolution for point clouds with depth update.

  Based on the paper [Monte Carlo Convolution for Learning on Non-Uniformly
  Sampled Point Clouds. Hermosilla et al., 2018]
  (https://arxiv.org/abs/1806.01759).
  Uses a multiple MLPs as convolution kernels.

  Args:
    num_features_in: An `int`, `C_in`, the number of features per input point.
    num_features_out: An `int`, `C_out`, the number of features to compute.
    num_dims: An `int`, the input dimension to the kernel MLP. Should be the
      dimensionality of the point cloud.
    num_mlps: An `int`, number of MLPs used to compute the output features.
      Warning: num_features_out should be divisible by num_mlps.
    mlp_size: An ìnt list`, list with the number of layers and hidden neurons
      of the MLP used as kernel, defaults to `[8]`. (optional).
    non_linearity_type: An `string`, specifies the type of the activation
      function used inside the kernel MLP.
      Possible: `'ReLU', 'lReLU', 'ELU'`, defaults to leaky ReLU. (optional)
    initializer_weights: A `tf.initializer` for the kernel MLP weights,
      default `TruncatedNormal`. (optional)
    initializer_biases: A `tf.initializer` for the kernel MLP biases,
      default: `zeros`. (optional)

  """

  def __init__(self,
               num_features_in,
               num_features_out,
               num_dims,
               num_mlps=4,
               mlp_size=[8],
               non_linearity_type='leaky_relu',
               initializer_weights=None,
               initializer_biases=None,
               update_along_rays=False,
               update_scale=1.0,
               name=None):
    super().__init__(
        num_features_in,
        num_features_out + 1,
        num_dims,
        num_mlps,
        mlp_size,
        non_linearity_type,
        initializer_weights,
        initializer_biases,
        name)
    self.update_along_rays = update_along_rays
    self.update_scale = update_scale

  def __call__(self, *args, **kwargs):
    # the switch is in the initialization to prevent a `tf.cond` branching at runtime.
    if self.update_along_rays:
      return self._call_with_rays(*args, **kwargs)
    else:
      return self. _call_without_rays(*args, **kwargs)

  def _call_without_rays(self,
                         features,
                         point_cloud_in,
                         point_cloud_out,
                         radius,
                         neighborhood=None,
                         bandwidth=0.2,
                         return_sorted=False,
                         return_padded=False,
                         rays=None,
                         name=None):
    """ Computes the Monte-Carlo Convolution between two point clouds and updates
    the point locations

    Note:
      In the following, `A1` to `An` are optional batch dimensions.
      `C_in` is the number of input features.
      `C_out` is the number of output features.

    Args:
      features: A `float` `Tensor` of shape `[N_in, C_in]` or
        `[A1, ..., An,V, C_in]`.
      point_cloud_in: A 'PointCloud' instance, on which the features are
        defined.
      point_cloud_out: A `PointCloud` instance, on which the output
        features are defined.
      radius: A `float`, the convolution radius.
      neighborhood: A `Neighborhood` instance, defining the neighborhood
        with centers from `point_cloud_out` and neighbors in `point_cloud_in`.
        If `None` it is computed internally. (optional)
      bandwidth: A `float`, the bandwidth used in the kernel density
        estimation on the input point cloud. (optional)
      rays: not used.
      return_sorted: A `boolean`, if `True` the output tensor is sorted
        according to the batch_ids. (optional)
      return_padded: A `bool`, if 'True' the output tensor is sorted and
        zero padded. (optional)

    Returns:
      A `float` `Tensor` of shape
        `[N_out, C_out]`, if `return_padded` is `False`
      or
        `[A1, ..., An, V_out, C_out]`, if `return_padded` is `True`.

    """
    features = super().__call__(
        features,
        point_cloud_in,
        point_cloud_out,
        radius,
        neighborhood,
        bandwidth,
        return_sorted=False,
        return_padded=False,
        name=name)
    features, depth_update = tf.split(features, [-1, 1], axis=-1)
    zeros = tf.zeros_like(depth_update)
    point_cloud_out._points += tf.concat((zeros, zeros, depth_update), axis=-1)
    return features

  def _call_with_rays(self,
                      features,
                      point_cloud_in,
                      point_cloud_out,
                      radius,
                      rays,
                      neighborhood=None,
                      bandwidth=0.2,
                      name=None):
    """ Computes the Monte-Carlo Convolution between two point clouds and updates
    the point locations

    Note:
      In the following, `A1` to `An` are optional batch dimensions.
      `C_in` is the number of input features.
      `C_out` is the number of output features.

    Args:
      features: A `float` `Tensor` of shape `[N_in, C_in]` or
        `[A1, ..., An,V, C_in]`.
      point_cloud_in: A 'PointCloud' instance, on which the features are
        defined.
      point_cloud_out: A `PointCloud` instance, on which the output
        features are defined.
      radius: A `float`, the convolution radius.
      rays: A  `float` `Tensor` of shape `[N_out, 3]`.
      neighborhood: A `Neighborhood` instance, defining the neighborhood
        with centers from `point_cloud_out` and neighbors in `point_cloud_in`.
        If `None` it is computed internally. (optional)
      bandwidth: A `float`, the bandwidth used in the kernel density
        estimation on the input point cloud. (optional)

    Returns:
      A `float` `Tensor` of shape
        `[N_out, C_out]`, if `return_padded` is `False`
      or
        `[A1, ..., An, V_out, C_out]`, if `return_padded` is `True`.

    """
    features = super().__call__(
        features,
        point_cloud_in,
        point_cloud_out,
        radius,
        neighborhood,
        bandwidth,
        return_sorted=False,
        return_padded=False,
        name=name)
    features, depth_update = tf.split(features, [-1, 1], axis=-1)
    depth_update = tf.tanh(depth_update) * self.update_scale
    # print(tf.reduce_max(depth_update))
    point_cloud_out._points += depth_update * rays
    return features
