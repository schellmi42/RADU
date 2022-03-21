'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np
import tensorflow as tf
from code_dl.data_ops.geom_ops_numpy import constants


def phase_unwrapping_two_frequencies(depths_0, depths_1, frequencies, max_wraps, return_wrap_ids=False):
  """
  Args:
    depths_0: shape `[B, H, W]`
    depths_1: shape `[B, H, W]`
    frequencies: shape `[2]` in GHz
    max_wraps: shape `[2]`
  Returns
    unwrapped_depths_0: shape `[B, H, W]`
    unwrapped_depths_1: shape `[B, H, W]`

  """
  B, H, W = np.shape(depths_0)
  C = np.prod(np.array(max_wraps) + 1)
  max_depths = constants['lightspeed'] / (2 * np.array(frequencies))

  candidate_depths_0 = np.expand_dims(depths_0, axis=-1) + (np.arange(max_wraps[0] + 1)).reshape([1, 1, 1, -1]) * max_depths[0]
  candidate_depths_1 = np.expand_dims(depths_1, axis=-1) + (np.arange(max_wraps[1] + 1)).reshape([1, 1, 1, -1]) * max_depths[1]

  delta = np.abs(np.expand_dims(candidate_depths_0, axis=-2) - np.expand_dims(candidate_depths_1, axis=-1)).reshape([B, H, W, C])

  wrap_ids = np.argmin(delta, axis=-1)
  wrap_ids_0 = wrap_ids % (max_wraps[0] + 1)
  wrap_ids_1 = wrap_ids // (max_wraps[0] + 1)
  PU_0 = depths_0 + wrap_ids_0 * max_depths[0]
  PU_1 = depths_1 + wrap_ids_1 * max_depths[1]
  if return_wrap_ids:
    return PU_0, PU_1, wrap_ids_0, wrap_ids_1
  return PU_0, PU_1


def phase_unwrapping_two_frequencies_tf(depths_0, depths_1, frequencies, max_wraps, return_wrap_ids=False):
  """
  Args:
    depths_0: shape `[B, H, W]`
    depths_1: shape `[B, H, W]`
    frequencies: shape `[2]` in GHz
    max_wraps: shape `[2]`
  Returns
    unwrapped_depths_0: shape `[B, H, W]`
    unwrapped_depths_1: shape `[B, H, W]`

  """
  depths_0 = tf.convert_to_tensor(depths_0, dtype=tf.float32)
  depths_1 = tf.convert_to_tensor(depths_1, dtype=tf.float32)
  frequencies = tf.convert_to_tensor(frequencies, dtype=tf.float32)
  max_wraps = tf.convert_to_tensor(max_wraps, dtype=tf.int32)

  B, H, W = tf.shape(depths_0)
  C = tf.reduce_prod(max_wraps + 1)
  max_depths = constants['lightspeed'] / (2 * frequencies)

  candidate_depths_0 = tf.expand_dims(depths_0, axis=-1) + tf.reshape(tf.range(max_wraps[0] + 1, dtype=tf.float32), [1, 1, 1, -1]) * max_depths[0]
  candidate_depths_1 = tf.expand_dims(depths_1, axis=-1) + tf.reshape(tf.range(max_wraps[1] + 1, dtype=tf.float32), [1, 1, 1, -1]) * max_depths[1]

  delta = tf.reshape(tf.abs(tf.expand_dims(candidate_depths_0, axis=-2) - tf.expand_dims(candidate_depths_1, axis=-1)), [B, H, W, C])

  wrap_ids = tf.argmin(delta, axis=-1, output_type=tf.int32)
  wrap_ids_0 = wrap_ids % (max_wraps[0] + 1)
  wrap_ids_1 = wrap_ids // (max_wraps[0] + 1)
  PU_0 = depths_0 + tf.cast(wrap_ids_0, dtype=tf.float32) * max_depths[0]
  PU_1 = depths_1 + tf.cast(wrap_ids_1, dtype=tf.float32) * max_depths[1]
  if return_wrap_ids:
    return PU_0, PU_1, wrap_ids_0, wrap_ids_1
  return PU_0, PU_1
