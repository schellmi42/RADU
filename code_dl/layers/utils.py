'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf


def mask(data, masks):
  """ Removes the masked data.
  Args:
    data: `float` of shape `[N, 3]`,
    masks: `bool`of shape `[N]`.
  Returns:
    `float` of shape `[N_filtered, 3]`.
  """
  return tf.boolean_mask(data, masks)


def demask_points(point_masked, masks):
  """ Fills the masked points with zeros.
  Args:
    point_masked: `float` of shape `[N_filtered, 3]`,
    masks: `bool`of shape `[N]`.
  Returns:
    `float` of shape `[N, 3]`.
  """
  shape = [tf.shape(masks)[0], 3]
  ids = tf.reshape(tf.range(0, tf.shape(masks)[0] * 3), shape)
  ids_filtered = tf.reshape(tf.boolean_mask(ids, masks), [-1])
  ids_filtered_sparse_format = tf.stack([ids_filtered // 3, ids_filtered % 3], axis=-1)
  points_sparse = tf.SparseTensor(indices=tf.cast(ids_filtered_sparse_format, dtype=tf.int64), values=tf.reshape(point_masked, [-1]), dense_shape=shape)
  return tf.sparse.to_dense(points_sparse)


def demask(data_masked, masks):
  """ Fills the masked data with zeros.
  Args:
    data_masked: `float` of shape `[N_filtered, C]`,
    masks: `bool`of shape `[N]`.
  Returns:
    `float` of shape `[N, C]`.
  """
  C = tf.shape(data_masked)[-1]
  masks = tf.reshape(masks, [-1])
  shape = [tf.shape(masks)[0], C]
  ids = tf.reshape(tf.range(0, tf.shape(masks)[0] * C), shape)
  ids_filtered = tf.reshape(tf.boolean_mask(ids, masks), [-1])
  ids_filtered_sparse_format = tf.stack([ids_filtered // C, ids_filtered % C], axis=-1)
  points_sparse = tf.SparseTensor(indices=tf.cast(ids_filtered_sparse_format, dtype=tf.int64), values=tf.reshape(data_masked, [-1]), dense_shape=shape)
  return tf.sparse.to_dense(points_sparse)
