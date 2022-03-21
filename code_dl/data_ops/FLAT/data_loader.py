'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import cv2
import numpy as np
from code_dl.data_ops.geom_ops_numpy import tof_depth_from_single_frequency
from code_dl.data_ops.FLAT.camera_parameters import  frequencies
from code_dl.data_ops.FLAT.camera_parameters import  phase_offsets_neg as phase_offsets

DATA_PATH = 'data/data_FLAT/'
DATA_PATH = '../denoising/data_FLAT/'

kinect_all = {'path': DATA_PATH + 'kinect/full/',
              'path_GT': DATA_PATH + 'kinect/gt/',
              'path_msk': DATA_PATH + 'kinect/msk/',
              'list': DATA_PATH + 'kinect/list/all.txt',
              'shape': [424, 512, 3, 3],
              'num_scenes': 1926,
              'type': 'correlations',
              'mask': DATA_PATH + 'kinect/msk'}

kinect_train = {'path': DATA_PATH + 'kinect/full/',
                'path_GT': DATA_PATH + 'kinect/gt/',
                'path_msk': DATA_PATH + 'kinect/msk/',
                'list': DATA_PATH + 'kinect/list/train.txt',
                'shape': [424, 512, 3, 3],
                'num_scenes': 1104,
                'type': 'correlations',
                'mask': DATA_PATH + 'kinect/msk'}

kinect_test = {'path': DATA_PATH + 'kinect/full/',
               'path_GT': DATA_PATH + 'kinect/gt/',
               'path_msk': DATA_PATH + 'kinect/msk/',
               'list': DATA_PATH + 'kinect/list/test.txt',
               'shape': [424, 512, 3, 3],
               'num_scenes': 104,
               'type': 'correlations',
               'mask': DATA_PATH + 'kinect/msk'}

kinect_val = {'path': DATA_PATH + 'kinect/full/',
              'path_GT': DATA_PATH + 'kinect/gt/',
              'path_msk': DATA_PATH + 'kinect/msk/',
              'list': DATA_PATH + 'kinect/list/val.txt',
              'shape': [424, 512, 3, 3],
              'num_scenes': 50,
              'type': 'correlations',
              'mask': DATA_PATH + 'kinect/msk'}

deeptof = {'path': DATA_PATH + 'deeptof/full/',
           'path_GT': DATA_PATH + 'kinect/gt/',
           'path_msk': DATA_PATH + 'kinect/msk/',
           'list': DATA_PATH + 'deeptof/list/all.txt',
           'shape': [424, 512],
           'num_scenes': 104,
           'type': 'tof',
           'mask': DATA_PATH + 'kinect/msk'}

phasor = {'path': DATA_PATH + 'phasor/full/',
          'path_GT': DATA_PATH + 'kinect/gt/',
          'path_msk': DATA_PATH + 'kinect/msk/',
          'list': DATA_PATH + 'phasor/list/all.txt',
          'shape': [424, 512],
          'num_scenes': 104,
          'type': 'tof',
          'mask': DATA_PATH + 'kinect/msk'}


def load_scene_names(data_set='kinect_train'):
  """ Loads data of the datasets from publications of Agresti et al.
  Args:
    data_set: `str`, can be `'kinect', 'kinect_train', 'kinect_val', 'kinect_test'`,
      `'deeptof', 'phasor'`.

  Returns:
    scenes = `list` of scene names.
  """
  if data_set == 'kinect':
    Set = kinect_all
  elif data_set == 'kinect_train':
    Set = kinect_train
  elif data_set == 'kinect_val':
    Set = kinect_val
  elif data_set == 'kinect_test':
    Set = kinect_test
  elif data_set == 'deeptof':
    Set = deeptof
  elif data_set == 'phasor':
    Set = phasor
  else:
    raise ValueError('Unkowns dataset: ' + set + '!')

  scenes = []
  with open(Set['list']) as f:
    for line in f:
      scenes.append(line.replace('\n', ''))

  return np.array(scenes, dtype=str)


def load_data(data_set, scenes):
  """ Loads data of the datasets from FLAT.
  Args:
    data_set: `str`, can be `'kinect', 'kinect_train', 'kinect_val', 'kinect_test'`,
      `'deeptof', 'phasor'`.
    scenes: `list` of a batch of scenes.

  Returns:
    depths_gt: `float`, shape `[B, H, W]`.
    tof_depths: `float`, shape `[B, H, W, 3]`.
    correlations: `float`, shape `[B, H, W, 3, 3]`.
    amplitudes: `float`, shape `[B, H, W, 3]`.
    intensities: `float`, shape `[B, H, W, 3]`.
    masks: `float`, shape `[B, H, W]`.
  """
  if data_set == 'kinect':
    Set = kinect_all
  elif data_set == 'kinect_train':
    Set = kinect_train
  elif data_set == 'kinect_val':
    Set = kinect_val
  elif data_set == 'kinect_test':
    Set = kinect_test
  elif data_set == 'deeptof':
    Set = deeptof
  elif data_set == 'phasor':
    Set = phasor
  else:
    raise ValueError('Unkowns dataset: ' + set + '!')
  shape = Set['shape']
  gt_shape = [Set['shape'][0] * 4, Set['shape'][1] * 4]

  depths_gt = []
  correlations = []
  masks = {}
  masks['background'] = []
  masks['edge'] = []
  masks['noise'] = []
  masks['reflection'] = []
  for scene in scenes:
    # load ground truth
    filename = Set['path_GT'] + scene
    with open(filename, 'rb') as f:
      data = np.fromfile(f, dtype=np.float32)
    depth_gt = np.reshape(data, gt_shape)
    # downsample ground truth
    # this one is deprecated in scipy: depths_gt.append(imresize(depth_gt, shape[:2], mode='F'))
    depths_gt.append(cv2.resize(src=depth_gt, dsize=(shape[1], shape[0])))
    #  load correlations
    filename = Set['path'] + scene
    with open(filename, 'rb') as f:
      data = np.fromfile(f, dtype=np.int32)
    correlations.append(np.reshape(data, shape).astype(np.float32))
    # load mask
    filename = Set['path_msk'] + scene
    with open(filename, 'rb') as f:
      data = np.fromfile(f, dtype=np.float32)
    data = np.reshape(data, [shape[0], shape[1], 4])
    masks['background'].append(data[:, :, 0])
    masks['edge'].append(data[:, :, 1])
    masks['noise'].append(data[:, :, 2])
    masks['reflection'].append(data[:, :, 3])
    # masks.append(depth_gt != 0)
  depths_gt = np.array(depths_gt)
  correlations = np.array(correlations)
  masks['background'] = np.array(masks['background'], dtype=np.float32)
  masks['edge'] = np.array(masks['background'], dtype=np.float32)
  masks['noise'] = np.array(masks['background'], dtype=np.float32)
  masks['reflection'] = np.array(masks['background'], dtype=np.float32)
  # fix correlations data
  sign_fix = np.concatenate(
    (np.tile([1, -1], shape[0] // 4), np.tile([-1, 1], shape[0] // 4)), axis=0
                            ).reshape([1, -1, 1, 1, 1])

  correlations = - correlations * sign_fix

  tof_depths_40 = tof_depth_from_single_frequency(correlations[:, :, :, 0, :], frequencies[0] / 1e3, phase_offsets[0])
  tof_depths_30 = tof_depth_from_single_frequency(correlations[:, :, :, 1, :], frequencies[1] / 1e3, phase_offsets[1])
  tof_depths_58 = tof_depth_from_single_frequency(correlations[:, :, :, 2, :], frequencies[2] / 1e3, phase_offsets[2])
  tof_depths = np.stack((tof_depths_40, tof_depths_30, tof_depths_58), axis=-1)

  # amplitudes and intensities
  phase_offsets_broadcast = np.reshape(phase_offsets, [1, 1, 1, 3, 3])
  I = np.sum(np.sin(phase_offsets_broadcast) * correlations, axis=-1)
  Q = np.sum(np.cos(phase_offsets_broadcast) * correlations, axis=-1)

  amplitudes = 2 / 3 * np.sqrt(I**2 + Q**2)
  intensities = np.sum(correlations, axis=-1) / len(phase_offsets[0])

  return depths_gt, tof_depths, correlations, amplitudes, intensities, masks


def load_full_data(data_set):
  """ Loads data of the datasets from FLAT.
  Args:
    data_set: `str`, can be `'kinect', 'kinect_train', 'kinect_val', 'kinect_test',
      'deeptof', 'phasor'`.
    scenes: `list` of a batch of scenes.

  Returns:
    depths_gt: `float`, shape `[S, H, W]`.
    correlations: `float`, shape `[S, H, W, 9]`.
  """
  if data_set == 'kinect':
    Set = kinect_all
  elif data_set == 'kinect_train':
    Set = kinect_train
  elif data_set == 'kinect_val':
    Set = kinect_val
  elif data_set == 'kinect_test':
    Set = kinect_test
  elif data_set == 'deeptof':
    Set = deeptof
  elif data_set == 'phasor':
    Set = phasor
  else:
    raise ValueError('Unkowns dataset: ' + set + '!')
  shape = Set['shape']
  gt_shape = [Set['shape'][0] * 4, Set['shape'][1] * 4]

  scenes = []
  with open(Set['list']) as f:
    for line in f:
      scenes.append(line.replace('\n', ''))

  depths_gt = []
  correlations = []
  for scene in scenes:
    # load ground truth
    filename = Set['path_GT'] + scene
    with open(filename, 'rb') as f:
      data = np.fromfile(f, dtype=np.float32)
    depth_gt = np.reshape(data, gt_shape)
    # downsample ground truth
    # this one is deprecated in scipy: depths_gt.append(imresize(depth_gt, shape[:2], mode='F'))
    depths_gt.append(cv2.resize(src=depth_gt, dsize=(shape[1], shape[0])))
    #  load correlations
    filename = Set['path'] + scene
    with open(filename, 'rb') as f:
      data = np.fromfile(f, dtype=np.int32)
    correlations.append(np.reshape(data, shape).astype(np.float32))
  depths_gt = np.array(depths_gt, dtype=np.float32)
  correlations = np.array(correlations, dtype=np.float32)
  # fix correlations data
  sign_fix = np.concatenate(
    (np.tile([1, -1], shape[0] // 4), np.tile([-1, 1], shape[0] // 4)), axis=0
                            ).reshape([1, -1, 1, 1])

  return depths_gt, - correlations * sign_fix
