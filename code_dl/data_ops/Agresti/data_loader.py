'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from scipy.io import loadmat
import numpy as np

DATA_PATH = 'data/data_agresti/'

S1_test = {'path': DATA_PATH + 'S1/synthetic_dataset/test_set/test_',
           'path_GT': DATA_PATH + 'S1/synthetic_dataset/test_set/ground_truth/test_',
           'frequencies': ['20', '50', '60'],
           'num_scenes': 14,
           'zfill': 4}
S1_train = {'path': DATA_PATH + 'S1/synthetic_dataset/training_set/',
            'path_GT': DATA_PATH + 'S1/synthetic_dataset/training_set/ground_truth/',
            'frequencies': ['20', '50', '60'],
            'num_scenes': 40,
            'zfill': 4}
S2 = {'path': DATA_PATH + 'S2/S2/',
      'path_GT': None,
      'frequencies': ['20', '50', '60'],
      'num_scenes': 8,
      'zfill': 3}
S3 = {'path': DATA_PATH + 'S3/S3/',
      'path_GT': DATA_PATH + 'S3/S3/ground_truth/',
      'frequencies': ['10', '20', '30', '40', '50', '60'],
      'num_scenes': 8,
      'zfill': 3}
S4 = {'path': DATA_PATH + 'S4/real_dataset/',
      'path_GT': DATA_PATH + 'S4/real_dataset/ground_truth/',
      'frequencies': ['20', '50', '60'],
      'num_scenes': 8,
      'zfill': 3}
S5 = {'path': DATA_PATH + 'S5/S5/',
      'path_GT': DATA_PATH + 'S5/S5/ground_truth/',
      'frequencies': ['10', '20', '30', '40', '50', '60'],
      'num_scenes': 8,
      'zfill': 3}


def load_data(data_set='S3', frequencies=None, squeeze=False):
  """ Loads data of the datasets from publications of Agresti et al.
  Args:
    data_set: `str`, can be `'S3', 'S4', 'S5', 'S1_train', 'S1_test'`.
    frequencies: `list` of `str`, must be subset of the frequencies in the
      respective dataset.
      If `None` loads all.
    squeeze: `bool` to squeeze frequency dimension if only 1 freq.

  Returns:
    depths: `float`, shape `[S, H, W]`.
    tof_depths: `float`, shape `[S, F, H, W]`.
    amplitudes: `float`, shape `[S, F, H, W]`.
    intensities: `float`, shape `[S, F, H, W]`.
    PU: `bool`, shape `[S, F]`.
  """
  if ',' in data_set:
    # recursive call of multiple data sets
    data_sets = data_set.split(',')
    depths, tof_depths, amplitudes, intensities, PU = [], [], [], [], []
    for d in data_sets:
      depths_tmp, tof_depths_tmp, amplitudes_tmp, intensities_tmp, PU_tmp = load_data(d, frequencies, squeeze)
      depths.append(depths_tmp)
      tof_depths.append(tof_depths_tmp)
      amplitudes.append(amplitudes_tmp)
      intensities.append(intensities_tmp)
      PU.append(PU_tmp)
    depths = np.concatenate(depths, axis=0)
    tof_depths = np.concatenate(tof_depths, axis=0)
    amplitudes = np.concatenate(amplitudes, axis=0)
    intensities = np.concatenate(intensities, axis=0)
    PU = np.concatenate(PU, axis=0)
    return depths, tof_depths, amplitudes, intensities, PU

  if data_set == 'S1':
    # recursive call of train and test split
    depths_train, tof_depths_train, amplitudes_train, intensities_train, PU_train = load_data('S1_train', frequencies, squeeze)
    depths_test, tof_depths_test, amplitudes_test, intensities_test, PU_test = load_data('S1_test', frequencies, squeeze)
    depths = np.concatenate((depths_train, depths_test), axis=0)
    tof_depths = np.concatenate((tof_depths_train, tof_depths_test), axis=0)
    amplitudes = np.concatenate((amplitudes_train, amplitudes_test), axis=0)
    intensities = np.concatenate((intensities_train, intensities_test), axis=0)
    PU = np.concatenate((PU_train, PU_test), axis=0)
    return depths, tof_depths, amplitudes, intensities, PU
  elif data_set == 'S1_test':
    Set = S1_test
  elif data_set == 'S1_train':
    Set = S1_train
  elif data_set == 'S2':
    Set = S2
  elif data_set == 'S3':
    Set = S3
  elif data_set == 'S4':
    Set = S4
  elif data_set == 'S5':
    Set = S5
  else:
    raise ValueError('Unkowns dataset: ' + data_set + '!')
  if frequencies is None:
    frequencies = Set['frequencies']
  GT_available = Set['path_GT'] is not None
  depths = []
  tof_depths = []
  amplitudes = []
  intensities = []
  PU = []
  for scene in range(Set['num_scenes']):
    if GT_available:
      path = Set['path_GT'] + 'scene_' + str(scene).zfill(Set['zfill']) + '_'
      depths.append(loadmat(path + 'depth.mat')['depth_GT_radial'])
    for f in frequencies:

      path = Set['path'] + 'scene_' + str(scene).zfill(Set['zfill']) + '_MHz' + str(f) + '_'
      try:
        tof_depths.append(loadmat(path + 'depth.mat')['depth'])
        PU.append(False)
      except KeyError:
        tof_depths.append(loadmat(path + 'depth.mat')['depth_PU'])
        PU.append(True)
      amplitudes.append(loadmat(path + 'amplitude.mat')['amplitude'])
      intensities.append(loadmat(path + 'intensity.mat')['intensity'])
  H, W = tof_depths[0].shape
  if GT_available:
    depths = np.array(depths, dtype='f4').reshape(Set['num_scenes'], H, W)
  else:
    depths = None
  if squeeze and len(frequencies) == 1:
    shape_output = [Set['num_scenes'], H, W]
  else:
    shape_output = [Set['num_scenes'], len(frequencies), H, W]
  tof_depths = np.array(tof_depths, dtype='f4').reshape(shape_output)
  amplitudes = np.array(amplitudes, dtype='f4').reshape(shape_output)
  intensities = np.array(intensities, dtype='f4').reshape(shape_output)

  PU = np.array(PU, dtype=bool).reshape(shape_output[:-2])

  return depths, tof_depths, amplitudes, intensities, PU
