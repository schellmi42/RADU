'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import os
import numpy as np
import h5py

DATA_PATH = 'data/data_CB/'

S_train = {'path': DATA_PATH,
           'list': 'train.txt',
           'frequencies': ['20', '50', '70'],
           'shape': [600, 600, 4],
           'num_frames': 50,
           'num_scenes': 116}

S_val = {'path': DATA_PATH,
         'list': 'val.txt',
         'frequencies': ['20', '50', '70'],
         'shape': [600, 600, 4],
         'num_samples': [30, 20, 20, 20, 20],
         'num_scenes': 13}

S_test = {'path': DATA_PATH,
          'list': 'test.txt',
          'frequencies': ['20', '50', '70'],
          'shape': [240, 600, 4],
          'num_samples': [30, 20, 20, 20, 20],
          'num_scenes': 13}


def load_filenames(data_set):
  """ Loads data of the datasets from publications of Agresti et al.
  Args:
    data_set: `str`, can be `'S3', 'S4', 'S5', 'S1_train', 'S1_test'`.
  Returns:
    filenames: `float`, shape `[S, 50]`.
  """

  if 'train' in data_set:
    Set = S_train
  elif 'val' in data_set:
    Set = S_val
  elif 'test' in data_set:
    Set = S_test
  scenes = []
  if 'dummy' in data_set:
    Set['path'] = 'data_dummy/'
  with open(Set['path'] + Set['list'], 'r') as inFile:
    for line in inFile:
      scenes.append(line.replace('\n', ''))
  frames = []
  for scene in scenes:
    frames.append([Set['path'] + scene + '/' + str(i).zfill(3) + '_render_' for i in range(50)])
  return np.array(frames, dtype=str)


def load_batch(files, frequencies=None, slice_id=None):
  """ Loads data of the datasets from publications of Agresti et al.
  Args:
    files: `list`of `str`.
    frequencies: `list` of `str`, must be subset of the frequencies in the
      respective dataset.
      If `None` loads all.
    slice_ids: an `int` [0,1,2], to slice the color channel for different materials.

  Returns:
    depths: `float`, shape `[S, H, W]`.
    tof_depths: `float`, shape `[S, F, H, W]`.
    correlations: `float`, shape `[S, F, 4, H, W]`.
    amplitudes: `float`, shape `[S, F, H, W]`.
    intensities: `float`, shape `[S, F, H, W]`.
  """
  import imageio
  depths = []
  tof_depths = []
  correlations = []
  amplitudes = []
  intensities = []
  if frequencies is None:
    frequencies = ['20', '50', '70']
  for frame in files:
    if slice_id is None:
      depths.append(imageio.imread(frame + 'depth.hdr', format='HDR-FI'))
      tof_depths.append(
          [imageio.imread(frame + str(freq) + 'MHz_ToF.hdr', format='HDR-FI') for freq in frequencies]
      )
      correlations.append(
          [[imageio.imread(frame + str(freq) + 'MHz_phase0.hdr', format='HDR-FI'),
            imageio.imread(frame + str(freq) + 'MHz_phase1.hdr', format='HDR-FI'),
            imageio.imread(frame + str(freq) + 'MHz_phase2.hdr', format='HDR-FI'),
            imageio.imread(frame + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')]
           for freq in frequencies]
      )
    else:
      depths.append(imageio.imread(frame + 'depth.hdr', format='HDR-FI')[:, :, slice_id])
      tof_depths.append(
          [imageio.imread(frame + str(freq) + 'MHz_ToF.hdr', format='HDR-FI')[:, :, slice_id] for freq in frequencies]
      )
      correlations.append(
          [[imageio.imread(frame + str(freq) + 'MHz_phase0.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame + str(freq) + 'MHz_phase1.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame + str(freq) + 'MHz_phase2.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')[:, :, slice_id]]
           for freq in frequencies]
      )
  depths = np.array(depths, dtype=np.float32)
  tof_depths = np.array(tof_depths, dtype=np.float32)
  correlations = np.array(correlations, dtype=np.float32)
  amplitudes = 0.5 * np.sqrt((correlations[:, :, 0] - correlations[:, :, 2])**2 + \
                             (correlations[:, :, 3] - correlations[:, :, 1])**2)
  intensities = np.mean(correlations, axis=2)

  return depths, tof_depths, correlations, amplitudes, intensities
