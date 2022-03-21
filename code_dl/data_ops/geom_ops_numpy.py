'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np

"""
These are all numpy functions.
"""


# lightspeed in nanoseconds
constants = {'lightspeed': 0.299792458}
# in GHz
constants['frequencies'] = np.array([20, 50, 70]) / 1e3
constants['frequencies_str'] = ['20MHz', '50MHz', '70MHz']
# in radians
constants['phase_offsets'] = [0, np.pi / 2, np.pi, 3 / 2 * np.pi]
# in nanoseconds
constants['exposure_time'] = 0.01 / constants['lightspeed']


def tof_depth_from_single_frequency(correlations, frequency, phase_offsets=[0, 120, 240]):
  """ Computes depth from single frequency measurements. Contains phase wrapping.
  Args:
    correlations: `float` of shape `[B, H, W, P]`.
    frequency: `float` in GHz.
    phase_offsets: `float` of shape '[P]` in degree.
  Returns:
    `float` of shape `[B, H, W]`.
  """
  phase_offsets = (np.array(phase_offsets) / 180 * np.pi).reshape([1, 1, 1, -1])
  I = np.sum(-np.sin(phase_offsets) * correlations, axis=-1)
  Q = np.sum(np.cos(phase_offsets) * correlations, axis=-1)
  delta_phi = np.arctan2(I, Q)
  # resolve to positive domain
  delta_phi[delta_phi < 0] += 2 * np.pi
  #print('phase ', delta_phi)
  depth = constants['lightspeed'] / (4 * np.pi * frequency) * delta_phi
  return depth


def correlation2depth(correlations, frequency):
  """ Computes ToF depth from intensity images. (in meter [m])
    Loops around at `1/2 * f`
    Optimized version for four phase measurements at 90° step offsets.
  Args:
    correlations: `floats` of shape `[B, H, W, 4]`.
      ordered with offsets `[0°, 90°, 180°, 270°]`
    frequency: `float` in GHz
  Returns:
    `floats` of shape `[B, H, W, 1]`
  """
  # phase offset on light path
  delta_phi = np.arctan2(correlations[:, :, :, 3] - correlations[:, :, :, 1], correlations[:, :, :, 0] - correlations[:, :, :, 2])
  # resolve to positive domain
  delta_phi[delta_phi < 0] += 2 * np.pi
  #print('phase ', delta_phi)
  tof_depth = constants['lightspeed'] / (4 * np.pi * frequency) * delta_phi
  return np.expand_dims(tof_depth, axis=-1)


def compute_points_from_depth(depth, fov=[65, 65], return_rays=False):
  """ Projects 2.5D depths from camera coordinates to global coordinates.
  Args:
    depth: `float` shape `[B, H, W, 1]`
    fov: `float`, shape `[2]`, field of view of the camera in angles for height and width.
  Returns:
    points: shape `[B, H, W, 3]`
    rays: shape `[1, H, W, 3]`
  """
  fov_u = fov[0]
  fov_quot = fov[1] / fov[0]
  fov_u = fov_u / 180 * np.pi
  # B = depth.shape[0]
  H = depth.shape[1]
  W = depth.shape[2]
  u, v = np.meshgrid(
      np.linspace(-1, 1, H),
      np.linspace(-fov_quot, fov_quot, W),
      indexing='ij')
  w = np.ones(u.shape) * 1 / np.tan(fov_u / 2)
  p = np.stack((u, v, w), axis=-1)
  p = np.expand_dims(p, axis=0)
  norm_p = np.linalg.norm(p, axis=-1, keepdims=True)
  points = p * depth / norm_p
  if return_rays:
    rays = p / norm_p
    return points, rays
  return points


def global_depth_to_camera_depth(points):
  """ Projects 3D depths from global coordinates to camera coordinates in 2.5D.
  Args:
    points: `float` shape `[B, H, W, 3]`
  Returns:
    depths: `float` shape `[B, H, W, 1]`
  """
  return np.linalg.norm(points, axis=-1, keepdims=True)


def camera_rays(shape, fov=[65, 65]):
  """ normalized camera ray directions

  Args:
    shape: `ints` of shape `[B, H, W]`.
    fov: `float`, field of view of the camera in angles.
  Returns:
    A `float` `np.array`of shape `[1, H, W, 3]`.
  """
  B, H, W = shape
  fov_u = fov[0]
  fov_quot = fov[1] / fov[0]
  fov_u = fov_u / 180 * np.pi
  u, v = np.meshgrid(
      np.linspace(-1, 1, H),
      np.linspace(-fov_quot, fov_quot, W),
      indexing='ij')
  w = np.ones(u.shape) * 1 / np.tan(fov_u / 2)
  p = np.stack((u, v, w), axis=-1)
  norm_p = np.linalg.norm(p, axis=-1, keepdims=True)
  rays = p / norm_p
  return rays


def depth_on_pixels(depth):
  """ Turns 2.5D into points using pixel ids as coordinates.
  For 3D Plotting
  """
  H = depth.shape[0]
  W = depth.shape[1]
  x, y = np.meshgrid(np.arange(0, H), np.arange(0, W), indexing='ij')
  z = depth
  # x = x.reshape(-1)
  # y = y.reshape(-1)
  # z = z.reshape(-1)
  return np.stack((x, y, z), axis=-1)


def reconstruct_correlations(amplitudes, intensities, tof_depths, frequency, phase_offsets=[0, 90, 180, 270]):
  """
  Args:
    amplitudes: `float` of shape `[B, H, W]`.
    intensities:  `float` of shape `[B, H, W]`.
    tof_depths:  `float` of shape `[B, H, W]`.
    frequenciy: 'float` in GHz.
    phase_offsets: `floats` of shape `[4]`.
  Returns:
    `float` array of shape `[B, H, W, 4]`
  """

  amplitudes = np.expand_dims(amplitudes, axis=-1)
  intensities = np.expand_dims(intensities, axis=-1)
  tof_depths = np.expand_dims(tof_depths, axis=-1)
  phase_offsets = (np.array(phase_offsets, dtype=np.float32) * np.pi / 180).reshape([1, 1, 1, -1])

  delta_phi = (4 * np.pi * frequency) / constants['lightspeed'] * tof_depths
  delta_phi[delta_phi < 0] += 2 * np.pi
  return intensities + amplitudes * np.cos(delta_phi + phase_offsets)


def amplitude_and_intensity_from_correlation(corr):
  """ Computes amplitude and intensities for correlations measured at [0, 90, 180, 270] degrees.
  Args:
    corr: leading dimension is `4`.
  Returns:
    amp:  shape of corr, except for first dimension.
    int: shape of corr, except for first dimension.
  """
  amp = 0.5 * np.sqrt((corr[0] - corr[2])**2 + (corr[1] - corr[3])**2)
  int = np.sum(corr, axis=0) / 4
  return amp, int


def amplitude_and_intensity_from_correlationv2(correlations, phase_offsets):
  """
  Args:
    correlations: shape `[B, H, W, P]`
  Returns:
  amplitudes: shape `[B, H, W]`
  intensities: shape `[B, H, W]`
  """
  phase_offsets = np.reshape(phase_offsets, [1, 1, 1, -1])
  I = np.sum(np.sin(phase_offsets) * correlations, axis=3)
  Q = np.sum(np.cos(phase_offsets) * correlations, axis=3)
  amplitudes = 0.5 * np.sqrt(I**2 + Q**2)
  intensities = np.sum(correlations, axis=3) / len(phase_offsets)
  return amplitudes, intensities


if __name__ == '__main__':
  import imageio
  from mpl_toolkits import mplot3d
  import matplotlib.pyplot as plt
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--f')
  args = parser.parse_args()
  if not args.f.endswith('hdr'):
    print(args.f)
    raise ValueError('invalid file format')
  im = imageio.imread(args.f, format='HDR-FI')
  i = 1
  stride = 5

  fig = plt.figure(figsize=plt.figaspect(0.5))
  ax = fig.add_subplot(1, 2, 1, projection='3d')
  points = depth_on_pixels(im[:, :, i])
  points = np.reshape(points[::stride, ::stride], [-1, 3])
  ax.scatter(points[:, 0], points[:, 2], -points[:, 1], c=points[:, 2])
  ax.set_title('camera coordinates')

  ax = fig.add_subplot(1, 2, 2, projection='3d')
  depth = im[:, :, i].reshape([1, 600, 600, 1])
  points = compute_points_from_depth(depth)[0]
  points_subset = np.reshape(points[::stride, ::stride], [-1, 3])
  ax.scatter(points_subset[:, 0], points_subset[:, 2], -points_subset[:, 1], c=points_subset[:, 2])
  ax.set_title('global coordinates')

  plt.show()
