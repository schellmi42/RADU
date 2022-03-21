'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf
from numpy import pi
from code_dl.data_ops.geom_ops_numpy import constants


# def intensity2depth_v2(amplitudes, frequency):
#     """ Computes ToF depth from intensity images. (in meter [m])
#         Loops around at `1/2 * f`
#     Args:
#         amplitudes: `floats` of shape `[B, H, W, 4]`.
#             ordered with offsets `[0°, 90°, 180°, 270°]`
#         frequency: `float` in GHz
#     Returns:
#         `floats` of shape `[B, H, W, 1]`
#     """
#     amplitudes = tf.convert_to_tensor(value=amplitudes, dtype=tf.float32)
#     frequency = tf.convert_to_tensor(value=frequency, dtype=tf.float32)
#     # phase offset on light path
#     delta_phi = tf.math.atan(amplitudes[:, :, :, 3] - amplitudes[:, :, :, 1], amplitudes[:, :, :, 0] - amplitudes[:, :, :, 2])
#     # resolve to strictly positive domain
#     delta_phi[delta_phi < 0] += 2 * pi
#     #print('phase ', delta_phi)
#     tof_depth = constants['lightspeed'] / (4 * pi * frequency) * delta_phi
#     return tf.expand_dims(tof_depth, axis=-1)


def compute_points_from_depth(depth, fov=[65, 65], return_rays=False):
    """ Projects 2.5D depths from camera coordinates to global coordinates.
    Args:
        depth: `float` shape `[B, H, W, 1]`
        fov: `float`, shape `[2]`, field of view of the camera in angles for height and width.
    Returns:
        points: shape `[B, H, W, 3]`
        rays: shape `[1, H, W, 3]`
    """
    depth = tf.convert_to_tensor(value=depth, dtype=tf.float32)
    fov = tf.convert_to_tensor(value=fov, dtype=tf.float32)
    fov_u = fov[0]
    fov_quot = fov[1] / fov[0]
    fov_u = fov_u / 180 * pi
    # B = depth.shape[0]
    H = tf.shape(depth)[1]
    W = tf.shape(depth)[2]
    u, v = tf.meshgrid(
            tf.linspace(-1., 1., H),
            tf.linspace(-fov_quot, fov_quot, W),
            indexing='ij')
    w = tf.ones(tf.shape(u)) * 1 / tf.math.tan(fov_u / 2)
    p = tf.stack((u, v, w), axis=-1)
    p = tf.expand_dims(p, axis=0)
    norm_p = tf.linalg.norm(p, axis=-1, keepdims=True)
    points = p * depth / norm_p
    if return_rays:
        rays = p / norm_p
        return points, rays
    return points


def points_to_camera_depth(points):
    """ Projects 3D depths from global coordinates to camera coordinates in 2.5D.
    Args:
        points: `float`, shape `[B, H, W, 3]`
    Returns:
        depths: `float`, shape `[B, H, W, 1]`
    """
    points = tf.convert_to_tensor(value=points, dtype=tf.float32)
    return tf.linalg.norm(points, axis=-1, keepdims=True)


def global_depth_to_camera_depth(depth_z, fov=[65, 65]):
    """ Projects 2.5D depths from global coordinates to camera coordinates.
    Args:
        depth: `float`, shape `[B, H, W, 1]`.
        fov: `float`, shape `[2]` in angles.
    Returns:
        `float`, shape `[B, H, W, 1]`.
    """
    depth_z = tf.convert_to_tensor(value=depth_z, dtype=tf.float32)
    fov = tf.convert_to_tensor(value=fov, dtype=tf.float32)
    fov = fov / 180 * pi

    # B = depth.shape[0]
    H = tf.shape(depth_z)[1]
    W = tf.shape(depth_z)[2]
    # angles per pixel
    u, v = tf.meshgrid(
            tf.linspace(-fov[0] / 2, fov[0] / 2, H),
            tf.linspace(-fov[1] / 2, fov[1] / 2, W),
            indexing='ij')
    # 3D position of pixel
    u = tf.reshape(tf.math.tan(u), [1, H, W, 1])
    v = tf.reshape(tf.math.tan(v), [1, H, W, 1])

    ray_length = tf.math.sqrt(u**2 + v**2 + 1) * depth_z
    return ray_length


def global_depth_to_camera_depth_on_cropped(depth_z, fov=[65, 65], shape=None, pos=(0, 0)):
    """ Projects 2.5D depths from global coordinates to camera coordinates.
    Args:
        depth: `float`, shape `[B, H, W, 1]`.
        fov: `float`, shape `[2]` in angles.
        shape: two `ints`, original height and width.
        pos: two `ints`, starting position of crop along x axis and y axis.
    Returns:
        `float`, shape `[B, H, W, 1]`.
    """
    depth_z = tf.convert_to_tensor(value=depth_z, dtype=tf.float32)
    fov = tf.convert_to_tensor(value=fov, dtype=tf.float32)
    pos = tf.convert_to_tensor(value=pos, dtype=tf.float32)
    fov = fov / 180 * pi
    pos_x, pos_y = pos
    H = tf.shape(depth_z)[1]
    W = tf.shape(depth_z)[2]
    if shape is None:
        shape = [H, W]
    else:
        shape = tf.convert_to_tensor(value=shape, dtype=tf.float32)

    # B = depth.shape[0]

    # adjust fov to crop:
    fov_x_start = (-fov[0] / 2) * (1 - 2 * pos_x / shape[0])
    fov_x_end = fov_x_start + fov[0] * tf.cast(H, tf.float32) / shape[0]
    fov_y_start = (-fov[1] / 2) * (1 - 2 * pos_y / shape[1])
    fov_y_end = fov_x_start + fov[1] * tf.cast(W, tf.float32) / shape[1]
    # angles per pixel
    u, v = tf.meshgrid(
            tf.linspace(fov_x_start, fov_x_end, H),
            tf.linspace(fov_y_start, fov_y_end, W),
            indexing='ij')
    # 3D position of pixel
    u = tf.reshape(tf.math.tan(u), [1, H, W, 1])
    v = tf.reshape(tf.math.tan(v), [1, H, W, 1])

    ray_length = tf.math.sqrt(u**2 + v**2 + 1) * depth_z
    return ray_length


def global_depth_to_camera_depth_with_rays(depth_z, rays):
    """ Projects 2.5D depths from global coordinates to camera coordinates.
    Args:
        depth: `float`, shape `[B, H, W, 1]`.
        rays: `float`, shape `[B, H, W, 3]`, normalized camera rays.
    Returns:
        `float`, shape `[B, H, W, 1]`.
    """
    depth_z = tf.convert_to_tensor(value=depth_z, dtype=tf.float32)
    rays = tf.convert_to_tensor(value=rays, dtype=tf.float32)

    ray_length = depth_z / tf.expand_dims(rays[:, :, :, 2], axis=-1)
    return ray_length


def camera_rays(shape, fov=[65, 65]):
    """ normalized camera ray directions

    Args:
        shape: `ints` of shape `[B, H, W]`.
        fov: `float`, field of view of the camera in angles.
    Returns:
        A `float` `np.array`of shape `[1, H, W, 3]`.
    """
    fov = tf.convert_to_tensor(value=fov, dtype=tf.float32)
    B, H, W = shape
    fov_u = fov[0]
    fov_quot = fov[1] / fov[0]
    fov_u = fov_u / 180 * pi
    u, v = tf.meshgrid(
            tf.linspace(-1, 1, H),
            tf.linspace(-fov_quot, fov_quot, W),
            indexing='ij')
    w = tf.ones(tf.shape(u)) * 1 / tf.math.tan(fov_u / 2)
    p = tf.stack((u, v, w), axis=-1)
    norm_p = tf.linalg.norm(p, axis=-1, keepdims=True)
    rays = p / norm_p
    return rays


if __name__ == '__main__':
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import argparse
    import imageio
    import numpy as np
    from geom_ops_numpy import depth_on_pixels
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
    points = compute_points_from_depth(depth)
    points_subset = np.reshape(points[0, ::stride, ::stride], [-1, 3])
    ax.scatter(points_subset[:, 0], points_subset[:, 2], -points_subset[:, 1], c=points_subset[:, 2])
    ax.set_title('global coordinates')

    points_to_restore = np.stack((points[0, :, :, 2], 2 * points[0, :, :, 2]), axis=0)
    print(points_to_restore.shape)
    tof_depth_restored = global_depth_to_camera_depth(np.expand_dims(points_to_restore, axis=-1))
    print(im[0:5, 0:5, 0])
    print(tof_depth_restored[0, 0:5, 0:5, 0])
    print(2 * im[0:5, 0:5, 0])
    print(tof_depth_restored[1, 0:5, 0:5, 0])

    plt.show()
