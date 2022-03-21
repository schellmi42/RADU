import numpy as np

def correlation_noise(data, relative=True, noise_level=0.02):
    """ Add random gaussian noise. (shot noise)
    Args:
      data: of shape `[B, H, W, C]`
    """
    if relative:
      noise = np.random.normal(size=data.shape, scale=noise_level * np.abs(data))
    else:
      noise = np.random.normal(size=data.shape, scale=noise_level)
    return data + noise
