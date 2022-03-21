'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np
from .data_generator import data_generator
import code_dl.data_ops.Agresti
import code_dl.data_ops.FLAT

fov_degree = [65, 65]
fov_radian = fov_degree[0] / 180 * np.pi
focal_length = 1 / np.tan(fov_radian / 2)
