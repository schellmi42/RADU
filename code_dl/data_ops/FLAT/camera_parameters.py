""" Parameters of the FLAT data sets, as documented for Kinect2 camera in the FLAT code"""

resolution = [424, 512]
fov_real = [58.5, 46.6]
fov_synth = [42.13, 42.13]

# frequencies of the first and last 3 measurements, the middle measurements are not sinusoidal
frequencies = [40, 1e2 / 3.3, 1e2 / 1.7]

frequencies_2 = [80.1675385, 16.05444453, 120.44403642]

phase_offsets = [240, 120, 0]

phase_offsets_2 = [[150.93792, 30.93792, 270.93792],
                   [-83.11536, -203.11536, 36.88464],
                   [68.19318, -51.80682, 188.19318]]

phase_offsets_neg = [[-150.93792, -30.93792, -270.93792],
                     [83.11536, 203.11536, -36.88464],
                     [-68.19318, 51.80682, -188.19318]]

faulty_test = [7, 17, 31, 33, 34, 35, 64, 96, 98, 106]
