import numpy as np
import torch
from pathlib import Path
from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_euler_angles


def convert_rot6d_to_euler(motion_data):
    num_cols = motion_data.shape[1]
    motion_euler = np.empty((motion_data.shape[0], 0))
    motion_euler = np.hstack((motion_euler, motion_data[:, 0: 3]))

    for i in range(3, num_cols, 6):
        sub_arr = motion_data[:, i: i + 6]
        matrix = rotation_6d_to_matrix(torch.from_numpy(sub_arr))
        euler = matrix_to_euler_angles(matrix, 'XYZ')

        motion_euler = np.hstack((motion_euler, euler.numpy()*180./3.1415926))

    return motion_euler


def save_motion_to_bvh_file(filename, motions):
    np.savetxt(Path(filename), convert_rot6d_to_euler(motions), '%s', ' ', '\n')
    print('Saved bvh path:', filename)
