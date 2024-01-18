import numpy as np
import sys
import os
from os.path import join as pjoin


def mean_variance(data, name_list):
    """
        data from datadict in wMIB dataset
        data['motion'] : motion data which we will use to calc mean and var
        data['length'] : length of motion data
    """
    motion_list = []
    for item in name_list:
        motion_list.append(data[item]['motion'])

    motions = np.concatenate(motion_list, axis=0)
    print(motions.shape)

    Mean = motions.mean(axis=0)
    Std = motions.std(axis=0)
    Std[0: 3] = Std[0: 3].mean() / 1.
    Std[3: ] = Std[3: ].mean() / 1.

    np.save(pjoin('./data_loaders/wMIB/', 'Mean.npy'), Mean)
    np.save(pjoin('./data_loaders/wMIB/', 'Std.npy'), Std)

    return Mean, Std

