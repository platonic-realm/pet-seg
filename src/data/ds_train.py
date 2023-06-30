"""
Author: Arash Fatehi
Date:   13.06.2023
"""

# Python Imports
import os

# Library Imports
import torch
import numpy as np
from torch.utils.data import Dataset


# Local Imports


sample_shape = (3, 400, 400, 16)


class FDGDataset(Dataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, _source_directory):

        self.source_directory = _source_directory

        self.list = os.listdir(self.source_directory)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        file_path = self.list[index]

        file_path = os.path.join(self.source_directory, file_path)

        sample = np.load(file_path)
        sample = torch.from_numpy(sample)

        assert sample.shape == sample_shape, \
            f"Shape mismatch in file: {file_path}," +\
            f" the shape is: {sample.shape}"

        return sample
