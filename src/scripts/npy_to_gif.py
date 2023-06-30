"""
Author: Arash Fatehi
Date:   28.06.2022
"""

import numpy as np
import imageio


file_path = '/data/afatehi/pet/data/200x200x64/negative/001ddbe6-c4a8-4aac-94e2-f37e97df6a67.npy'

data = np.load(file_path, allow_pickle=True)
data = data['patch']

input = data[0, :, :, :]
file_path = './result.gif'

with imageio.get_writer(file_path, mode='I') as writer:
    for index in range(input.shape[0]):
        writer.append_data(image[index])
