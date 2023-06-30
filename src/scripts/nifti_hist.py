"""
Author: Arash Fatehi
Date:   12.06.2022
"""

import numpy as np
import nibabel as nib
import os
import csv

path = '/data/afatehi/pet/data/FDG-PET-CT-Lesions/'


def load_nifti_file(_path: str) -> np.array:
    nifti_file = nib.load(_path)

    # Access the image data as a NumPy array
    data = nifti_file.get_fdata(dtype=np.float32)

    return data


def analyze(_path: str) -> None:
    file_list = []
    print('Listing the files')
    for root, dirs, files in os.walk(_path):
        for file in files:
            if file.endswith('CTres.nii'):
                file_path = os.path.join(root, file)
                file_list.append(file_path)

    print('Start analyzing')
    with open("result.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Path', 'Mean', 'Variance'])
        for file in file_list:
            image_data = load_nifti_file(file)
            mean = np.mean(image_data)
            variance = np.var(image_data)
            realtive_path = file.split(os.path.sep)[-3:]
            realtive_path = os.path.sep.join(realtive_path)
            writer.writerow([realtive_path, mean, variance])
            print(f"path: {realtive_path}, mean: {mean}, variance: {variance}")


analyze(path)
