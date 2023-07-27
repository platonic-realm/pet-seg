"""
Author: Arash Fatehi
Date:   27.06.2022
"""

# This scripts does 5 things:
# 1. Divides the voxel space of each sample into mulitiple patches
# 2. Stacks PET, CT, and label channels
# 3. Separates patches with or without positive samples
# 4. Perform data augmentation if needed
# 5. Standardizes or Normalizes the samples

import os
import uuid
import numpy as np
import nibabel as nib
import multiprocessing as mp

src_path = '/data/afatehi/pet/data/raw/FDG-PET-CT-Lesions/'
dst_path = '/data/afatehi/pet/data/200_64/'
# sample_dimension = [400, 400, 16]
sample_dimension = [200, 200, 64]
stride = [50, 50, 8]
allowed_empty = 100
max_process = 32


def get_directories(_source_directory: str) -> list:
    directory_list = []
    for root, dirs, files in os.walk(_source_directory):
        for file in files:
            if file.endswith('CTres.nii'):
                file_path = os.path.join(root, file)
                directory_list.append(os.path.dirname(file_path))
    return directory_list


def load_sample_files(_sample_directory: str) -> np.ndarray:
    ct_path = os.path.join(_sample_directory,
                           'CTres.nii')
    pet_path = os.path.join(_sample_directory,
                            'SUV.nii')
    seg_path = os.path.join(_sample_directory,
                            'SEG.nii')

    ct = nib.load(ct_path)
    pet = nib.load(pet_path)
    seg = nib.load(seg_path)

    ct_data = ct.get_fdata(dtype=np.float32)
    pet_data = pet.get_fdata(dtype=np.float32)
    seg_data = seg.get_fdata(dtype=np.float32)

    ########
    # Standardizing Pet and CT channels
    ct_mean = np.mean(ct_data)
    pet_mean = np.mean(pet_data)

    ct_std = np.std(ct_data)
    pet_std = np.std(pet_data)

    ct_data = (ct_data - ct_mean) / ct_std
    pet_data = (pet_data - pet_mean) / pet_std
    # Standardizing Pet and CT channels
    ########

    sample = np.stack((ct_data, pet_data, seg_data), axis=0)

    return sample


def process_sample(_sample_directory: str,
                   _sample_dimension: tuple,
                   _stride: list) -> np.ndarray:

    global allowed_empty

    sample = load_sample_files(_sample_directory)

    shape = sample.shape[1:]

    # Image's dimention is like (X, Y, Z)
    # Sample dimention is like (X, Y, Z)
    steps_per_x = int((shape[0] - _sample_dimension[0]) //
                      _stride[0]) + 1
    steps_per_y = int((shape[1] - _sample_dimension[1]) //
                      _stride[1]) + 1
    steps_per_z = int((shape[2] - _sample_dimension[2]) //
                      _stride[2]) + 1

    number_of_patches = steps_per_x * steps_per_y * steps_per_z

    for sample_id in range(number_of_patches):
        z_start = sample_id // (steps_per_x * steps_per_y)
        z_start = z_start * _stride[2]

        # Same as sample_id but in the xy plane instead of the image stack
        xy_id = sample_id % (steps_per_x * steps_per_y)

        y_start = xy_id // steps_per_x
        y_start = y_start * _stride[1]

        x_start = xy_id % steps_per_x
        x_start = x_start * _stride[0]

        patch = sample[:,
                       x_start: x_start + _sample_dimension[0],
                       y_start: y_start + _sample_dimension[1],
                       z_start: z_start + _sample_dimension[2]]

        # offsets = np.array([x_start, y_start, z_start])

        is_empty = np.sum(patch) == 0
        if is_empty:
            if allowed_empty > 0:
                allowed_empty -= 1
            else:
                return

        has_positive = np.sum(patch[2, :, :, :]) > 0
        sub_dir = 'positive' if has_positive else 'negative'
        dst_dir = os.path.join(dst_path, sub_dir)

        file_name = f"{uuid.uuid4()}"
        file_path = os.path.join(dst_dir, file_name)

        assert patch.shape == (3, 200, 200, 64), \
            f"Shape mismatch, shape: {patch.shape}"

        np.savez_compressed(f"{file_path}.data.npz", patch)

        print(f"Source: {_sample_directory}\n" +
              f"Source shape: {sample.shape}\n" +
              f"Sample ID: {sample_id}\n" +
              f"x_start: {x_start}, y_start: {y_start}, z_start: {z_start}\n" +
              f"Saved: {file_path}\n" +
              "-------------------------", flush=True)


def preprocess(_source_directory: str) -> None:
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        os.mkdir(os.path.join(dst_path, 'negative/'))
        os.mkdir(os.path.join(dst_path, 'positive/'))
    directory_list = get_directories(_source_directory)

    global sample_dimension
    global stride

    pool = mp.Pool(processes=max_process)

    parameters = []
    for dir_name in directory_list:
        parameters.append((dir_name, sample_dimension, stride))

    # for dir_name, sample_dimension, stride in parameters:
    #    process_sample(dir_name, sample_dimension, stride)

    pool.starmap(process_sample, parameters)

    pool.close()
    pool.join()


if __name__ == '__main__':
    preprocess(src_path)
