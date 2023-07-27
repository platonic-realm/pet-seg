"""
Author: Arash Fatehi
Date:   13.06.2023
"""

# Python Imports
import os
import logging

# Library Imports
import torch
import numpy as np
import nibabel as nib

# Local Imports
from src.data.ds_base import BaseDataset


class FDGDataset(BaseDataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 _source_directory,
                 _patch_dimension,
                 _stride,
                 _positive):

        super().__init__(_source_directory,
                         _patch_dimension,
                         _stride)

        self.positive = _positive
        if _positive:
            self.length = self.query_positive_patch_number()
        else:
            self.length = self.query_negative_patch_number()

        self.patches = self.query_patches()
        logging.info("Query the patch info from the database")

        self.files = {}
        self.last_dir = None
        self.buffer_count = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # path, x_start, y_start, z_start
        patch_info = self.patches[index]

        dir = patch_info[0].strip()

        self._remove_last_buffer(dir)

        ct_path = os.path.join(dir, 'CTres.nii')
        pet_path = os.path.join(dir, 'SUV.nii')
        seg_path = os.path.join(dir, 'SEG.nii')

        ct_tag = f"{dir}_ct"
        pet_tag = f"{dir}_pet"
        seg_tag = f"{dir}_seg"

        if ct_tag not in self.files:
            self.files[ct_tag] = np.asarray(nib.load(ct_path).dataobj)

        if pet_tag not in self.files:
            self.files[pet_tag] = np.asarray(nib.load(pet_path).dataobj)

        if seg_tag not in self.files:
            self.files[seg_tag] = np.asarray(nib.load(seg_path).dataobj)

        ct_data = self.files[ct_tag]
        pet_data = self.files[pet_tag]
        seg_data = self.files[seg_tag]

        x_start = patch_info[1]
        y_start = patch_info[2]
        z_start = patch_info[3]

        ct_data = ct_data[x_start: x_start + self.patch_dimension[0],
                          y_start: y_start + self.patch_dimension[1],
                          z_start: z_start + self.patch_dimension[2]]

        ct_data = ct_data / 1024

        pet_data = pet_data[x_start: x_start + self.patch_dimension[0],
                            y_start: y_start + self.patch_dimension[1],
                            z_start: z_start + self.patch_dimension[2]]

        seg_data = seg_data[x_start: x_start + self.patch_dimension[0],
                            y_start: y_start + self.patch_dimension[1],
                            z_start: z_start + self.patch_dimension[2]]

        sample = np.stack((ct_data, pet_data, seg_data), axis=0)
        sample = torch.from_numpy(sample)

        return sample

    def _remove_last_buffer(self, _dir):
        if self.last_dir is None or self.last_dir != _dir:
            logging.debug("Loading a new directory.")
            self.last_dir = _dir
            self.buffer_count += 1

        if self.buffer_count >= 20:
            self.files.clear()
            logging.debug("Dataset cache cleared.")
            self.buffer_count = 0

    def query_patches(self):
        query = "SELECT path, x_start, y_start, z_start FROM " + \
                f" patches LEFT JOIN dirs on patches.dir_id = dirs.dir_id WHERE positive = {self.positive*1}"
        self.db_cursor.execute(query)
        result = self.db_cursor.fetchall()
        return result

    def query_dirs(self):
        query = "SELECT DISTINCT path FROM(SELECT path FROM patches LEFT JOIN dirs " + \
                f" ON patches.dir_id = dirs.dir_id WHERE positive = {self.positive*1})"
        self.db_cursor.execute(query)
        result = self.db_cursor.fetchall()
        return result

    def query_positive_patch_number(self) -> int:
        query = "SELECT count(*) FROM patches WHERE positive = 1"
        self.db_cursor.execute(query)
        result = self.db_cursor.fetchall()
        return result[0][0]

    def query_negative_patch_number(self) -> int:
        query = "SELECT count(*) FROM patches WHERE positive = 0"
        self.db_cursor.execute(query)
        result = self.db_cursor.fetchall()
        return result[0][0]
