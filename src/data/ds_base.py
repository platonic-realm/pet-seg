"""
Author: Arash Fatehi
Date:   04.07.2023
"""

# Python Imports
import os
import logging

# Library Imports
import sqlite3
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

# Local Imports


class BaseDataset(Dataset):
    def __init__(self,
                 _source_dir: str,
                 _patch_dimension: tuple,
                 _stride: tuple,
                 _cache_db_name: str = None):

        logging.info("Creating the Dataset from the directory: %s",
                     _source_dir)
        logging.info("Patch dimension: %s", _patch_dimension)
        logging.info("Stride: %s", _stride)

        # Local Variables
        self.commit_counter = 0

        # Storing the parameters
        self.source_dir = _source_dir
        self.patch_dimension = _patch_dimension
        self.stride = _stride

        # Loading the cache DB if available
        if _cache_db_name is None:
            _cache_db_name = '_'.join(str(item) for item in _patch_dimension)
        self.cache_db_path = os.path.join(_source_dir,
                                          "cache")
        logging.info("Setting cache database name to: %s", _cache_db_name)
        self.cache_db_exist = os.path.exists(self.cache_db_path)
        if not os.path.exists(self.cache_db_path):
            logging.info("Creating the cache directory: %s",
                         self.cache_db_path)
            os.mkdir(self.cache_db_path)

        self.cache_db_path = os.path.join(self.cache_db_path,
                                          f"{_cache_db_name}.db")
        self.cache_db_exist = os.path.exists(self.cache_db_path)
        self.init_cache_db()

        if not self.cache_db_exist:
            self.preprocess_cache_db()
        else:
            logging.info("Skipping the preprocssing, cache database exists")

    def __del__(self):
        if self.db_conn:
            logging.info("Closing the cache databse connection")
            self.db_conn.commit()
            self.db_conn.close()

    def init_cache_db(self):
        logging.info("Initializing the cache database connection")
        self.db_conn = sqlite3.connect(self.cache_db_path)
        self.db_cursor = self.db_conn.cursor()
        self.create_tables()

    def preprocess_cache_db(self):

        logging.info("Getting the list of sample directories")
        directory_list = self.get_directories()
        directory_list = sorted(directory_list)
        logging.info("Directory list length: %d", len(directory_list))

        logging.info("Enumerating on directory list for processing them")
        for dir_id, dir in enumerate(directory_list):
            logging.info("%d/%d - processing directory: %s",
                         dir_id, len(directory_list), dir)

            ct_data, ct_mean, ct_std, pet_data, pet_mean, \
                pet_std, seg_data, shape = self.load_sample_file(dir)

            logging.info("CT mean: %.2f, CT std: %.2f, PET mean: %.2f, PET std: %.2f",
                         ct_mean, ct_std, pet_mean, pet_std)

            self.insert_dir_record(dir_id, dir, ct_mean, ct_std,
                                   pet_mean, pet_std, shape[2])

            sample = np.stack((ct_data, pet_data, seg_data), axis=0)

            # Image's dimention is like (X, Y, Z)
            # Sample dimention is like (X, Y, Z)
            steps_per_x = int((shape[0] - self.patch_dimension[0]) // self.stride[0]) + 1
            steps_per_y = int((shape[1] - self.patch_dimension[1]) // self.stride[1]) + 1
            steps_per_z = int((shape[2] - self.patch_dimension[2]) // self.stride[2]) + 1

            logging.info("Steps X-Axis: %d, Y-Axis: %d, Z-Axis: %d",
                         steps_per_x, steps_per_y, steps_per_z)

            number_of_patches = steps_per_x * steps_per_y * steps_per_z

            logging.info("Number of patches: %d",
                         number_of_patches)

            for patch_id in range(number_of_patches):
                z_start = patch_id // (steps_per_x * steps_per_y)
                z_start = z_start * self.stride[2]

                # id for the xy plane instead of the image slides
                xy_id = patch_id % (steps_per_x * steps_per_y)

                y_start = xy_id // steps_per_x
                y_start = y_start * self.stride[1]

                x_start = xy_id % steps_per_x
                x_start = x_start * self.stride[0]

                patch = sample[:,
                               x_start: x_start + self.patch_dimension[0],
                               y_start: y_start + self.patch_dimension[1],
                               z_start: z_start + self.patch_dimension[2]]

                # offsets = np.array([x_start, y_start, z_start])

                is_empty = np.sum(patch) == 0
                has_positive = np.sum(patch[2, :, :, :]) > 0

                logging.debug("Processing patch number: %d", patch_id)
                logging.debug("x_start: %d, y_start: %d, z_start: %d",
                              x_start, y_start, z_start)
                logging.debug("Is empty: %r, Has positive labels: %r",
                              is_empty, has_positive)

                self.insert_patch_record(dir_id, patch_id,
                                         has_positive * 1, is_empty * 1,
                                         x_start, y_start, z_start)

        self.db_conn.commit

    def db_commit(self):
        self.commit_counter += 1
        if self.commit_counter > 100:
            self.commit_counter = 0
            logging.info("Commiting inserts into the cache database")
            self.db_conn.commit()

    def insert_dir_record(self,
                          _dir_id, _path,
                          _ct_mean, _ct_std,
                          _pet_mean, _pet_std,
                          _z):
        logging.debug("Inserting directory record to database")

        insert_query = f'''
            INSERT INTO dirs(path, dir_id, ct_mean, ct_std, pet_mean, pet_std, z)
            VALUES ("{_path}", {_dir_id}, {_ct_mean}, {_ct_std}, {_pet_mean}, {_pet_std}, {_z})
                       '''
        self.db_cursor.execute(insert_query)
        self.db_commit()

    def insert_patch_record(self,
                            _dir_id: int,
                            _patch_id: int,
                            _positive: int,
                            _empty: int,
                            _x_start: int,
                            _y_start: int,
                            _z_start: int):
        logging.debug("Inserting patch record to database, for dir_id: %d, patch_id: %d",
                      _dir_id, _patch_id)

        insert_query = f'''
            INSERT INTO patches(dir_id, patch_id, positive, empty, x_start, y_start, z_start)
            VALUES ({_dir_id}, {_patch_id}, {_positive}, {_empty}, {_x_start}, {_y_start}, {_z_start})
                       '''
        self.db_cursor.execute(insert_query)
        self.db_commit()

    def create_tables(self):
        create_table = '''
            CREATE TABLE IF NOT EXISTS dirs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dir_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                ct_mean REAL NOT NULL,
                ct_std REAL NOT NULL,
                pet_mean REAL NOT NULL,
                pet_std REAL NOT NULL,
                z INTEGER NOT NULL
                    )
                       '''
        self.db_cursor.execute(create_table)

        create_table = '''
            CREATE TABLE IF NOT EXISTS patches(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dir_id INTEGER NOT NULL,
                patch_id INTEGER NOT NULL,
                positive INTEGER NOT NULL,
                empty INTEGER NOT NULL,
                x_start INTEGER NOT NULL,
                y_start INTEGER NOT NULL,
                z_start INTEGER NOT NULL
                    )
                       '''

        self.db_cursor.execute(create_table)

        self.db_conn.commit

    def get_directories(self) -> list:
        directory_list = []
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith('CTres.nii'):
                    file_path = os.path.join(root, file)
                    directory_list.append(os.path.dirname(file_path))
        return directory_list

    @staticmethod
    def load_sample_file(_sample_directory: str):
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

        return ct_data, ct_mean, ct_std, pet_data, \
            pet_mean, pet_std, seg_data, ct_data.shape
