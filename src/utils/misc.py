"""
Author: Arash Fatehi
Date:   27.06.2023
"""

# Python Imports
import logging
import sys
import os
import shutil
from pathlib import Path

# Library Imports
import torch


def basic_logger() -> None:
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level='INFO',
                        format=log_format)


def configure_logger(_configs: dict) -> None:
    LOG_LEVEL = _configs['logging']['log_level']
    root_path = _configs['root_path']
    log_file = os.path.join(root_path,
                            _configs['trainer']['result_path'],
                            _configs['logging']['log_file'])
    log_std = _configs['logging']['log_std']

    handlers = []
    if log_file is not None:
        log_file = Path(log_file)
        create_dirs_recursively(log_file)
        handlers.append(logging.FileHandler(log_file))
    if log_std:
        handlers.append(logging.StreamHandler(sys.stdout))

    log_format = "%(asctime)s [%(levelname)s] %(message)s"

    logging.basicConfig(level=LOG_LEVEL,
                        format=log_format,
                        handlers=handlers)

    logging.info("Log Level: %s", LOG_LEVEL)


def create_dirs_recursively(_path: str):
    dir_path = os.path.dirname(_path)
    path: Path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)


def copy_directory(_source_dir, _dest_dir, _exclude_list: list):
    for item in os.listdir(_source_dir):
        source_path = os.path.join(_source_dir, item)
        dest_path = os.path.join(_dest_dir, item)
        if item not in _exclude_list:
            if os.path.isdir(source_path):
                shutil.copytree(source_path, dest_path)
            else:
                shutil.copy2(source_path, dest_path)


def to_numpy(_gpu_tensor):
    if torch.is_tensor(_gpu_tensor):
        return _gpu_tensor.clone().detach().to('cpu').numpy()

    return _gpu_tensor
