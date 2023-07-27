"""
Author: Arash Fatehi
Date:   03.05.2022
"""


# Python Imports
import os
import logging
import subprocess
import shutil
import yaml

# Library Imports
import torch

# Local Imports
from src.utils.misc import create_dirs_recursively, copy_directory
from src.utils.args import read_configurations, summerize
from src.train.trainer_unet3d import Unet3DTrainer
from src.train.trainer_unet3d_me import Unet3DMETrainer
from src.infer.inference import Inference
from src.utils.misc import configure_logger


def experiment_exists(_root_path, _name) -> bool:
    result = False
    if os.path.exists(_root_path):
        for item in os.listdir(_root_path):
            if os.path.isdir(
                    os.path.join(_root_path,
                                 item)) and item == _name:
                result = True
    return result


def list_experiments(_root_path):
    print("Experiments:")
    if os.path.exists(_root_path):
        for item in sorted(os.listdir(_root_path)):
            if os.path.isdir(os.path.join(_root_path,
                                          item)):
                print(f"* {item}")


def list_snapshots(_name,
                   _root_path):
    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)
    print(f"Snapshots of {_name}:")
    snapshots_path = \
        os.path.join(_root_path, _name, 'results-train/snapshots/')
    if os.path.exists(snapshots_path):
        for item in sorted(os.listdir(snapshots_path)):
            if os.path.isfile(os.path.join(snapshots_path,
                                           item)):
                print(f"* {item}")


def infer_experiment(_name: str,
                     _root_path: str,
                     _snapshot: str,
                     _batch_size: int,
                     _sample_dimension: list,
                     _stride: list,
                     _scale: int,
                     _channel_map: list):

    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)

    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configurations(configs_path)

    inference_root_path = os.path.join(_root_path, _name, 'results-infer')
    create_dirs_recursively(os.path.join(inference_root_path, 'dummy'))

    inference_tag =\
        f"{_snapshot}_{''.join(_sample_dimension)}_{''.join(_stride)}_{_scale}"

    _sample_dimension = [int(item) for item in _sample_dimension]
    _stride = [int(item) for item in _stride]
    _channel_map = [int(item) for item in _channel_map]
    _scale = int(_scale)
    _batch_size = int(_batch_size)

    inference_result_path = os.path.join(inference_root_path, inference_tag)
    if os.path.exists(inference_result_path):
        answer = input("Ineference already exists,"
                       " overwrite? (y/n) [default=n]: ")
        if answer.lower() == "y":
            shutil.rmtree(inference_result_path)
        else:
            return
    create_dirs_recursively(os.path.join(inference_result_path, 'dummy'))

    configs['inference']['model']['name'] =\
        configs['trainer']['model']['name']

    configs['inference']['model']['feature_maps'] =\
        configs['trainer']['model']['feature_maps']

    configs['inference']['snapshot_path'] =\
        os.path.join(_root_path,
                     _name,
                     'results-train/snapshots/',
                     _snapshot)

    configs['inference']['result_dir'] = inference_result_path

    configs['inference']['inference_ds']['path'] =\
        os.path.join(_root_path,
                     _name,
                     'datasets/',
                     'ds_test/')

    configs['inference']['inference_ds']['batch_size'] = _batch_size

    configs['inference']['inference_ds']['sample_dimension'] =\
        _sample_dimension

    configs['inference']['inference_ds']['pixel_stride'] = _stride

    configs['inference']['inference_ds']['scale_factor'] = _scale

    configs['inference']['inference_ds']['workers'] =\
        configs['trainer']['train_ds']['workers']

    inference_configs = configs['inference']

    with open(os.path.join(inference_result_path, 'configs.yaml'), 'w',
              encoding='UTF-8') as config_file:
        yaml.dump(inference_configs,
                  config_file,
                  default_flow_style=None,
                  sort_keys=False)

    inference = Inference(configs)
    inference.infer()


def train_experiment(_name: str,
                     _root_path: str):
    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)
    configs_path = os.path.join(_root_path, _name, 'configs.yaml')
    configs = read_configurations(configs_path)
    configure_logger(configs)

    if configs['logging']['log_summary']:
        summerize(configs)

    if configs['trainer']['cudnn_benchmark']:
        torch.backends.cudnn.benchmark = True
        logging.info("Enabling cudnn benchmarking")

    if configs['trainer']['model']['name'] == 'unet_3d':
        trainer = Unet3DTrainer(configs)
    elif configs['trainer']['model']['name'] == 'unet_3d_me':
        trainer = Unet3DMETrainer(configs)
    else:
        assert False, "Please provide a valid model name in the config file"

    trainer.train()


def delete_experiment(_name: str,
                      _root_path: str):
    if not experiment_exists(_root_path, _name):
        message = f"Experiment '{_name}' doesn't exist"
        raise FileNotFoundError(message)
    answer = input("Are you sure? (y/n) [default=n]: ")
    if answer.lower() == "y":
        logging.info("Removing the experiment: %s", _name)
        shutil.rmtree(os.path.join(_root_path, _name))
        logging.info('Experiment "%s" has been deleted', _name)


def create_new_experiment(_name: str,
                          _root_path: str,
                          _source_path: str,
                          _dataset_path: str,
                          _batch_size: int,
                          _semi_supervised: bool = False,
                          _configs: str = None):

    destination_path = os.path.join(_root_path, f'{_name}/')
    logging.info("Creating a new experiment in '%s%s/'", _root_path, _name)
    if os.path.exists(destination_path):
        message = f"Experiment already exists: {destination_path}"
        raise FileExistsError(message)

    logging.info("Copying project's source code")
    create_dirs_recursively(os.path.join(destination_path, 'dummy'))

    code_path = os.path.join(destination_path, 'code/')
    create_dirs_recursively(os.path.join(code_path, 'dummy'))
    copy_directory(_source_path,
                   code_path,
                   ['.git', 'tags'])

    logging.info("Saving the requirements file to '%s'",
                 destination_path)
    # Run the 'pip freeze' command and capture the output
    output = subprocess.check_output(["pip", "freeze"])
    output_str = output.decode().strip()

    # Write the output to a file named 'requirements.txt'
    with open(os.path.join(destination_path, 'requirements.txt'), "w",
              encoding='UTF-8') as f:
        f.write(output_str)

    logging.info("Saving the configuration file to '%s'",
                 destination_path)

    if _configs is None:
        logging.warning("Don't forget to edit the configurations")

        with open('./configs/template.yaml',
                  encoding='UTF-8') as template_file:
            configs = yaml.safe_load(template_file)

        batch_ratio = None
        if configs['experiments']['default_batch_size'] != _batch_size\
           and configs['experiments']['scale_lerning_rate_for_batch_size']:
            batch_ratio =\
                _batch_size / configs['experiments']['default_batch_size']

        configs['root_path'] = destination_path

        configs['trainer']['epochs'] = \
            configs['experiments']['default_epochs']

        configs['trainer']['loss_weights'] = \
            configs['experiments']['default_loss_weights']

        if batch_ratio is not None:
            configs['trainer']['optim']['lr'] =\
                configs['trainer']['optim']['lr'] / batch_ratio

        configs['trainer']['report_freq'] = \
            configs['experiments']['default_report_freq']

        configs['trainer']['train_ds']['batch_size'] = \
            configs['experiments']['default_batch_size']

        configs['trainer']['train_ds']['workers'] = \
            configs['experiments']['default_ds_workers']

        configs['trainer']['valid_ds']['batch_size'] = \
            configs['experiments']['default_batch_size']

        configs['trainer']['valid_ds']['workers'] = \
            configs['experiments']['default_ds_workers']

        configs['inference']['inference_ds']['path'] = \
            f"{configs['experiments']['default_data_path']}ds_test/"

        configs['inference']['inference_ds']['batch_size'] = \
            configs['experiments']['default_batch_size']

        configs['inference']['inference_ds']['workers'] = \
            configs['experiments']['default_ds_workers']

        del configs['experiments']

        with open(os.path.join(destination_path, 'configs.yaml'), 'w',
                  encoding='UTF-8') as config_file:
            yaml.dump(configs,
                      config_file,
                      default_flow_style=None,
                      sort_keys=False)
