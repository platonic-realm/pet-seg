"""
Author: Arash Fatehi
Date:   22.11.2022
"""

# Python Imports
# Python's wierd implementation of abstract methods
from abc import ABC, abstractmethod
from pathlib import Path
import os
import glob
import random
import logging
from datetime import datetime
import threading

# Libary Imports
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.profiler import ProfilerActivity

# Local Imports
from src.utils.misc import create_dirs_recursively
from src.utils.visual import VisualizerUnet3D
from src.data.ds_train import FDGDataset
from src.utils.metrics.memory import GPURunningMetrics


# Tip for using abstract methods in python... dont use
# double __ for the abstract method as python name
# mangeling will mess them and you are going to have a hard time
class Trainer(ABC):
    def __init__(self,
                 _configs: dict,
                 _label_correction_function=None):
        self.configs: dict = _configs['trainer']

        # these variables are declared because some methods need
        # them, but they will be defined in the subclasses
        self.model = None
        self.optimizer = None
        self.loss = None

        self.label_correction = _label_correction_function

        self.root_path = _configs['root_path']

        # Note we are using self.configs from now on ...
        self.model_name = self.configs['model']['name']
        self.epochs: int = self.configs['epochs']
        self.epoch_resume = 0
        self.step = 0
        self.result_path = os.path.join(self.root_path,
                                        self.configs['result_path'])
        self.snapshot_path = os.path.join(self.root_path,
                                          self.result_path,
                                          self.configs['snapshot_path'])
        self.device: str = self.configs['device']
        self.mixed_precision: bool = self.configs['mixed_precision']
        if self.mixed_precision:
            # Needed for gradient scaling
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            self.scaler = torch.cuda.amp.GradScaler()

        # Data Parallelism
        self.dp = self.configs['dp']

        self.visualization: bool = \
            self.configs['visualization']['enabled']
        self.visualization_chance: float = \
            self.configs['visualization']['chance']
        self.visualization_path = \
            os.path.join(self.root_path,
                         self.result_path,
                         self.configs['visualization']['path'])

        self.visualizer = VisualizerUnet3D(
                _generate_tif=self.configs['visualization']['tif'],
                _generate_gif=self.configs['visualization']['gif'],
                _generate_mesh=self.configs['visualization']['mesh'])

        self.tensorboard: bool = \
            self.configs['tensorboard']['enabled']
        self.tensorboard_ls: bool =\
            self.configs['tensorboard']['label_seen']
        self.tensorboard_path = \
            Path(os.path.join(self.root_path,
                              self.result_path,
                              self.configs['tensorboard']['path']))
        self.tensorboard_path.mkdir(parents=True, exist_ok=True)

        if self.snapshot_path is not None:
            create_dirs_recursively(self.snapshot_path)

        self.gpu_metrics = GPURunningMetrics(self.configs,
                                             self.device)

        self.seen_labels = 0
        self.resume_count = 0

        self.number_class = 2

        self._prepare_profiling()
        self._prepare_data()

    def train(self):
        for epoch in range(self.epoch_resume, self.epochs):
            self._train_epoch(epoch)

    def _init_tensorboard(self):
        if self.step > 0:
            return

        if not self.tensorboard:
            return

        zero_metrics = GPURunningMetrics(self.configs,
                                         self.device)

        for index, data in enumerate(self.validation_loader):

            results = self._validate_step(_epoch_id=0,
                                          _batch_id=index,
                                          _data=data)

            self.gpu_metrics.add(results)

        zero_metrics = zero_metrics.calculate()
        self._log_tensorboard_metrics(0,
                                      'train',
                                      zero_metrics)
        self._log_tensorboard_metrics(0,
                                      'valid',
                                      zero_metrics)

    def _log_metrics(self,
                     _epoch: int,
                     _step: int,
                     _tag: str,
                     _metrics: dict) -> None:

        self._log_tensorboard_metrics(_step, _tag, _metrics)

    def _log_tensorboard_metrics(self,
                                 _n_iter: int,
                                 _mode: str,
                                 _metrics: dict) -> None:
        if not self.tensorboard:
            return

        tb_writer = SummaryWriter(self.tensorboard_path.resolve())

        for metric in _metrics.keys():
            tb_writer.add_scalar(f'{metric}/{_mode}',
                                 _metrics[metric],
                                 _n_iter)

        tb_writer.close()

    def _visualize_validation(self,
                              _epoch_id: int,
                              _batch_id: int,
                              _inputs,
                              _labels,
                              _predictions,
                              _all: bool = False):

        if not self.visualization:
            return

        random.seed(datetime.now().timestamp())
        dice = random.random()
        if dice > self.visualization_chance:
            return

        batch_size = len(_inputs)
        sample_id = random.randint(0, batch_size-1)

        base_path: str = self.visualization_path + \
            f"/epoch-{_epoch_id}/batch-{_batch_id}/"

        if _all:
            for index in range(batch_size):
                path: Path = Path(f"{base_path}{index}/")
                path.mkdir(parents=True, exist_ok=True)
                output_dir: str = path.resolve()
                self.visualizer.draw_channels(_inputs[index],
                                              output_dir,
                                              _multiplier=127)
                self.visualizer.draw_labels(_labels[index],
                                            output_dir,
                                            _multiplier=127)
                self.visualizer.draw_predictions(_predictions[index],
                                                 output_dir,
                                                 _multiplier=255)
        else:
            path: Path = Path(f"{base_path}")
            path.mkdir(parents=True, exist_ok=True)
            output_dir: str = path.resolve()

            self.visualizer.draw_channels(_inputs[sample_id],
                                          output_dir)
            self.visualizer.draw_labels(_labels[sample_id],
                                        output_dir,
                                        _multiplier=127)
            self.visualizer.draw_predictions(_predictions[sample_id],
                                             output_dir,
                                             _multiplier=127)

    def _save_sanpshot(self, _epoch: int) -> None:
        if self.snapshot_path is None:
            return

        snapshot = {}
        snapshot['EPOCHS'] = _epoch
        snapshot['STEP'] = self.step
        snapshot['SEEN_LABELS'] = self.seen_labels
        snapshot['RESUME_COUNT'] = self.resume_count
        if self.dp:
            snapshot['MODEL_STATE'] = self.model.module.state_dict()
        else:
            snapshot['MODEL_STATE'] = self.model.state_dict()

        thread = threading.Thread(target=self._save_snapshot_async,
                                  args=(snapshot, _epoch))
        thread.start()

    def _save_snapshot_async(self, _snapshot: dict, _epoch: int) -> None:

        save_path = \
            os.path.join(self.snapshot_path,
                         f"{_epoch:03d}-{self.step:04d}.pt")
        torch.save(_snapshot, save_path)
        logging.info("Snapshot saved on epoch: %d, step: %d",
                     _epoch+1,
                     self.step)

    def _load_snapshot(self) -> None:
        if not os.path.exists(self.snapshot_path):
            return

        snapshot_list = sorted(filter(os.path.isfile,
                                      glob.glob(self.snapshot_path + '*')),
                               reverse=True)

        if len(snapshot_list) <= 0:
            return

        load_path = snapshot_list[0]

        snapshot = torch.load(load_path,
                              map_location=torch.device(self.device))

        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epoch_resume = snapshot['EPOCHS'] + 1
        self.step = snapshot['STEP'] + 1
        self.seen_labels = snapshot['SEEN_LABELS']
        self.resume_count = snapshot['RESUME_COUNT'] + 1

        logging.info("Resuming training at epoch: %d", self.epoch_resume)
        logging.info("Resuming training at step: %d", self.step)

    def _prepare_data(self) -> None:

        train_dataset_path = self.configs['train_ds']['path']
        valid_dataset_path = self.configs['valid_ds']['path']

        self.train_batch_size = self.configs['train_ds']['batch_size']
        self.valid_batch_size = self.configs['valid_ds']['batch_size']

        positive_train_dataset = FDGDataset(_source_directory=train_dataset_path,
                                            _patch_dimension=self.configs['train_ds']['patch_dimension'],
                                            _stride=self.configs['train_ds']['stride'],
                                            _positive=True)

        negative_train_dataset = FDGDataset(_source_directory=train_dataset_path,
                                            _patch_dimension=self.configs['train_ds']['patch_dimension'],
                                            _stride=self.configs['train_ds']['stride'],
                                            _positive=False)

        positive_valid_dataset = FDGDataset(_source_directory=valid_dataset_path,
                                            _patch_dimension=self.configs['valid_ds']['patch_dimension'],
                                            _stride=self.configs['valid_ds']['stride'],
                                            _positive=True)

        negative_valid_dataset = FDGDataset(_source_directory=valid_dataset_path,
                                            _patch_dimension=self.configs['valid_ds']['patch_dimension'],
                                            _stride=self.configs['valid_ds']['stride'],
                                            _positive=False)

        positive_train_batch_size = self.configs['train_ds']['positive_per_patch']
        negative_train_batch_size = self.train_batch_size - self.configs['train_ds']['positive_per_patch']
        positive_valid_batch_size = self.configs['valid_ds']['positive_per_patch']
        negative_valid_batch_size = self.valid_batch_size - self.configs['valid_ds']['positive_per_patch']

        self.train_positive_loader = DataLoader(positive_train_dataset,
                                                batch_size=positive_train_batch_size,
                                                num_workers=self.configs['train_ds']['workers'],
                                                shuffle=self.configs['train_ds']['shuffle'],
                                                drop_last=True)

        self.train_negative_loader = DataLoader(negative_train_dataset,
                                                batch_size=negative_train_batch_size,
                                                num_workers=self.configs['train_ds']['workers'],
                                                shuffle=self.configs['train_ds']['shuffle'],
                                                drop_last=True)

        self.valid_positive_loader = DataLoader(positive_valid_dataset,
                                                batch_size=positive_valid_batch_size,
                                                num_workers=self.configs['valid_ds']['workers'],
                                                drop_last=True)

        self.valid_negative_loader = DataLoader(negative_valid_dataset,
                                                batch_size=negative_valid_batch_size,
                                                num_workers=self.configs['valid_ds']['workers'],
                                                drop_last=True)

    def _prepare_optimizer(self) -> None:
        optimizer_name: str = self.configs['optim']['name']
        if optimizer_name == 'adam':
            lr: float = self.configs['optim']['lr']
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=lr)

    def _prepare_loss(self) -> None:

        weights = torch.tensor(self.configs['loss_weights']).to(self.device)

        loss_name: str = self.configs['loss']
        if loss_name == 'CrossEntropy':
            self.loss = nn.CrossEntropyLoss(weight=weights)

    def _prepare_profiling(self) -> None:
        if self.configs['profiling']['enabled']:
            scheduler_wait = self.configs['profiling']['scheduler']['wait']
            scheduler_warmup = self.configs['profiling']['scheduler']['warmup']
            scheduler_active = self.configs['profiling']['scheduler']['active']
            scheduler_repeat = self.configs['profiling']['scheduler']['repeat']

            save_path = os.path.join(self.root_path,
                                     self.result_path,
                                     self.configs['profiling']['path'])

            profile_memory = self.configs['profiling']['profile_memory']
            record_shapes = self.configs['profiling']['record_shapes']
            with_flops = self.configs['profiling']['with_flops']
            with_stack = self.configs['profiling']['with_stack']

            tb_trace_handler = \
                torch.profiler.tensorboard_trace_handler(save_path)

            trace_file = os.path.join(save_path,
                                      "trace.txt")

            def txt_trace_handler(prof):
                with open(trace_file, 'w', encoding='UTF-8') as file:
                    file.write(prof.key_averages().table(
                                    sort_by="self_cuda_time_total",
                                    row_limit=-1))

            def print_trace_handler(prof):
                print(prof.key_averages().table(
                                sort_by="self_cuda_time_total",
                                row_limit=-1))

            def trace_handler(prof):
                if self.configs['profiling']['save']['tensorboard']:
                    tb_trace_handler(prof)

                if self.configs['profiling']['save']['text']:
                    txt_trace_handler(prof)

                if self.configs['profiling']['save']['print']:
                    print_trace_handler(prof)

            self.pytorch_profiling = True
            self.prof = torch.profiler.profile(
                    activities=[ProfilerActivity.CUDA,
                                ProfilerActivity.CPU],
                    schedule=torch.profiler.schedule(wait=scheduler_wait,
                                                     warmup=scheduler_warmup,
                                                     active=scheduler_active,
                                                     repeat=scheduler_repeat),
                    on_trace_ready=trace_handler,
                    profile_memory=profile_memory,
                    record_shapes=record_shapes,
                    with_flops=with_flops,
                    with_stack=with_stack)
        else:
            self.pytorch_profiling = False

    def _reports_metrics(self,
                         _metrics,
                         _loss):
        results = {}
        results['Loss'] = _loss

        metric_list = self.configs['metrics']

        for metric in metric_list[1:]:
            if metric == 'Accuracy':
                results[metric] = getattr(_metrics, metric)()
            else:
                results[metric] = getattr(_metrics, metric)(_class_id=1)

        return results

    @abstractmethod
    def _training_step(self,
                       _epoch_id: int,
                       _batch_id: int,
                       _data: dict) -> (dict, dict):
        pass

    @abstractmethod
    def _validate_step(self,
                       _epoch_id: int,
                       _batch_id: int,
                       _data: dict) -> (dict, dict):
        pass

    @abstractmethod
    def _train_epoch(self, _epoch: int) -> None:
        pass
