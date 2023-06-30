"""
Author: Arash Fatehi
Date:   23.11.2022
"""

# Python Imports
import logging

# Library Imports
import torch
from torch.nn.parallel import DataParallel as DP

# Local Imports
from src.train.trainer import Trainer
from src.models.unet3d import Unet3D
from src.utils.metrics.classification import Metrics


class Unet3DTrainer(Trainer):
    def __init__(self,
                 _configs: dict,
                 _label_correction_function=None):

        super().__init__(_configs,
                         _label_correction_function)
        assert self.configs['model']['name'] == 'unet_3d', \
               "This class should only be used with unet_3d configuration." + \
               f"{self.configs['model']['name']} was given instead."

        self.feature_maps: list = self.configs['model']['feature_maps']
        self.channels: int = 2
        self.metrics: list = self.configs['metrics']

        self.model = Unet3D(self.channels,
                            self.number_class,
                            _feature_maps=self.feature_maps)

        self._load_snapshot()

        self.model.to(self.device)

        if self.dp:
            self.model = DP(self.model)

        self._prepare_optimizer()
        self._prepare_loss()

    def _training_step(self,
                       _epoch_id: int,
                       _batch_id: int,
                       _data: dict) -> dict:

        self.step += 1

        device = self.device

        ct = torch.unsqueeze(_data[:, 0, :, :], dim=1).to(device)
        pet = torch.unsqueeze(_data[:, 1, :, :], dim=1).to(device)
        labels = _data[:, 2, :, :].to(device)
        labels = labels.long()

        sample = torch.cat((ct, pet), dim=1)

        self.optimizer.zero_grad()

        if self.mixed_precision:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, results = self.model(sample)
                loss = self.loss(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, results = self.model(sample)
            loss = self.loss(logits, labels)
            loss.backward()
            self.optimizer.step()

        metrics = Metrics(self.number_class,
                          results,
                          labels)

        return self._reports_metrics(metrics, loss)

    def _validate_step(self,
                       _epoch_id: int,
                       _batch_id: int,
                       _data: dict) -> dict:

        device = self.device

        ct = torch.unsqueeze(_data[:, 0, :, :], dim=1).to(device)
        pet = torch.unsqueeze(_data[:, 1, :, :], dim=1).to(device)
        labels = _data[:, 2, :, :].to(device)
        labels = labels.long()

        sample = torch.cat((ct, pet), dim=1)

        with torch.no_grad():

            logits, results = self.model(sample)

            self._visualize_validation(_epoch_id=_epoch_id,
                                       _batch_id=_batch_id,
                                       _inputs=sample,
                                       _labels=labels,
                                       _predictions=results)

            metrics = Metrics(self.number_class,
                              results,
                              labels)

            loss = self.loss(logits, labels)

            return self._reports_metrics(metrics, loss)

    def _train_epoch(self, _epoch: int):

        freq = self.configs['report_freq']

        if self.pytorch_profiling:
            self.prof.start()

        train_negative_iterator = iter(self.train_negative_loader)

        for index, positive_samples in enumerate(self.train_positive_loader):

            negative_samples = next(train_negative_iterator)
            batch = torch.cat((positive_samples, negative_samples), dim=0)
            results = self._training_step(_epoch,
                                          index,
                                          batch)

            self.gpu_metrics.add(results)

            if self.pytorch_profiling:
                self.prof.step()

            if self.step % freq == 0:
                # We should calculate once and report twice
                metrics = self.gpu_metrics.calculate()
                self._log_metrics(_epoch,
                                  self.step,
                                  'train',
                                  metrics)

                logging.info("Epoch: %d/%d, Batch: %d/%d, Step: %d\n"
                             "Info: %s",
                             _epoch+1,
                             self.epochs,
                             index+1,
                             len(self.valid_positive_loader),
                             self.step,
                             metrics)

                # Save the snapshot
                self._save_sanpshot(_epoch)

                valid_negative_iterator = iter(self.valid_negative_loader)
                for valid_index, valid_positive_samples \
                        in enumerate(self.valid_positive_loader):

                    valid_negative_samples = next(valid_negative_iterator)
                    valid_batch = torch.cat((valid_positive_samples,
                                             valid_negative_samples),
                                            dim=0)

                    results = self._validate_step(_epoch_id=_epoch,
                                                  _batch_id=valid_index,
                                                  _data=valid_batch)

                    self.gpu_metrics.add(results)

                # We should calculate once and report twice
                metrics = self.gpu_metrics.calculate()
                logging.info("Validation, Step: %d\n"
                             "Info: %s",
                             self.step,
                             metrics)

                self._log_metrics(_epoch,
                                  self.step,
                                  'valid',
                                  metrics)
                self._log_tensorboard_metrics(self.step,
                                              'valid',
                                              metrics)

        if self.pytorch_profiling:
            self.prof.stop()
