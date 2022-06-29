from pytorch_lightning import LightningDataModule ,LightningModule
import os
import torch
from  torchvision.datasets import ImageFolder
import torchvision.io as io
import torchvision.transforms as T
import torchvision
from functools import partial
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics.functional import accuracy
import torch.nn as nn
from math import sqrt
import torch.optim as optim
from enum import Enum
import numpy as np
import time
import torch.nn.functional as F
import torchvision.models.resnet as RES
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer ,LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy


class LightModel(LightningModule):
    def __init__(self,val_length):
        super().__init__()

        self.model = torchvision.models.googlenet(pretrained=True)
        num_classes = 131
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()
        self.val_length = val_length



        self.train_losses = list()
        self.train_accs = list()
        self.train_loss = list()
        self.train_acc = list()
        self.val_loss = list()
        self.val_acc = list()
        self.best_acc = 0.0

    def forward(self,x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.train_losses.append(loss.item())
        self.train_accs.append(acc.item())
        metrics = {'loss': loss, 'train_acc': acc}
        pbar = {'train_loss': loss, 'train_acc': acc}
        self.log_dict(pbar, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return metrics

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return metrics

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x['val_loss'].item() for x in outputs]).mean()
        avg_val_acc = torch.tensor([x['val_acc'].item() for x in outputs]).mean()
        print(f'Epoch {self.current_epoch + 1} ' \
              f'Val Loss: {avg_val_loss:.3f}, Val Acc: {avg_val_acc:.2f}')
        self.val_loss.append(avg_val_loss.item())
        self.val_acc.append(avg_val_acc.item())
        if avg_val_acc.item() > self.best_acc:
          self.best_acc = avg_val_acc.item()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        _, index = torch.max(logits, 1)
        loss = self.criterion(logits, y)
        acc = accuracy(logits, y)
        metrics = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        metrics['index'] = index
        return metrics

    def test_step_end(self, outputs):
        return outputs['index']

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self.model(x)
        _, index = torch.max(logits, 1)
        return index

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        acc = accuracy(logits, y)
        return loss, acc

    def configure_optimizers(self):
        optim= torch.optim.Adam(self.parameters(), lr=0.02)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.8)
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }