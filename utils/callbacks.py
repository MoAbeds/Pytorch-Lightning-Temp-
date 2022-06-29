import numpy as np
import time
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class MyPrintingCallback(Callback):

    def __init__(self):
        super().__init__()
        self.epoch_start = None
        self.epoch_end = None
        self.start = None
        self.end = None
        self.targets = np.array(list())
        self.preds = np.array(list())
        self.num_samples = 0
        self.running_corrects = 0
        self.num_classes = 131
        self.n_correct_class = [0 for i in range(self.num_classes)]
        self.n_class_samples = [0 for i in range(self.num_classes)]
        self.absent_class = list()
        self.accdict = {}
        self.corrects = list()

    def on_fit_start(self, trainer, pl_module):
        print('===> Sanity check...')

    def on_train_start(self, trainer, pl_module):
        self.start = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        print('In training...')
        self.epoch_start = time.time()

    def on_validation_epoch_start(self, trainer, pl_module):
        avg_train_loss = torch.tensor(pl_module.train_losses).mean()
        avg_train_acc = torch.tensor(pl_module.train_accs).mean()
        pl_module.train_loss.append(avg_train_loss.item())
        pl_module.train_acc.append(avg_train_acc.item())
        print(f'Epoch {pl_module.current_epoch + 1} ' \
              f'Train Loss: {avg_train_loss:.3f}, Train Acc: {avg_train_acc:.2f}')

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_end = time.time()
        duration = self.epoch_start - self.epoch_end
        print(f'Time spent for Epoch {pl_module.current_epoch + 1} -----> {int(duration // 60)}m {int(duration % 60)}s')
        print("")

    def on_train_end(self, trainer, pl_module):
        self.end = time.time()
        duration = self.start - self.end
        print(f'Training completes in -----> {int(duration // 60)}m {int(duration % 60)}s')
        print(f'Best Validation Accuracy is {pl_module.best_acc:.3f}')

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        x, y = batch
        self.targets = np.concatenate((self.targets, y.cpu().numpy()), axis=None)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.preds = np.concatenate((self.preds, outputs.cpu().numpy()), axis=None)
        self.num_samples += outputs.shape[0]
        x, y = batch
        self.running_corrects += torch.sum(outputs == y)
        labels = y.cpu().numpy()
        outputs_ = outputs.cpu().numpy()
        for i in range(labels.shape[0]):
            label = labels[i]
            index_i = outputs_[i]

            if label == index_i:
                self.n_correct_class[label] += 1
            self.n_class_samples[label] += 1

    def on_test_epoch_end(self, trainer, pl_module):
        print(f'Got {self.running_corrects.item()}/{self.num_samples} correct samples.')
        for i in range(self.num_classes):
            if self.n_class_samples[i] != 0:
                acc_ = self.n_correct_class[i] / self.n_class_samples[i]
                self.accdict[i] = [acc_]
                temp = {'class': i, 'n_correct_class': self.n_correct_class[i],
                        'n_class_samples': self.n_class_samples[i]}
                self.corrects.append(temp)
            else:
                self.absent_class.append(i)


def Tesorboardlogger(name):
    logger = TensorBoardLogger("tb_logs", name=name)
    return logger

def Savecheckpoint():
    callback = ModelCheckpoint(filename="{epoch}-{val_acc}",
                               monitor='val_acc',
                               save_last=True,
                               mode='max')

    return  callback
