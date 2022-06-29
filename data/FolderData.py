from pytorch_lightning import LightningDataModule
import os
import torch
from  torchvision.datasets import ImageFolder
import torchvision.io as io
import torchvision.transforms as T
from functools import partial
from torch.utils.data import DataLoader

class DataModule(LightningDataModule):
    def __init__(self,data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.image_read_func = partial(io.read_image,mode=io.image.ImageReadMode.RGB)
        self.batch_size = 64
        self.train_transform = T.Compose([
                T.Resize(size = (256,256)),
                T.RandomRotation(degrees = (-20,+20)),
                T.CenterCrop(size=224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

        self.test_trasform = T.Compose([
                T.Resize(size = (224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])


    def setup(self, stage= None) -> None:
        self.train_data = ImageFolder(os.path.join(self.data_dir,'Training'),transform=self.train_transform)
        print(len(self.train_data))
        self.train_set, self.val_set = torch.utils.data.random_split(self.train_data, [50000,17692])
        self.test_data = ImageFolder(os.path.join(self.data_dir,'Test'),transform=self.train_transform,)
        self.val_length = len(self.val_set)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set,batch_size=self.batch_size,shuffle=True,)

    def val_dataloader(self) :
        return DataLoader(dataset=self.val_set,batch_size=self.batch_size,)

    def test_dataloader(self) :
        return DataLoader(dataset=self.test_data,batch_size=self.batch_size,)