from data import FolderData
from model import LightModel
from utils import callbacks
from utils import Visualize

from pytorch_lightning import Trainer ,LightningModule

data_dir = '/content/data/fruits-360_dataset/fruits-360'
cb = callbacks.MyPrintingCallback()
logger = callbacks.TensorBoardLogger('fruit_model')
save_check = callbacks.ModelCheckpoint()
Epochs = 5
Precision = 16
dm = FolderData.DataModule(data_dir)
dm.setup()
model = LightModel(val_length=dm.val_length)


def main():


    trainer = Trainer(gpus=1,
                      benchmark=True,
                      max_epochs=Epochs,
                      precision=Precision,
                      callbacks=[cb, save_check],
                      check_val_every_n_epoch=1,
                      gradient_clip_val=8,
                      logger=logger)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, dataloaders=dm.test_dataloader())

    #Visualize.visualize_cost(model.train_loss, model.train_acc, model.val_loss, model.val_acc)





if __name__ == "__main__":
    main()
