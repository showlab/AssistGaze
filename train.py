from modeling import EyeGazeQAclip
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as data
import torch
import torch.nn.functional as F
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import os
seed_everything(0, workers=True)
class EncodeDatatrain(Dataset):
    def __init__(self):
        super().__init__()
        self.path = []
        self.length = 0
        
        self.data = torch.load('final_data_questions2006train.pt',map_location=torch.device('cpu'))


    def __getitem__(self, index):
        instance = self.data[index]
        questionfeatures = instance['Question']
        textanswer = instance['Text_Answer']
        imageanswer = instance['Image_Answer']
        videoanswer = instance['Video_Answer']
        questionid = instance['question_number']
        label =instance['Label']

        videofeature1 = instance['Video_Feature1']
        videofeature2 = instance['Video_Feature2']
        videofeature3 = instance['Video_Feature3']
        videofeature4 = instance['Video_Feature4']
        videofeature5 = instance['Video_Feature5']
        samples = [questionid,questionfeatures,videofeature1,videofeature2,videofeature3,videofeature4,videofeature5,textanswer,imageanswer,videoanswer,label]
        return samples
    #print(batch)
    def __len__(self, ):
        return len(self.data)

class EncodeDatatest(Dataset):
    def __init__(self):
        super().__init__()
        self.path = []
        self.length = 0
        
        self.data = torch.load('final_data_questions2006test.pt',map_location=torch.device('cpu'))


    def __getitem__(self, index):
        instance = self.data[index]
        questionfeatures = instance['Question']
        textanswer = instance['Text_Answer']
        imageanswer = instance['Image_Answer']
        videoanswer = instance['Video_Answer']
        questionid = instance['question_number']
        label =instance['Label']

        videofeature1 = instance['Video_Feature1']
        videofeature2 = instance['Video_Feature2']
        videofeature3 = instance['Video_Feature3']
        videofeature4 = instance['Video_Feature4']
        videofeature5 = instance['Video_Feature5']
        samples = [questionid,questionfeatures,videofeature1,videofeature2,videofeature3,videofeature4,videofeature5,textanswer,imageanswer,videoanswer,label]
        return samples
    #print(batch)
    def __len__(self, ):
        return len(self.data)

class EyeGazeQADataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1024):
        super().__init__()
        self.batch_size = batch_size

        dataset=EncodeDatatrain()
        dataset2 = EncodeDatatest()
        self.train_set = dataset
        self.test_set = dataset2


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,shuffle=True,num_workers=4,pin_memory=True)


    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,shuffle=False,num_workers=4,pin_memory=True)


seed_everything(0, workers=True)

logger = TensorBoardLogger("0621tensorboardlog", name="my_model")
trainer = Trainer(
    devices = [1],
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_false",
    callbacks=[
            LearningRateMonitor(logging_interval='step'), 
            ModelCheckpoint(save_top_k = 1)
        ],
    benchmark=False, 
    deterministic=False,
    logger=logger,
    max_epochs=20,
    default_root_dir='training_results',
    check_val_every_n_epoch=1,
    log_every_n_steps=1
)
model = EyeGazeQAclip()
datamoduleA = EyeGazeQADataModule()
trainer.fit(model, datamodule=datamoduleA,ckpt_path=None)
trainer.test(model,datamoduleA,ckpt_path='last')