"""
Transformer part of ClipBERT
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
import lightning as pl
from torch.nn import Parameter
import numpy as np
from transformers import AutoModel
import torchmetrics

class EyeGazeQAclip(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.class_token = nn.Parameter(torch.zeros(1,1,768))
        el = nn.TransformerEncoderLayer(768,12, batch_first=True)
        self.model = nn.TransformerEncoder(encoder_layer=el, num_layers=3)
        self.classifier = nn.Linear(768, 1)
        self.correctchoice = 0
        self.numberofquestion = 0
        self.threshold = 0.1
        self.layer = nn.Linear(1024,768)
        self.cliplayer = nn.Linear(512,768)
        self.testid = []
        self.testlabel = []
        self.testpredict = []
        self.questionnum = []
        self.test_step_outputs=[]
        self.validation_step_outputs=[]
        #changed for 16 figure
        self.conv = nn.Conv1d(in_channels=512,out_channels=512,kernel_size=32,stride=32)
        
    def training_step(self,batch):
        loss = 0
        questionid,questionfeatures,videofeature1,videofeature2,videofeature3,videofeature4,videofeature5,textanswer,imageanswer,videoanswer,label= batch
        videofeature1 = self.cliplayer(self.conv(videofeature1.to(torch.float32).transpose(2,1)).transpose(1,2))
        videofeature2 = self.cliplayer(self.conv(videofeature2.to(torch.float32).transpose(2,1)).transpose(1,2))
        videofeature3 = self.cliplayer(self.conv(videofeature3.to(torch.float32).transpose(2,1)).transpose(1,2))
        videofeature4 = self.cliplayer(self.conv(videofeature4.to(torch.float32).transpose(2,1)).transpose(1,2))
        videofeature5 = self.cliplayer(self.conv(videofeature5.to(torch.float32).transpose(2,1)).transpose(1,2))
        image = self.layer(imageanswer)
        questionfeatures = self.cliplayer(questionfeatures.to(torch.float32))
        videoanswer=self.cliplayer(self.conv(videoanswer.to(torch.float32).transpose(2,1)).transpose(1,2))
        textanswer =self.cliplayer(textanswer.to(torch.float32))
            #assume one batch is [{question},{n videos choices features},{4 text answers},{correctlabel}]
        features = torch.cat([questionfeatures,videofeature1,videofeature2,videofeature3,videofeature4,videofeature5,image,textanswer,videoanswer],dim=1)
        class_token = self.class_token.expand(features.shape[0],-1,-1)
        _features = torch.cat([class_token,features],dim=1)
        multimodaloutput = self.model(_features)
        multimodalhead = multimodaloutput[:,0]#with shape of 1*768
        logit = self.classifier(multimodalhead)#dot product of multimodal features head with video features head to calculate similarity
        device = torch.cuda.current_device()
        label= label.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(logit.float().to('cuda'),label.float().to('cuda'))
        self.log('train_loss',loss)
        return loss
    
    def validation_step(self,batch,batch_idx):
        questionid,questionfeatures,videofeature1,videofeature2,videofeature3,videofeature4,videofeature5,textanswer,imageanswer,videoanswer,label= batch
        videofeature1 = self.cliplayer(self.conv(videofeature1.to(torch.float32).transpose(2,1)).transpose(1,2))
        videofeature2 = self.cliplayer(self.conv(videofeature2.to(torch.float32).transpose(2,1)).transpose(1,2))
        videofeature3 = self.cliplayer(self.conv(videofeature3.to(torch.float32).transpose(2,1)).transpose(1,2))
        videofeature4 = self.cliplayer(self.conv(videofeature4.to(torch.float32).transpose(2,1)).transpose(1,2))
        videofeature5 = self.cliplayer(self.conv(videofeature5.to(torch.float32).transpose(2,1)).transpose(1,2))
        image = self.layer(imageanswer)
        questionfeatures = self.cliplayer(questionfeatures.to(torch.float32))
        videoanswer=self.cliplayer(self.conv(videoanswer.to(torch.float32).transpose(2,1)).transpose(1,2))
        textanswer =self.cliplayer(textanswer.to(torch.float32))
            #assume one batch is [{question},{n videos choices features},{4 text answers},{correctlabel}]
        features = torch.cat([questionfeatures,videofeature1,videofeature2,videofeature3,videofeature4,videofeature5,textanswer,videoanswer],dim=1)
        class_token = self.class_token.expand(features.shape[0],-1,-1)
        _features = torch.cat([class_token,features],dim=1)
        features_img = torch.cat([features,image],dim=1)
        #assume one batch is [{question},{n videos choices features},{4 text answers},{correctlabel}]
        _features_img = torch.cat([class_token,features_img],dim=1)
        multimodaloutput = self.model(_features)
        multimodaloutput_img = self.model(_features_img)
        multimodalhead = multimodaloutput[:,0]#with shape of 1*768
        multimodalhead_img = multimodaloutput_img[:,0]
        logit = self.classifier(multimodalhead)
        logit_img = self.classifier(multimodalhead_img)
        predicted = torch.sigmoid(logit)
        predicted_img = torch.sigmoid(logit_img)
        num = image.get_device()
        random = torch.rand(_features.shape[0]).view(features.shape[0],1)
        random = random.to(torch.device(f'cuda:{num}'))
        self.validation_step_outputs.append([predicted,predicted_img, random, label])
        valloss = F.binary_cross_entropy_with_logits(logit.float().to('cuda'),label.float().unsqueeze(1).to('cuda'))
        self.log('val_loss',valloss)
        return predicted,predicted_img, random, label,valloss

    def on_validation_epoch_end(self) -> None:
        predicted=[sublist[0] for sublist in self.validation_step_outputs]
        predicted_img=[sublist[1] for sublist in self.validation_step_outputs]
        random=[sublist[2] for sublist in self.validation_step_outputs]
        label=[sublist[-1] for sublist in self.validation_step_outputs]
        predicted = torch.cat(predicted).to('cuda')
        predicted_img = torch.cat(predicted_img).to('cuda')
        label = torch.cat(label).to('cuda').unsqueeze(1)
        random = torch.cat(random).to('cuda')
        metric = torchmetrics.AveragePrecision(task="binary")
        ap = metric(predicted.to('cuda'), label.long().to('cuda'))
        ap_img = metric(predicted_img, label.long())
        self.log('ap_val', ap,sync_dist=True)
        self.log('ap_withimage_val', ap_img,sync_dist=True)
        self.validation_step_outputs = []
        return ap,ap_img
        
    def test_step(self,batch,batch_idx):
        questionid,questionfeatures,videofeature1,videofeature2,videofeature3,videofeature4,videofeature5,textanswer,imageanswer,videoanswer,label= batch
        videofeature1 = self.cliplayer(self.conv(videofeature1.to(torch.float32).transpose(2,1)).transpose(1,2))
        videofeature2 = self.cliplayer(self.conv(videofeature2.to(torch.float32).transpose(2,1)).transpose(1,2))
        videofeature3 = self.cliplayer(self.conv(videofeature3.to(torch.float32).transpose(2,1)).transpose(1,2))
        videofeature4 = self.cliplayer(self.conv(videofeature4.to(torch.float32).transpose(2,1)).transpose(1,2))
        videofeature5 = self.cliplayer(self.conv(videofeature5.to(torch.float32).transpose(2,1)).transpose(1,2))
        image = self.layer(imageanswer)
        questionfeatures = self.cliplayer(questionfeatures.to(torch.float32))
        videoanswer=self.cliplayer(self.conv(videoanswer.to(torch.float32).transpose(2,1)).transpose(1,2))
        textanswer =self.cliplayer(textanswer.to(torch.float32))
            #assume one batch is [{question},{n videos choices features},{4 text answers},{correctlabel}]
        features = torch.cat([questionfeatures,videofeature1,videofeature2,videofeature3,videofeature4,videofeature5,textanswer,videoanswer],dim=1)
        class_token = self.class_token.expand(features.shape[0],-1,-1)
        _features = torch.cat([class_token,features],dim=1)
        features_img = torch.cat([features,image],dim=1)
        #assume one batch is [{question},{n videos choices features},{4 text answers},{correctlabel}]
        _features_img = torch.cat([class_token,features_img],dim=1)
        multimodaloutput = self.model(_features)
        multimodaloutput_img = self.model(_features_img)
        multimodalhead = multimodaloutput[:,0]#with shape of 1*768
        multimodalhead_img = multimodaloutput_img[:,0]

        logit = self.classifier(multimodalhead)
        logit_img = self.classifier(multimodalhead_img)
        predicted = torch.sigmoid(logit)
        predicted_img = torch.sigmoid(logit_img)
        for item in predicted_img:
            self.testpredict.append(item.cpu().numpy())
        num = image.get_device()
        random = torch.rand(_features.shape[0]).view(features.shape[0],1)
        random = random.to(torch.device(f'cuda:{num}'))
        self.test_step_outputs.append([predicted,predicted_img, random, label])
        return predicted,predicted_img, random, label

    def on_test_epoch_end(self) -> None:
        #self.tesid = self.all_gather(self.testid)
        self.testlabel = self.all_gather(self.testlabel)
        self.questionnum = self.all_gather(self.questionnum)
        self.testpredict = self.all_gather(self.testpredict)
        predicted_img=[sublist[1] for sublist in self.test_step_outputs]
        random=[sublist[2] for sublist in self.test_step_outputs]
        label=[sublist[-1] for sublist in self.test_step_outputs]
        predicted_img = torch.cat(predicted_img).to('cuda')
        label = torch.cat(label).to('cuda').unsqueeze(1)
        random = torch.cat(random).to('cuda')
        metric = torchmetrics.AveragePrecision(task="binary")
        ap_img = metric(predicted_img, label.long())
        ap_random = metric(random,label.long())
        metric = torchmetrics.Accuracy(task="binary").to('cuda')

        self.log('ap', ap_img)
        self.log('ap_randomguess', ap_random)
        return ap_img,ap_random

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=6e-7)
        return optimizer



