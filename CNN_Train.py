import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms

from PIL import Image
import os
import numpy as np
import pickle


# Check availability of GPU
use_gpu = torch.cuda.is_available()

# Load train-test list
with open('trainList_5class.pckl','rb') as f:
    trainList = pickle.load(f)
with open('testList_5class.pckl','rb') as f:
    testList = pickle.load(f)
    
classes = []
for item in trainList:
    c = item.split('_')[1]
    if c not in classes:
        classes.append(c)

net = models.resnet18()
net.fc = nn.Linear(512,101)
# Loading saved states
net.load_state_dict(torch.load('resnet18Pre_fcOnly5class_ucf101_10adam_1e-4_b128.pt'))

# Removing fully connected layer for feature extraction
model = nn.Sequential(*list(net.children())[:-1])
if use_gpu:
    model = model.cuda()

data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),            
        transforms.ToTensor()
    ])

framePath = 'frames/'
for item in trainList:
    cName = item.split('_')[1]
    srcPath = framePath+cName+'/'+item    
    fNames = os.listdir(srcPath)
    # filename template
    fTemplate = fNames[0].split('_')
    fCount = len(fNames)
    for fNum in range(fCount):
        fileName = fTemplate[0]+'_'+fTemplate[1]+'_'+fTemplate[2]+'_'+fTemplate[3]+'_'+str(fNum+1)+'.jpg'
        if os.path.exists(srcPath+'/'+fileName):
            # Loading image
            img = Image.open(srcPath+'/'+fileName)
            # Transform to tensor
            imgTensor = data_transforms(img).unsqueeze(0)
            if use_gpu:
                inp = Variable(imgTensor.cuda())
            else:
                inp = Variable(imgTensor)
            # Feed-forward through model+stack features for each video
            if fNum == 0:
                out = model(inp)                
                out = out.view(out.size()[0],-1).data.cpu()                
            else:
                out1 = model(inp)               
                out1 = out1.view(out1.size()[0],-1).data.cpu()                
                out = torch.cat((out,out1),0)
        else:
            print(fileName+ ' missing!')       
    # out dimension -> frame count x 512
    featSavePath = 'ucf101_resnet18Feat/train/'+cName # Directory for saving features
    if not os.path.exists(featSavePath):
        os.makedirs(featSavePath)
    torch.save(out,os.path.join(featSavePath,item+'.pt'))

framePath = 'frames/'
for item in testList:
    cName = item.split('_')[1]
    srcPath = framePath+cName+'/'+item    
    fNames = os.listdir(srcPath)
    fTemplate = fNames[0].split('_')
    fCount = len(fNames)
    for fNum in range(fCount):
        fileName = fTemplate[0]+'_'+fTemplate[1]+'_'+fTemplate[2]+'_'+fTemplate[3]+'_'+str(fNum+1)+'.jpg'
        if os.path.exists(srcPath+'/'+fileName):
            img = Image.open(srcPath+'/'+fileName)
            imgTensor = data_transforms(img).unsqueeze(0)
            inp = Variable(imgTensor.cuda())
            if fNum == 0:
                out = model(inp)                
                out = out.view(out.size()[0],-1).data.cpu()
                
            else:
                out1 = model(inp)               
                out1 = out1.view(out1.size()[0],-1).data.cpu()                
                out = torch.cat((out,out1),0)
        else:
            print(fileName+ ' missing!')
      
    featSavePath = 'ucf101_resnet18Feat/test/'+cName
    if not os.path.exists(featSavePath):
        os.makedirs(featSavePath)
    torch.save(out,os.path.join(featSavePath,item+'.pt'))