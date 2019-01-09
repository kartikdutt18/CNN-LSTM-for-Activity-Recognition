import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

import os
import numpy as np
from random import shuffle
import copy
import time

use_gpu = torch.cuda.is_available()

trainPath = 'ucf101_resnet18Feat/train/'
testPath = 'ucf101_resnet18Feat/test/'

classes = os.listdir(trainPath)
classes.sort()
labels = np.arange(101)
trainShuffList = []
labelShuffList = []
for c in range(101):
    files = os.listdir(trainPath+classes[c])
    for f in files:
        trainShuffList.append(classes[c]+'/'+f)  
        labelShuffList.append(float(labels[c]))
# Shuffling data list and label list
trainList = list(zip(trainShuffList, labelShuffList))
shuffle(trainList)
trainShuffList, labelShuffList = zip(*trainList)

testList = []
testLabelList = []

for c in range(101):
    files = os.listdir(testPath+classes[c])
    for f in files:
        testList.append(classes[c]+'/'+f)  
        testLabelList.append(float(labels[c]))
        

class net_LSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, nLayers, nClasses):
        super(net_LSTM, self).__init__()       
        self.lstm = nn.LSTM(input_sz, hidden_sz, nLayers, batch_first=True)
        self.fc = nn.Linear(hidden_sz, nClasses)        
    
    def forward(self, x):      
        out, _ = self.lstm(x)       
        # Output from hidden state of last time step
        out = self.fc(out[:, -1, :])  
        return out

def train(net, inputs, labels, optimizer, criterion):
    net.train(True)
    if use_gpu:
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    # Feed-forward
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)     
    # Initialize gradients to zero
    optimizer.zero_grad() 
    # Compute loss/error
    loss = criterion(F.log_softmax(outputs), labels)
    # Backpropagate loss and compute gradients
    loss.backward()
    # Update the network parameters
    optimizer.step()
    if use_gpu:
        correct = (predicted.cpu() == labels.data.cpu()).sum()
    else:
        correct = (predicted == labels.data).sum()
    return net, loss.data[0], correct

def test(net, inputs, labels, criterion):
    net.train(False)
    if use_gpu:
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)  
    # Compute loss/error
    loss = criterion(F.log_softmax(outputs), labels)   
    if use_gpu:
        correct = (predicted.cpu() == labels.data.cpu()).sum()
    else:
        correct = (predicted == labels.data).sum()
    return loss.data[0], correct


net = net_LSTM(512, 8, 2, nClasses=101) # Input feature length->512, hidden layer size->8, number of layers->2
if use_gpu:
    net = net.cuda()

criterion = nn.NLLLoss() # Negative Log-likelihood
optimizer = optim.Adam(net.parameters(), lr=1e-4) # Adam

epochs = 500
bSize = 32 # Batch size
L = 32 # Number of time steps

bCount = len(trainShuffList)//bSize # Number of batches in train set
lastBatch = len(trainShuffList)%bSize # Number of samples in last batch of train set

test_bCount = len(testList)//bSize # Number of batches in test set
test_lastBatch = len(testList)%bSize # Number of samples in last batch of test set

# Lists for saving train/test loss and accuracy
trainLoss = []
trainAcc = []
testLoss = []
testAcc = []

start = time.time()

for epochNum in range(epochs):
    # Shuffling train data for each epoch
    trainList = list(zip(trainShuffList, labelShuffList))
    shuffle(trainList)
    trainShuffList, labelShuffList = zip(*trainList)
    
    trainRunLoss = 0.0
    testRunLoss = 0.0
    trainRunCorr = 0
    testRunCorr = 0
    
    epochStart = time.time()
    
    ## Train
    # Load data tensors batchwise     
    idx = 0    
    for bNum in range(bCount):
        first = True
        # Loading one batch
        for dNum in range(idx,idx+bSize):
            if first:
                loadData = torch.load(trainPath+trainShuffList[dNum])
                sz = loadData.size(0)
                idx1 = torch.from_numpy(np.arange(0,(sz//L)*L,sz//L))
                batchData = torch.index_select(loadData,dim=0,index=idx1).unsqueeze(0)
                batchLabel = torch.Tensor([labelShuffList[dNum]]).long()                          
                first = False                
            else:
                loadData = torch.load(trainPath+trainShuffList[dNum])
                sz = loadData.size(0)
                idx1 = torch.from_numpy(np.arange(0,(sz//L)*L,sz//L))
                tempData = torch.index_select(loadData,dim=0,index=idx1).unsqueeze(0)
                batchData = torch.cat((batchData,tempData), dim=0)
                batchLabel = torch.cat((batchLabel,torch.Tensor([labelShuffList[dNum]]).long()),dim=0)            
        
        # Train the network on current batch
        net, tr_loss, tr_corr = train(net, batchData, batchLabel, optimizer, criterion)
        trainRunLoss += tr_loss
        trainRunCorr += tr_corr
        idx += bSize
        
    # Loading last batch
    if lastBatch != 0:        
        first = True
        for dNum in range(idx,idx+lastBatch):
            if first:
                loadData = torch.load(trainPath+trainShuffList[dNum])
                sz = loadData.size(0)
                idx1 = torch.from_numpy(np.arange(0,(sz//L)*L,sz//L))
                batchData = torch.index_select(loadData,dim=0,index=idx1).unsqueeze(0)
                batchLabel = torch.Tensor([labelShuffList[dNum]]).long()
                first = False                
            else:
                loadData = torch.load(trainPath+trainShuffList[dNum])
                sz = loadData.size(0)
                idx1 = torch.from_numpy(np.arange(0,(sz//L)*L,sz//L))
                tempData = torch.index_select(loadData,dim=0,index=idx1).unsqueeze(0)
                batchData = torch.cat((batchData,tempData), dim=0)
                batchLabel = torch.cat((batchLabel,torch.Tensor([labelShuffList[dNum]]).long()),dim=0)          
        
        # Training network on last batch
        net, tr_loss, tr_corr = train(net, batchData, batchLabel, optimizer, criterion)
        trainRunLoss += tr_loss
        trainRunCorr += tr_corr
    
    # Average training loss and accuracy for each epoch
    avgTrainLoss = trainRunLoss/float(len(trainShuffList))
    trainLoss.append(avgTrainLoss)
    avgTrainAcc = trainRunCorr/float(len(trainShuffList))
    trainAcc.append(avgTrainAcc)
    
    ## Test
    # Load data tensors batchwise     
    idx = 0    
    for bNum in range(test_bCount):
        first = True
        # Loading one batch
        for dNum in range(idx,idx+bSize): 
            if first:
                loadData = torch.load(testPath+testList[dNum])
                sz = loadData.size(0)
                idx1 = torch.from_numpy(np.arange(0,(sz//L)*L,sz//L))
                batchData = torch.index_select(loadData,dim=0,index=idx1).unsqueeze(0)
                batchLabel = torch.Tensor([testLabelList[dNum]]).long()
                first = False                
            else:
                loadData = torch.load(testPath+testList[dNum])
                sz = loadData.size(0)
                idx1 = torch.from_numpy(np.arange(0,(sz//L)*L,sz//L))
                tempData = torch.index_select(loadData,dim=0,index=idx1).unsqueeze(0)
                batchData = torch.cat((batchData,tempData), dim=0)
                batchLabel = torch.cat((batchLabel,torch.Tensor([testLabelList[dNum]]).long()),dim=0)            
        
        # Test the network on current batch
        ts_loss, ts_corr = test(net, batchData, batchLabel, criterion)
        testRunLoss += ts_loss
        testRunCorr += ts_corr
        idx += bSize
     
    # Loading last batch    
    if test_lastBatch != 0:        
        first = True
        for dNum in range(idx,idx+test_lastBatch):
            if first:
                loadData = torch.load(testPath+testList[dNum])
                sz = loadData.size(0)
                idx1 = torch.from_numpy(np.arange(0,(sz//L)*L,sz//L))
                batchData = torch.index_select(loadData,dim=0,index=idx1).unsqueeze(0)               
                batchLabel = torch.Tensor([testLabelList[dNum]]).long()
                first = False                
            else:
                loadData = torch.load(testPath+testList[dNum])
                sz = loadData.size(0)
                idx1 = torch.from_numpy(np.arange(0,(sz//L)*L,sz//L))
                tempData = torch.index_select(loadData,dim=0,index=idx1).unsqueeze(0)
                batchData = torch.cat((batchData,tempData), dim=0)
                batchLabel = torch.cat((batchLabel,torch.Tensor([testLabelList[dNum]]).long()),dim=0)          
        
        # Test network on last batch
        ts_loss, ts_corr = test(net, batchData, batchLabel, criterion)
        testRunLoss += ts_loss
        testRunCorr += tr_corr
        
    # Average testing loss and accuracy for each epoch
    avgTestLoss = testRunLoss/float(len(testList))
    testLoss.append(avgTestLoss)
    avgTestAcc = testRunCorr/float(len(testList))
    testAcc.append(avgTestAcc)   

    
    # Plotting training loss vs Epochs
    fig1 = plt.figure(1)        
    plt.plot(range(epochNum+1),trainLoss,'r-',label='train')  
    plt.plot(range(epochNum+1),testLoss,'g-',label='test') 
    if epochNum==0:
        plt.legend(loc='upper left')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')   
    # Plotting testing accuracy vs Epochs
    fig2 = plt.figure(2)        
    plt.plot(range(epochNum+1),trainAcc,'r-',label='train')    
    plt.plot(range(epochNum+1),testAcc,'g-',label='test')        
    if epochNum==0:
        plt.legend(loc='upper left')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
    
    epochEnd = time.time()-epochStart
    print('Iteration: {:.0f} /{:.0f};  Training Loss: {:.6f} ; Training Acc: {:.3f}'\
          .format(epochNum + 1,epochs, avgTrainLoss, avgTrainAcc*100))
    print('Iteration: {:.0f} /{:.0f};  Testing Loss: {:.6f} ; Testing Acc: {:.3f}'\
          .format(epochNum + 1,epochs, avgTestLoss, avgTestAcc*100))
    
    print('Time consumed: {:.0f}m {:.0f}s'.format(epochEnd//60,epochEnd%60))
end = time.time()-start
print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))