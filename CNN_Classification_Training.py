import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,datasets, models
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

use_gpu = torch.cuda.is_available()
if use_gpu:
    pinMem = True
else:
    pinMem = False
    
trainDir = 'train_5class'
valDir = 'test_5class'
apply_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

# Training dataloader
train_dataset = datasets.ImageFolder(trainDir,transform=apply_transform)
trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=4, pin_memory=pinMem)

# Test dataloader
test_dataset = datasets.ImageFolder(valDir,transform=apply_transform)
testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False,num_workers=4, pin_memory=pinMem)

# Size of train and test datasets
print('No. of samples in train set: '+str(len(trainLoader.dataset)))
print('No. of samples in test set: '+str(len(testLoader.dataset)))

net = models.resnet18(pretrained=True)
print(net)

#params

totalParams = 0
for params in net.parameters():
    print(params.size())
    totalParams += np.sum(np.prod(params.size()))
print('Total number of parameters: '+str(totalParams))

net.fc = nn.Linear(512,101)

iterations = 10

trainLoss = []
trainAcc = []
testLoss = []
testAcc = []

start = time.time()
for epoch in range(iterations):
    epochStart = time.time()
    runningLoss = 0.0   
    avgTotalLoss = 0.0
    running_correct = 0   
    
    net.train(True) # For training 
    batchNum = 1
    for data in trainLoader:
        inputs,labels = data
        # Wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)            
            running_correct += (predicted.cpu() == labels.data.cpu()).sum()           
        else:
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels.data).sum()            
       
        # Initialize gradients to zero
        optimizer.zero_grad()             
        
        # Compute loss/error
        loss = criterion(F.log_softmax(outputs), labels)
        # Backpropagate loss and compute gradients
        loss.backward()
        # Update the network parameters
        optimizer.step()
        # Accumulate loss per batch
        runningLoss += loss.item()  
        batchNum += 1

    avgTrainAcc = running_correct/float(len(trainLoader.dataset))
    avgTrainLoss = runningLoss/float(len(trainLoader.dataset))    
    trainAcc.append(avgTrainAcc)
    trainLoss.append(avgTrainLoss)  
    
    # Evaluating performance on test set for each epoch
    net.train(False) # For testing [Affects batch-norm and dropout layers (if any)]
    running_correct = 0 
    for data in testLoader:
        inputs,labels = data
        # Wrap them in Variable
        if use_gpu:
            inputs, labels= Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)            
            running_correct += (predicted.cpu() == labels.data.cpu()).sum()
        else:
            inputs, labels = Variable(inputs), Variable(labels)
            # Model 1
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels.data).sum()
        
        loss = criterion(F.log_softmax(outputs), labels)
        
        runningLoss += loss.item()  

    avgTestLoss = runningLoss/float(len(testLoader.dataset))
    avgTestAcc = running_correct/float(len(testLoader.dataset))
    testAcc.append(avgTestAcc)  
    testLoss.append(avgTestLoss)
    
    # Plotting training loss vs Epochs
    fig1 = plt.figure(1)        
    plt.plot(range(epoch+1),trainLoss,'r-',label='train')  
    plt.plot(range(epoch+1),testLoss,'g-',label='test') 
    if epoch==0:
        plt.legend(loc='upper left')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')   
    # Plotting testing accuracy vs Epochs
    fig2 = plt.figure(2)        
    plt.plot(range(epoch+1),trainAcc,'r-',label='train')    
    plt.plot(range(epoch+1),testAcc,'g-',label='test')        
    if epoch==0:
        plt.legend(loc='upper left')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')  
        
  
        
    epochEnd = time.time()-epochStart
    print('Iteration: {:.0f} /{:.0f};  Training Loss: {:.6f} ; Training Acc: {:.3f}'\
          .format(epoch + 1,iterations,avgTrainLoss,avgTrainAcc*100))
    print('Iteration: {:.0f} /{:.0f};  Testing Loss: {:.6f} ; Testing Acc: {:.3f}'\
          .format(epoch + 1,iterations,avgTestLoss,avgTestAcc*100))
   
    print('Time consumed: {:.0f}m {:.0f}s'.format(epochEnd//60,epochEnd%60))
end = time.time()-start
print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))

torch.save(net.state_dict(), 'resnet18Pre_fcOnly5class_ucf101_10adam_1e-4_b128.pt')



