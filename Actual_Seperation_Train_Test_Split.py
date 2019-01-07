import os
import shutil
import numpy as np
import pickle

# Load train-test list
with open('trainList_5class.pckl','rb') as f:
    trainList = pickle.load(f)
with open('testList_5class.pckl','rb') as f:
    testList = pickle.load(f)
    
path = 'frames'
classes = os.listdir(path)
classes=classes[1:]
classes.sort()

for item in classes:
    print(item)
    srcPath = path+'/'+item
    files = os.listdir(srcPath)
    trainNum = np.floor(len(files)*0.8)
    testNum = len(files)-trainNum
    for idx in range(int(trainNum)):
        trainDst = 'train_5class/'+item+'/'+files[idx] 
        shutil.copytree(srcPath+'/'+files[idx],trainDst)         
        
    for idx2 in range(int(trainNum),int(trainNum+testNum)):
        testDst = 'test_5class/'+item+'/'+files[idx2]        
        shutil.copytree(srcPath+'/'+files[idx2],testDst)
