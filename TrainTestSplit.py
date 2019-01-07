import os
import shutil
import numpy as np
import pickle

path = 'frames'
classes = os.listdir(path)
classes=classes[1:]
classes.sort()
trainList = []
testList = []
for c in classes:
    trainIndvList = []
    vidList = os.listdir(path+'/'+c)
    vidList=vidList[1:]
    vidList.sort()   
    for item in vidList:
        # video name eg: v_ApplyEyeMakeup_g01_c01 => g01 
        user = item.split('_')[2]
        if (user not in trainIndvList):   
            if len(trainIndvList)<20:              
                trainIndvList.append(user) # Keeping track of train-test list
                trainList.append(item) # Adding the video name to train list
            else:
                testList.append(item) # Adding the video name to test list
        else:
            trainList.append(item)

with open('trainList_5class.pckl','wb') as f:
    pickle.dump(trainList,f)
with open('testList_5class.pckl','wb') as f:
    pickle.dump(testList,f)

#Deleting Videos with frame more than 1 frame drop 
# Filtering train set
count = 1
delList = []
for item in trainList:
    print(str(count)+'/'+str(len(trainList)))
    cl = item.split('_')[1]
    srcPath = 'frames/'+cl+'/'+item    
    fNames = os.listdir(srcPath)
    fNums = [int(x[:-4].split('_')[-1]) for x in fNames]
    fNums.sort()    
    if fNums[-1]-len(fNames)>1:
        delList.append('frames/'+cl+'/'+item)
    count += 1    
for item in delList:
    shutil.rmtree(item)

# Filtering test set
count = 1
testDelList = []
for item in testList:
    print(str(count)+'/'+str(len(testList)))
    cl = item.split('_')[1]
    srcPath = 'frames/'+cl+'/'+item    
    fNames = os.listdir(srcPath)
    fNums = [int(x[:-4].split('_')[-1]) for x in fNames]
    fNums.sort()
    if fNums[-1]-len(fNames)>1: 
        testDelList.append('frames/'+cl+'/'+item)       
    count += 1   
for item in testDelList:
    shutil.rmtree(item)


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
