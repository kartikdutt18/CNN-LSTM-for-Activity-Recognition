import os
import shutil
import numpy as np
import pickle

srcPath = 'UCF-101/'
classes = os.listdir(srcPath)
classes=classes[1:]
for c in classes:
    files = os.listdir(srcPath+c)
    for f in files:
        filename = srcPath+c+'/'+f
        dstPath = 'frames/'+c+'/'+f[:-4]
        if not os.path.exists(dstPath):
            os.makedirs(dstPath)
        os.system('ffmpeg -i {0} {1}/frame_%04d.jpg'.format(filename, dstPath))
