# Process raw data and save them into pickle file.
import os
import numpy as np
from PIL import Image
from PIL import ImageOps
from scipy import misc
import scipy.io
from skimage import io
import cv2
import sys
import cPickle as pickle
import glob
import random
from tqdm import tqdm
from eliaLib import dataRepresentation
from constants import *
import pdb

img_size = INPUT_SIZE
salmap_size = INPUT_SIZE

# Resize train/validation files
listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathOutputImages, '*'))]
listTestImages = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages, '*test*'))]


# LOAD DATA

# Train
listFilesTrain = [k for k in listImgFiles if 'train' in k]
trainData = []
for currFile in tqdm(listFilesTrain):
    trainData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.png'),
                                               os.path.join(pathOutputMaps, currFile + '.png'),
                                               os.path.join(pathToFixationMaps, currFile + '.png'),
                                               os.path.join(pathOutputweiMaps, currFile.split('_')[0] + '_'+currFile.split('_')[-1] + '.png'),
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.fixationMap,
                                               dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale))
with open(os.path.join(pathToPickle, 'train1745_180_with_fixa_wei.pickle'), 'wb') as f:
    pickle.dump(trainData, f)

# Validation
listFilesValidation = [k for k in listImgFiles if 'val' in k]
validationData = []
for currFile in tqdm(listFilesValidation):
    validationData.append(dataRepresentation.Target(os.path.join(pathOutputImages, currFile + '.png'),
                                                    os.path.join(pathOutputMaps, currFile + '.png'),
                                                    os.path.join(pathToFixationMaps, currFile + '.png'),
                                                    os.path.join(pathOutputweiMaps, currFile.split('_')[0] + '_'+currFile.split('_')[-1] + '.png'),
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.fixationMap,
                                                    dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale))
with open(os.path.join(pathToPickle, 'validation1845_180_with_fixa_wei.pickle'), 'wb') as f:
    pickle.dump(validationData, f)
