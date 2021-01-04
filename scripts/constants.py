# Work space directory
HOME_DIR = '/home/fchang/Bureau/saliency-salgan-2017/'

pathToImages = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_17/img17_45_fov120/'
pathToMaps = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_17/sal17_45_fov120/'

# Path to SALICON raw data
pathToImagesTrain = '/media/fchang/My Passport/Salient360_traindata/train/image'
pathToMapsTrain = '/media/fchang/My Passport/Salient360_traindata/train/salmap'
pathToFixationMapsTrain = '/media/fchang/My Passport/Salient360_traindata/train/fixation'

pathToImagesValid = '/media/fchang/My Passport/Salient360_traindata/valid/image'
pathToMapsValid = '/media/fchang/My Passport/Salient360_traindata/valid/salmap'
pathToFixationMapsValid = '/media/fchang/My Passport/Salient360_traindata/valid/fixation'

# Path to processed data
pathOutputImages = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_18/img_sal_fix_45/img18_45_fov180/'
#pathOutputImages = '/home/fchang/Bureau/salgauconvn_pytorch/salgan_pytorch_original/src/salicon/images256x192_val'
pathOutputMaps = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_18/img_sal_fix_45/sal18_45_fov180/'
#pathOutputMaps = '/home/fchang/Bureau/salgan_pytorch/salgan_pytorch_original/src/salicon/maps256x192_val/'
pathOutputweiMaps = '/home/fchang/Bureau/salgan360_finetune/datasets/weightMap/img17_45_wei180_5000_8/'
pathToFixationMaps = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_18/img_sal_fix_45/fix18_45_fov180_small/'
#pathToFixationMaps = '/home/fchang/Bureau/salgan_pytorch/salgan_pytorch_original/src/salicon/fixation_maps256x192/val/'

pathToPickle = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_18/'

# Path to pickles which contains processed data
#TRAIN_DATA_DIR = '/media/fchang/My Passport/Salient360_traindata/trainData.pickle'
TRAIN_DATA_DIR = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_17/train17_90_with_fixa_wei.pickle'
#TRAIN_DATA_DIR = '/home/fchang/Bureau/salgan360_finetune/datasets/salicon/train_short_salicon_with_fixa.pickle'
#VAL_DATA_DIR = '/media/fchang/My Passport/Salient360_traindata/validationData.pickle'
VAL_DATA_DIR = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_17/validation17_90_with_fixa_wei.pickle'
#VAL_DATA_DIR = '/home/fchang/Bureau/salgan360_finetune/datasets/salicon/validation_short_salicon_with_fixa.pickle'
#TEST_DATA_DIR = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_17/testData17_90_120_180fov.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/home/fchang/Bureau/saliency-salgan-2017/vgg16.pkl'

# Input image and saliency map size
INPUT_SIZE = (256, 192)

# Directory to keep snapshots
DIR_TO_SAVE = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_17/adft90/fixbceft90_bce/'
#DIR_TO_SAVE = '/home/fchang/Bureau/salgan360_finetune/datasets/salicon/'