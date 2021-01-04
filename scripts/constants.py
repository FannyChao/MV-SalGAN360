# Work space directory
HOME_DIR = '/home/fchang/Bureau/saliency-salgan-2017/'

# Path to processed data
pathOutputImages = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_18/img_sal_fix_45/img18_45_fov180/'
pathOutputMaps = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_18/img_sal_fix_45/sal18_45_fov180/'
pathOutputweiMaps = '/home/fchang/Bureau/salgan360_finetune/datasets/weightMap/img45_wei180_5000_8/'
pathToFixationMaps = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_18/img_sal_fix_45/fix18_45_fov180/'

pathToPickle = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_18/'

# Path to pickles which contains processed data
TRAIN_DATA_DIR = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_17/train17_90_with_fixa_wei.pickle'
VAL_DATA_DIR = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_17/validation17_90_with_fixa_wei.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/home/fchang/Bureau/saliency-salgan-2017/vgg16.pkl'

# Input image and saliency map size
INPUT_SIZE = (256, 192)

# Directory to keep snapshots
DIR_TO_SAVE = '/home/fchang/Bureau/salgan360_finetune/datasets/salient360_17/adft180/'
