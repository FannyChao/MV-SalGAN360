# A Multi-FoV Viewport-based Visual Saliency ModelUsing Adaptive Weighting Losses for 360° Images
- This repo contains the codes that used in paper [*A Multi-FoV Viewport-based Visual Saliency ModelUsing Adaptive Weighting Losses for 360° Images (TMM2020)*](https://hal.archives-ouvertes.fr/hal-02881994) by Fang-Yi Chao, Lu Zhang, Wassim Hamidouche, Olivier Deforges.
- The improved version of our previous work [*SalGAN360 (ICMEW2018)*](https://github.com/FannyChao/SalGAN360)

## Abstract
360° media allows observers to explore the scene inall directions. The consequence is that the human visual attentionis guided by not only the perceived area in the viewport but alsothe overall content in 360°. In this paper, we propose a methodto estimate the 360° saliency map which extracts salient featuresfrom the entire 360° image in each viewport in three differentField of Views (FoVs). Our model is first pretrained with a large-scale 2D image dataset to enable the interpretation of semanticcontents, then fine-tuned with a relative small 360° image dataset. A novel weighting loss function attached with stretch weightedmaps is introduced to adaptively weight the losses of three evaluation metrics and attenuate the impact of stretched regions inequirectangular projection during training process. Experimentalresults demonstrate that our model achieves better performancewith the integration of three FoVs and its diverse viewportimages. Results also show that the adaptive weighting losses andstretch weighted maps effectively enhance the evaluation scorescompared to the fixed weighting losses solutions. Comparing  toother state of the art models, our method surpasses them on three different datasets and ranks the top using 5 performanceevaluation metrics on the Salient360! benchmark set.

## Visual Results


## Requirements
- Download [SalGAN](https://github.com/imatge-upc/saliency-salgan-2017)
- Python2
- Lasagne, Theano
- Matlab

## Pretrained models
- [MV-SalGAN360 Generator Model](https://drive.google.com/drive/folders/19ib-aC5adN7lx74YQnTtPORltInq0kPA?usp=sharing)


## Usage
Replace ```01-data_preprocessing.py```, ```02-train.py```, ```03-predict.py```, ```model_salgan.py```, ```dataRepresentation.py```, ```model.py``` and ``` utils.py ``` in SalGAN. 
- Test: To predict saliency maps, run ```MV-Salgan360.m``` after specifying the path to images and the path to the output saliency maps
- Train: 
   - 1. Run ```preprocessing_trainingdata.m``` to transfer 360° images into multiple viewports.
   - 2. Run ```01-data_preprocessing.py``` to make pickle files of training images.
   - 3. Run ```02-train.py``` to fine tune salgan model.

## Citing
```
@ARTICLE{9122430,
  author={F. {Chao} and L. {Zhang} and W. {Hamidouche} and O. {Deforges}},
  journal={IEEE Transactions on Multimedia}, 
  title={A Multi-FoV Viewport-based Visual Saliency Model Using Adaptive Weighting Losses for 360° Images}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2020.3003642}}

```
