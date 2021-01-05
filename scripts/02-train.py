#   Two mode of training available:
#       - BCE: CNN training, NOT Adversarial Training here. Only learns the generator network.
#       - SALGAN: Adversarial Training. Updates weights for both Generator and Discriminator.
#   The training uses data previously processed using "01-data_preocessing.py"
import os
import numpy as np
import sys
import cPickle as pickle
import random
import cv2
os.environ['THEANO_FLAGS'] = "device=cuda0, force_device=True, floatX=float32"
import theano
import theano.tensor as T
import lasagne

from tqdm import tqdm
from constants import *
from models.model_salgan import ModelSALGAN
from models.model_bce import ModelBCE
from utils import *
import pdb
import matplotlib.pyplot as plt

flag = 'salgan'

def bce_batch_iterator(model, train_data, validation_sample):
    num_epochs = 301 
    n_updates = 1
    nr_batches_train = int(len(train_data) / model.batch_size)
    for current_epoch in tqdm(range(num_epochs), ncols=20):
        e_cost = 0.

        random.shuffle(train_data)

        for currChunk in chunks(train_data, model.batch_size):

            if len(currChunk) != model.batch_size:
                continue

            batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],
                                     dtype=theano.config.floatX)

            batch_output_sal = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],
                                      dtype=theano.config.floatX)
            batch_output_sal = np.expand_dims(batch_output_sal, axis=1)
            
            batch_output_fixa = np.asarray([z.fixation.data.astype(theano.config.floatX) / 255. for z in currChunk],
                                      dtype=theano.config.floatX)
            batch_output_fixa = np.expand_dims(batch_output_fixa, axis=1)

            # train generator with one batch and discriminator with next batch
            G_cost = model.G_trainFunction(batch_input, batch_output_sal, batch_output_fixa)
            e_cost += G_cost
            n_updates += 1

        e_cost /= nr_batches_train

        print 'Epoch:', current_epoch, ' train_loss->', e_cost

        if current_epoch % 5 == 0:
            np.savez(DIR_TO_SAVE + '/ft30_1_mlr_gen_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.net['output']))
            predict(model=model, image_stimuli=validation_sample, num_epoch=current_epoch, path_output_maps=DIR_TO_SAVE)


def salgan_batch_iterator(model, train_data, validation_sample):
    num_epochs = 301 
    nr_batches_train = int(len(train_data) / model.batch_size)
    n_updates = 1
    
    adaptive_Glr = 3e-5
    adaptive_Dlr = 3e-5
   
    for current_epoch in tqdm(range(num_epochs), ncols=20):
        
        g_cost = 0.
        d_cost = 0.
        e_cost = 0.

        random.shuffle(train_data)

        eval_score = np.zeros((nr_batches_train, 3))
        inv_sigmaKL = 1.0
        inv_sigmaCC = 1.0
        inv_sigmaNSS = 1.0
        inv_sigmaBCE = 1.0
        meanKL = 0.0
        meanCC = 0.0
        meanNSS = 0.0
        meanBCE = 0.0
        
        adaptive_rate = (1.0 - float(current_epoch)/float(num_epochs))**0.9
        #adaptive_rate = 1.0
        adaptive_Glr = adaptive_Glr * adaptive_rate 
        adaptive_Dlr = adaptive_Dlr * adaptive_rate
        
        for currChunk in chunks(train_data, model.batch_size):

            if len(currChunk) != model.batch_size:
                continue
            
            batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk], dtype=theano.config.floatX)   # batch_input = (batch_size, 3, 192, 256)
            
            batch_output_sal = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk], dtype=theano.config.floatX)
            batch_output_sal = np.expand_dims(batch_output_sal, axis=1)
            
            batch_output_fixa = np.asarray([z.fixation.data.astype(theano.config.floatX) / 255. for z in currChunk], dtype=theano.config.floatX)
            batch_output_fixa = np.expand_dims(batch_output_fixa, axis=1)
          
            batch_output_wei = np.asarray([ (y.weimap.data.astype(theano.config.floatX) +1.0 )/255. for y in currChunk], dtype=theano.config.floatX)
            batch_output_wei = np.expand_dims(batch_output_wei, axis=1)
            
            # train generator with one batch and discriminator with next batch
            if n_updates % 2 == 0:
                G_obj, D_obj, G_cost, KLsc, CCsc, NSSsc, prediction = model.G_trainFunction(batch_input, batch_output_sal, batch_output_fixa, batch_output_wei, inv_sigmaKL, inv_sigmaCC, inv_sigmaNSS, adaptive_Glr)
                #G_obj, D_obj, G_cost = model.G_trainFunction(batch_input, batch_output_sal, batch_output_fixa)
                d_cost += D_obj
                g_cost += G_obj
                e_cost += G_cost
            else:
                G_obj, D_obj, G_cost, KLsc, CCsc, NSSsc, prediction = model.D_trainFunction(batch_input, batch_output_sal, batch_output_fixa, batch_output_wei, inv_sigmaKL, inv_sigmaCC, inv_sigmaNSS, adaptive_Dlr)
                #G_obj, D_obj, G_cost = model.D_trainFunction(batch_input, batch_output_sal, batch_output_fixa)
                d_cost += D_obj
                g_cost += G_obj
                e_cost += G_cost
              
            n_updates += 1

        g_cost /= nr_batches_train
        d_cost /= nr_batches_train
        e_cost /= nr_batches_train
        
        inv_sigmaKL = 1.0 / eval_score[:, 0].std()
        inv_sigmaCC = 1.0 / eval_score[:, 1].std()
        inv_sigmaNSS = 1.0 / eval_score[:, 2].std()
       
        meanKL = eval_score[:, 0].mean()
        meanCC = eval_score[:, 1].mean()
        meanNSS = eval_score[:, 2].mean()
               
        
        print '\n std: ', eval_score[:, 0].std(), eval_score[:, 1].std(), eval_score[:, 2].std()
        print 'mean: ', meanKL, meanCC, meanNSS
        
        # Save weights every 10 epoch
        if current_epoch % 10 == 0:
            np.savez(DIR_TO_SAVE + '/1745_90_gen_modelWeights{:04d}.npz'.format(current_epoch), *lasagne.layers.get_all_param_values(model.net['output']))
            np.savez(DIR_TO_SAVE + '/1745_90_discrim_modelWeights{:04d}.npz'.format(current_epoch), *lasagne.layers.get_all_param_values(model.discriminator['prob']))
            predict(model=model, image_stimuli=validation_sample, num_epoch=current_epoch, name='val', path_output_maps=DIR_TO_SAVE)
        print 'Epoch:', current_epoch, ' train_loss->', (g_cost, d_cost, e_cost)


def train():
    """
    Train both generator and discriminator
    :return:
    """
    # Load data
    print 'Loading training data...'
    with open(TRAIN_DATA_DIR, 'rb') as f:
        train_data = pickle.load(f)
    print '-->done!'
    print 'Loading validation data...'
    with open(VAL_DATA_DIR, 'rb') as f:
        validation_data = pickle.load(f)
    print '-->done!'

    # Choose a random sample to monitor the training
    num_random = random.choice(range(len(validation_data)))
    validation_sample = validation_data[num_random]

    cv2.imwrite(DIR_TO_SAVE + '/validationRandomSaliencyGT.png', validation_sample.saliency.data)
    cv2.imwrite(DIR_TO_SAVE + '/validationRandomImage.png', cv2.cvtColor(validation_sample.image.data, cv2.COLOR_RGB2BGR))

    # Create network

    if flag == 'salgan':
        model = ModelSALGAN(INPUT_SIZE[0], INPUT_SIZE[1])
        # Load a pre-trained model
        load_weights(net=model.net['uconv2_1'], path="gen_", epochtoload=90, layernum=48) # full layernum=54
        load_weights(net=model.discriminator['prob'], path='discrim_', epochtoload=90, layernum=20) # full layernum=20
        salgan_batch_iterator(model, train_data, validation_sample.image.data)

    elif flag == 'bce':
        model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1])
        # Load a pre-trained model
        #load_weights(net=model.net['output'], path='gen_', epochtoload=90) # load pretrained BCE model
        bce_batch_iterator(model, train_data, validation_sample.image.data)
    else:
        print "Invalid input argument."
if __name__ == "__main__":
    train()
  
