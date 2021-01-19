import lasagne
from lasagne.layers import InputLayer
import theano
import theano.tensor as T
import numpy as np

import generator
import discriminator
from model import Model
import pdb
import cv2
from scipy.stats.stats import pearsonr

from cube_to_equi import Cube2Equi


def KL_div(output, target, batch_size):
    #pdb.set_trace()
    #output = output*sample_map
    #target = target*sample_map
    #output = output/output.sum()
    #target = target/target.sum()   
    #a = a[sample_map > 0]
    #b = b[sample_map > 0]
    #return (a.sum()+b.sum())/(2.)
    output_reshape = T.reshape(output, (batch_size, 192*256)) 
    target_reshape = T.reshape(target, (batch_size, 192*256)) 
    batch_output_sum = T.reshape(T.extra_ops.repeat(T.sum(output_reshape, axis=1), 192*256, axis=0), (batch_size,1,192,256)) 
    batch_target_sum = T.reshape(T.extra_ops.repeat(T.sum(target_reshape, axis=1), 192*256, axis=0), (batch_size,1,192,256)) 

    output_n = output / batch_output_sum
    target_n = target / batch_target_sum    
    a = target_n * T.log(target_n/(output_n + 1e-20) + 1e-20)
    b = output_n * T.log(output_n/(target_n + 1e-20) + 1e-20)
    
    a_sum=T.sum(T.sum(T.sum(a, axis=1), axis=1), axis=1)
    b_sum=T.sum(T.sum(T.sum(b, axis=1), axis=1), axis=1)
    ab_sum = (a_sum + b_sum)/2
    return ab_sum.mean() #, 1/ab_sum.std()
    

def CC(output, target, batch_size):
    #output = output*sample_map
    #target = target*sample_map
    
    #output = T.div_proxy(T.sub(output, T.mean(output)), T.std(output))
    #target = T.div_proxy(T.sub(target, T.mean(target)), T.std(target))
    #num = T.sub(output, T.mean(output))*T.sub(target, T.mean(target))
    #out_square = T.square( T.sub(output, T.mean(output)))
    #tar_square = T.square( T.sub(target, T.mean(target)))
    #CC_score = T.sum(num)/(T.sqrt( T.sum(out_square)* T.sum(tar_square)))

    output_reshape = T.reshape(output, (batch_size, 192*256)) 
    target_reshape = T.reshape(target, (batch_size, 192*256)) 
    output_mean = T.reshape(T.extra_ops.repeat(T.mean(output_reshape, axis=1), 192*256, axis=0), (batch_size,1,192,256)) 
    target_mean = T.reshape(T.extra_ops.repeat(T.mean(target_reshape, axis=1), 192*256, axis=0), (batch_size,1,192,256)) 
    output_std = T.reshape(T.extra_ops.repeat(T.std(output_reshape, axis=1), 192*256, axis=0), (batch_size,1,192,256)) 
    target_std = T.reshape(T.extra_ops.repeat(T.std(target_reshape, axis=1), 192*256, axis=0), (batch_size,1,192,256)) 
    
    output_n = (output - output_mean) 
    target_n = (target - target_mean)
    
    num = output_n * target_n
    num_reshape = T.reshape(num, (batch_size, 192*256)) 
    out_square = T.square(output_n)
    out_square_reshape = T.reshape(out_square, (batch_size, 192*256)) 
    tar_square = T.square(target_n)
    tar_square_reshape = T.reshape(tar_square, (batch_size, 192*256)) 
    CC_score = T.sum(num_reshape, axis=1) / (T.sqrt( T.sum(out_square_reshape, axis=1)* T.sum(tar_square_reshape, axis=1)))
    
    #if T.isnan(CC_score):
    #    CC_score = 0        
    return CC_score.mean() #, 1/CC_score.std()
    
    
def NSS(output, fixationMap, batch_size):
    #output = output*sample_map
    #output = (output-output.mean())/output.std()
    output_reshape = T.reshape(output, (batch_size, 192*256)) 
    output_mean = T.reshape(T.extra_ops.repeat(T.mean(output_reshape, axis=1), 192*256, axis=0), (batch_size,1,192,256)) 
    output_std = T.reshape(T.extra_ops.repeat(T.std(output_reshape, axis=1), 192*256, axis=0), (batch_size,1,192,256)) 
    output_n = (output - output_mean) / output_std
    
    Sal = output_n*fixationMap
    Sal_reshape = T.reshape(Sal, (batch_size, 192*256))
    fixationMap_reshape = T.reshape(fixationMap, (batch_size, 192*256))
    NSS_score = T.sum(Sal_reshape, axis=1) / (T.sum(fixationMap_reshape, axis=1)+1e-20)
    
'''
    if T.isnan(NSS_score.mean()):
        NSS_mean = T.zeros(1)[0]
        NSS_std = T.ones(1)[0]
    else:
    '''
'''
    NSS_mean = NSS_score.mean()
    NSS_std = NSS_score.std()
        
    return NSS_mean #, 1/NSS_std   
'''


def KL_div(output, target):
    output = output/output.sum()
    target = target/target.sum()    
    a = target * T.log(target/(output+1e-20)+1e-20)
    b = output * T.log(output/(target+1e-20)+1e-20)
    return (a.sum()+b.sum())/(2.)

def CC(output, target):
    output = T.div_proxy(T.sub(output, T.mean(output)), T.std(output))
    target = T.div_proxy(T.sub(target, T.mean(target)), T.std(target))
    num = T.sub(output, T.mean(output))*T.sub(target, T.mean(target))
    out_square = T.square( T.sub(output, T.mean(output)))
    tar_square = T.square( T.sub(target, T.mean(target)))
    CC_score = T.sum(num)/(T.sqrt( T.sum(out_square)* T.sum(tar_square)))
    #if T.isnan(CC_score):
    #    CC_score = 0        
    return CC_score
    
def NSS(output, fixationMap):
    output = (output-output.mean())/output.std()
    Sal = output*fixationMap
    NSS_score = Sal.sum()/fixationMap.sum()
    #if T.isnan(NSS_score):
    #    NSS_score = 0
    return NSS_score   
  
    


class ModelSALGAN(Model):
    def __init__(self, w, h, batch_size=16, G_lr=3e-4, D_lr=3e-4, alpha=1/20.):
        super(ModelSALGAN, self).__init__(w, h, batch_size)

        # Build Generator
        self.net = generator.build(self.inputHeight, self.inputWidth, self.input_var)
        #self.net1 = generator.build(self.inputHeight, self.inputWidth, self.input_var)
        
        # Build Discriminator
        self.discriminator = discriminator.build(4, self.inputHeight, self.inputWidth,
                                                 T.concatenate([self.output_var_sal, self.input_var], axis=1))

        # Set prediction function
        output_layer_name = 'output'

        prediction = lasagne.layers.get_output(self.net[output_layer_name])
        test_prediction = lasagne.layers.get_output(self.net[output_layer_name], deterministic=True)
        self.predictFunction = theano.function([self.input_var], test_prediction)

        disc_lab = lasagne.layers.get_output(self.discriminator['prob'],
                                             T.concatenate([self.output_var_sal, self.input_var], axis=1))
        disc_gen = lasagne.layers.get_output(self.discriminator['prob'],
                                             T.concatenate([prediction, self.input_var], axis=1))

        # Downscale the saliency maps
        output_var_sal_pooled = T.signal.pool.pool_2d(self.output_var_sal, (4, 4), mode="average_exc_pad", ignore_border=True)
        output_var_fixa_pooled = T.signal.pool.pool_2d(self.output_var_fixa, (4, 4), mode="max", ignore_border=True)
        prediction_pooled = T.signal.pool.pool_2d(prediction, (4, 4), mode="average_exc_pad", ignore_border=True)
        
        #SampleMap_pooled = T.signal.pool.pool_2d(self.SampleMap, (4, 4), mode="average_exc_pad", ignore_border=True)
        #c2e = Cube2Equi(1024)
        #_equi = c2e.to_equi_cv2(prediction)
        
        '''
        ICME17 image dataset
        KLmiu = 2.4948
        KLstd = 1.7421
        CCmiu = 0.3932
        CCstd = 0.2565
        NSSmiu = 0.4539
        NSSstd = 0.2631
        bcemiu = 0.3194
        bcestd = 0.1209
        
        #ICME18 image dataset
        KLmiu = 2.9782 
        KLstd = 2.1767
        CCmiu = 0.3677 
        CCstd = 0.2484
        NSSmiu = 0.5635
        NSSstd = 0.2961
        bcemiu = 0.2374
        bcestd = 0.1066
        '''  
        
        #model6 
        #train_err = bcemiu+bcestd*((1.)*((KL_div(prediction_pooled, output_var_sal_pooled)-KLmiu)/KLstd) - (1.)*((CC(prediction_pooled, output_var_sal_pooled)-CCmiu)/CCstd) - (1.)*((NSS(prediction_pooled, output_var_fixa_pooled)-NSSmiu)/NSSstd))
        #model6_adaptive_weighting
        #KLsc = KL_div(prediction, self.output_var_sal, self.output_var_wei)
        #BCEsc = lasagne.objectives.binary_crossentropy(prediction_pooled, output_var_sal_pooled).mean()
        #inv_sigmaBCE = lasagne.objectives.binary_crossentropy(prediction_pooled, output_var_sal_pooled).std()
        KLsc = KL_div(prediction_pooled, output_var_sal_pooled)
        #CCsc = T.sub(1.0,  CC(prediction_pooled, output_var_sal_pooled))
        #CCsc = CC(prediction, self.output_var_sal, self.output_var_wei )
        CCsc = CC(prediction_pooled, output_var_sal_pooled)
        #NSSsc = T.sub(3.29, NSS(prediction_pooled, output_var_fixa_pooled)) 
        NSSsc = NSS(prediction_pooled, output_var_fixa_pooled)
        
        train_err = (self.inv_sigmaKL)*(KLsc) - (self.inv_sigmaCC)*(CCsc) - (self.inv_sigmaNSS)*(NSSsc)
        #train_err = (self.inv_sigmaKL)*(KLsc-self.meanKL) - (self.inv_sigmaCC)*(CCsc-self.meanCC) - (self.inv_sigmaNSS)*(NSSsc-self.meanNSS)
        #train_err = 10*(KLsc) - 2*(CCsc) - 1*(NSSsc)
        #train_err = 1.0*(KLsc) - 1.0*(CCsc) - 1.0*(NSSsc)
        #model8
        #train_err = lasagne.objectives.binary_crossentropy(prediction_pooled, output_var_sal_pooled).mean()-(bcemiu+bcestd*((1.)*((CC(prediction_pooled, output_var_sal_pooled)-CCmiu)/CCstd) + (1.)*((NSS(prediction_pooled, output_var_fixa_pooled)-NSSmiu)/NSSstd)))
        #model1
        
        #train_err = lasagne.objectives.binary_crossentropy(prediction_pooled, output_var_sal_pooled).mean()
        + 1e-4 * lasagne.regularization.regularize_network_params(self.net[output_layer_name], lasagne.regularization.l2)
        #pdb.set_trace()
        # Define loss function and input data
        ones = T.ones(disc_lab.shape)
        zeros = T.zeros(disc_lab.shape)
        D_obj = lasagne.objectives.binary_crossentropy(T.concatenate([disc_lab, disc_gen], axis=0),
                                                       T.concatenate([ones, zeros], axis=0)).mean()
        #D_obj = bcemiu+bcestd*((3.)*((KL_div(T.concatenate([disc_lab, disc_gen], axis=0), T.concatenate([ones, zeros], axis=0)).sum()-KLmiu)/KLstd) - (1.)*((CC(T.concatenate([disc_lab, disc_gen], axis=0), T.concatenate([ones, zeros], axis=0))-CCmiu)/CCstd) - (1.)*((NSS(T.concatenate([disc_lab, disc_gen], axis=0), T.concatenate([ones, zeros], axis=0))-NSSmiu)/NSSstd))
        #D_obj = (3.)*((KL_div(T.concatenate([disc_lab, disc_gen], axis=0), T.concatenate([ones, zeros], axis=0)).sum()-KLmiu)/KLstd)
        + 1e-4 * lasagne.regularization.regularize_network_params(self.discriminator['prob'], lasagne.regularization.l2)

        G_obj_d = lasagne.objectives.binary_crossentropy(disc_gen, T.ones(disc_lab.shape)).mean()
        #G_obj_d = bcemiu+bcestd*((3.)*((KL_div(disc_gen, T.ones(disc_lab.shape)).sum()-KLmiu)/KLstd) - (1.)*((CC(disc_gen, T.ones(disc_lab.shape))-CCmiu)/CCstd) - (1.)*((NSS(disc_gen, T.ones(disc_lab.shape))-NSSmiu)/NSSstd))
        + 1e-4 * lasagne.regularization.regularize_network_params(self.net[output_layer_name], lasagne.regularization.l2)

        G_obj = G_obj_d + train_err * alpha
        #cost = [G_obj, D_obj, train_err, BCEsc, KLsc, CCsc, NSSsc, inv_sigmaBCE, inv_sigmaKL, inv_sigmaCC, inv_sigmaNSS, prediction]
        #cost = [G_obj, D_obj, train_err, BCEsc, KLsc, CCsc, NSSsc, prediction]
        cost = [G_obj, D_obj, train_err, KLsc, CCsc, NSSsc, prediction]
        #cost = [G_obj, D_obj, train_err]
        
        # parameters update and training of Generator
       
        G_params = lasagne.layers.get_all_params(self.net[output_layer_name], trainable=True) 
        #self.G_lr = theano.shared(np.array(G_lr, dtype=theano.config.floatX))
        #self.G_lr = theano.shared(np.array(self.adaptive_Glr, dtype=theano.config.floatX))
        #G_updates = lasagne.updates.adagrad(G_obj, G_params, learning_rate= self.G_lr)
        G_updates = lasagne.updates.adagrad(G_obj, G_params, learning_rate=self.adaptive_Glr_in)
        self.G_trainFunction = theano.function(inputs=[self.input_var, self.output_var_sal, self.output_var_fixa, self.output_var_wei, self.inv_sigmaKL, self.inv_sigmaCC, self.inv_sigmaNSS, self.adaptive_Glr_in], outputs=cost,                                     
                                               updates=G_updates, allow_input_downcast=True,  on_unused_input='ignore')
        #self.G_trainFunction = theano.function(inputs=[self.input_var, self.output_var_sal, self.output_var_fixa], outputs=cost,                                     
        #                                       updates=G_updates, allow_input_downcast=True,  on_unused_input='ignore')

        #self.G_trainFunction = theano.function(inputs=[self.input_var, self.output_var_sal, self.output_var_fixa, self.inv_sigmaKL, self.inv_sigmaCC, self.inv_sigmaNSS, self.meanKL, self.meanCC, self.meanNSS, self.adaptive_Glr], outputs=cost,                                     
        #                                       updates=None, allow_input_downcast=True,  on_unused_input='ignore')



        # parameters update and training of Discriminator
        D_params = lasagne.layers.get_all_params(self.discriminator['prob'], trainable=True)
        #self.D_lr = theano.shared(np.array(self.adaptive_Dlr, dtype=theano.config.floatX))
        #D_updates = lasagne.updates.adagrad(D_obj, D_params, learning_rate= self.D_lr)
        D_updates = lasagne.updates.adagrad(D_obj, D_params, learning_rate=self.adaptive_Dlr_in)
        #self.D_trainFunction = theano.function([self.input_var, self.output_var_sal, self.output_var_fixa], outputs=cost, updates=D_updates,                                       
        #                                       allow_input_downcast=True, on_unused_input='ignore')
        self.D_trainFunction = theano.function([self.input_var, self.output_var_sal, self.output_var_fixa, self.output_var_wei, self.inv_sigmaKL, self.inv_sigmaCC, self.inv_sigmaNSS, self.adaptive_Dlr_in], outputs=cost, updates=D_updates,                                       
                                               allow_input_downcast=True, on_unused_input='ignore')
