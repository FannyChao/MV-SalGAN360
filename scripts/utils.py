import os
import numpy as np
import cv2
import theano
import lasagne
#from constants import HOME_DIR
import pdb


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def load_weights(net, path, epochtoload, layernum):
    """
    Load a pretrained model
    :param epochtoload: epoch to load
    :param net: model object
    :param path: path of the weights to be set
    """
    with np.load(path + "modelWeights{:04d}.npz".format(epochtoload)) as f:
        param_values = [f['arr_%d' % i] for i in range(layernum)]
    lasagne.layers.set_all_param_values(net, param_values)


def predict(model, image_stimuli, num_epoch=None, name=None, path_output_maps=None):

    size = (image_stimuli.shape[1], image_stimuli.shape[0])
    blur_size = 5 #fy:change from 5

    if image_stimuli.shape[:2] != (model.inputHeight, model.inputWidth):
        image_stimuli = cv2.resize(image_stimuli, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)

    blob = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)

    blob[0, ...] = (image_stimuli.astype(theano.config.floatX).transpose(2, 0, 1))

    result = np.squeeze(model.predictFunction(blob))
    saliency_map = (result * 255).astype(np.uint8)

    # resize back to original size
    saliency_map = cv2.resize(saliency_map, size, interpolation=cv2.INTER_CUBIC)
    # blur
    #saliency_map = cv2.GaussianBlur(saliency_map, (blur_size, blur_size), 0)
    # clip again
    saliency_map = np.clip(saliency_map, 0, 255)
    if name is None:
        # When we use for testing, there is no file name provided.
        cv2.imwrite('./' + path_output_maps + '/validationRandomSaliencyPred_{:04d}.png'.format(num_epoch), saliency_map)
    else:
        #cv2.imwrite(os.path.join(path_output_maps, name + '_e' + str(num_epoch) + '.jpg'), saliency_map)
        cv2.imwrite(os.path.join(path_output_maps, name + '.jpg'), saliency_map)
        

def predict90(model, image_stimuli, num_epoch=None, name=None, path_output_maps=None):

    image_stimuli = np.asarray([x.image90.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in image_stimuli], dtype=theano.config.floatX)
    size = (image_stimuli.shape[3], image_stimuli.shape[2])
    blur_size = 5 #fy:change from 5

    if image_stimuli.shape[2:] != (model.inputHeight, model.inputWidth):
        image_stimuli = cv2.resize(image_stimuli, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)

    #blob = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)
    #blob[0, ...] = (image_stimuli.astype(theano.config.floatX).transpose(2, 0, 1))
    blob = image_stimuli

    result = model.predictFunction90(blob)
    
    saliency_map = (np.squeeze(result) * 255).astype(np.uint8)

    # resize back to original size
    #saliency_map = cv2.resize(saliency_map, size, interpolation=cv2.INTER_CUBIC)
    saliency_map = np.asarray([cv2.resize(saliency_map[i,:,:], size, interpolation=cv2.INTER_CUBIC) for i in range(24)])
    # blur
    #saliency_map = cv2.GaussianBlur(saliency_map, (blur_size, blur_size), 0)
    # clip again
    saliency_map = np.clip(saliency_map, 0, 255)
    if name is None:
        # When we use for testing, there is no file name provided.
        cv2.imwrite('./' + path_output_maps + '/validationRandomSaliencyPred_{:04d}.png'.format(num_epoch), saliency_map[0,:,:])
    else:
        #cv2.imwrite(os.path.join(path_output_maps, name + '_e' + str(num_epoch) + '.jpg'), saliency_map)
        cv2.imwrite(os.path.join(path_output_maps, name + '.jpg'), saliency_map[0,:,:])
        
    return result   
        
        
def predict120(model, image_stimuli, num_epoch=None, name=None, path_output_maps=None):
    image_stimuli = np.asarray([x.image120.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in image_stimuli], dtype=theano.config.floatX)
    size = (image_stimuli.shape[3], image_stimuli.shape[2])
    blur_size = 5 #fy:change from 5

    if image_stimuli.shape[2:] != (model.inputHeight, model.inputWidth):
        image_stimuli = cv2.resize(image_stimuli, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)

    #blob = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)
    #blob[0, ...] = (image_stimuli.astype(theano.config.floatX).transpose(2, 0, 1))
    blob = image_stimuli
    result = model.predictFunction120(blob)
    saliency_map = (np.squeeze(result) * 255).astype(np.uint8)

    # resize back to original size
    #saliency_map = cv2.resize(saliency_map, size, interpolation=cv2.INTER_CUBIC)
    saliency_map = np.asarray([cv2.resize(saliency_map[i,:,:], size, interpolation=cv2.INTER_CUBIC) for i in range(24)])
    # blur
    #saliency_map = cv2.GaussianBlur(saliency_map, (blur_size, blur_size), 0)
    # clip again
    saliency_map = np.clip(saliency_map, 0, 255)
    if name is None:
        # When we use for testing, there is no file name provided.
        cv2.imwrite('./' + path_output_maps + '/validationRandomSaliencyPred_{:04d}.png'.format(num_epoch), saliency_map[0,:,:])
    else:
        #cv2.imwrite(os.path.join(path_output_maps, name + '_e' + str(num_epoch) + '.jpg'), saliency_map)
        cv2.imwrite(os.path.join(path_output_maps, name + '.jpg'), saliency_map[0,:,:])
    
    return result
        
def predict180(model, image_stimuli, num_epoch=None, name=None, path_output_maps=None):

    image_stimuli = image_stimuli[::6]
    image_stimuli = np.asarray([cv2.resize(x.image180.data, (256,192)).astype(theano.config.floatX).transpose(2, 0, 1) for x in image_stimuli], dtype=theano.config.floatX)
   
    size = (image_stimuli.shape[3], image_stimuli.shape[2])
    blur_size = 5 #fy:change from 5

    if image_stimuli.shape[2:] != (model.inputHeight, model.inputWidth):
        image_stimuli = cv2.resize(image_stimuli, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)

    #blob = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)
    #blob[0, ...] = (image_stimuli.astype(theano.config.floatX).transpose(2, 0, 1))
    blob = image_stimuli
    
    result = model.predictFunction180(blob)
    saliency_map = (np.squeeze(result) * 255).astype(np.uint8)
   
    # resize back to original size
    #saliency_map = cv2.resize(saliency_map, size, interpolation=cv2.INTER_CUBIC)
    saliency_map = np.asarray([cv2.resize(saliency_map[i,:,:], size, interpolation=cv2.INTER_CUBIC) for i in range(4)])
    # blur
    #saliency_map = cv2.GaussianBlur(saliency_map, (blur_size, blur_size), 0)
    # clip again
    saliency_map = np.clip(saliency_map, 0, 255)
    if name is None:
        # When we use for testing, there is no file name provided.
        cv2.imwrite('./' + path_output_maps + '/validationRandomSaliencyPred_{:04d}.png'.format(num_epoch), saliency_map[0,:,:])
    else:
        #cv2.imwrite(os.path.join(path_output_maps, name + '_e' + str(num_epoch) + '.jpg'), saliency_map)
        cv2.imwrite(os.path.join(path_output_maps, name + '.jpg'), saliency_map[0,:,:])
        
    return result
        
def predict_C(model, image_stimuli, num_epoch=None, name=None, path_output_maps=None):

    size = (image_stimuli.shape[3], image_stimuli.shape[2])
    blur_size = 5 #fy:change from 5

    if image_stimuli.shape[2:] != (model.inputHeight_C, model.inputWidth_C):
        image_stimuli = cv2.resize(image_stimuli, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)

    #blob = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)
    #blob[0, ...] = (image_stimuli.astype(theano.config.floatX).transpose(2, 0, 1))
    blob = image_stimuli
    result = np.squeeze(model.predictFunction_C(blob))
    saliency_map = (result * 255).astype(np.uint8)

    # resize back to original size
    saliency_map = cv2.resize(saliency_map, size, interpolation=cv2.INTER_CUBIC)
    # blur
    #saliency_map = cv2.GaussianBlur(saliency_map, (blur_size, blur_size), 0)
    # clip again
    saliency_map = np.clip(saliency_map, 0, 255)
    if name is None:
        # When we use for testing, there is no file name provided.
        cv2.imwrite('./' + path_output_maps + '/validationRandomSaliencyPred_{:04d}.png'.format(num_epoch), saliency_map)
    else:
        #cv2.imwrite(os.path.join(path_output_maps, name + '_e' + str(num_epoch) + '.jpg'), saliency_map)
        cv2.imwrite(os.path.join(path_output_maps, name + '.jpg'), saliency_map)
 
        
