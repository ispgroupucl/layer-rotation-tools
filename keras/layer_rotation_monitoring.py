'''
Methods for recording and plotting layer rotation curves
'''

import numpy as np
from scipy.spatial.distance import cosine

import matplotlib
import matplotlib.pyplot as plt

from keras.callbacks import Callback
import keras.backend as K
from keras.losses import categorical_crossentropy

def get_kernel_layer_names(model):
    '''
    collects name of all layers of a model that contain a kernel in topological order (input layers first).
    '''
    layer_names = []
    for l in model.layers:
        if len(l.weights) >0:
            if 'kernel' in l.weights[0].name:
                layer_names.append(l.name)
    return layer_names

def plot_layer_rotation_curves(distances, ax = None):
    '''
    utility to plot the layer-wise cosine distances between current parameters and initial parameters, 
        as measured over training (i.e. layer rotation curves).
    deviations is a list of lists with epoch index in first axis, layer index in second axis, 
        containing the cosine distances for each layer as recorded over training
    '''
    distances = np.array(distances)
    
    # get one color per layer
    cm = plt.get_cmap('viridis')
    cm_inputs = np.linspace(0,1,distances.shape[1])
    
    if not ax:
        ax = plt.subplot(1,1,1)
    for i in range(distances.shape[-1]):
        layer = i
        ax.plot(np.arange(distances.shape[0]+1), [0]+list(distances[:,layer]), label = str(layer), color = cm(cm_inputs[i]))

    ax.set_ylim([0,1.])
    ax.set_xlim([0,distances.shape[0]])
     
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine distance')

def compute_layer_rotation(current_model, initial_w):
    '''
    for each layer, computes cosine distance between current weights and initial weights
    initial_w is a list of tuples containing layer name and corresponding initial numpy weights
    '''
    s = []
    for l_name, w in initial_w:
        s.append(cosine( current_model.get_layer(l_name).get_weights()[0].flatten(), w.flatten()))
    return s

class LayerRotationCurves(Callback):
    '''
    Computes and saves layer rotation curves during training
    '''
    def __init__(self, batch_frequency=np.inf):
        '''
        batch_frequency is the frequency at which the cosine distances are computed (minimum once per epoch)
        '''
        super().__init__()
        self.batch_frequency = batch_frequency
        
        self.memory = []
    
    def set_model(self,model):
        super().set_model(model)
        layer_names = get_kernel_layer_names(model) 
        
        # initial_w is a list of tuples containing layer name and corresponding initial numpy weights
        self.initial_w = list(zip(layer_names,[model.get_layer(l).get_weights()[0] for l in layer_names]))
        
    def on_batch_end(self, batch, logs=None):
        if batch % self.batch_frequency == 0: #batch 0 is accepted, batch resets at 0 at every epoch

            dist = compute_layer_rotation(self.model, self.initial_w)

            self.memory.append(dist)
    
    def plot(self,ax = None):
        plot_layer_rotation_curves(self.memory,ax)        