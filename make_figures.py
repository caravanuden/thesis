    # second_rdms_sq = np.load(RESULTS_DIR + '{0}_RDM.npy'.format(savefile))
pairs = [['hyper','resnet'],['hyper','densenet'],['hyper','vgg'],['hyper','cornet-s'],['hyper','cornet-z'],['hyper','cornet-r'],['vgg','vgg'],['resnet','resnet'],['densenet','densenet'],['cornet-s','cornet-s'],['cornet-r','cornet-r'],['cornet-z','cornet-z']]
# 'cornet-z','cornet-r'
# 'cornet-z','cornet-s'
# 'cornet-s','cornet-r'
# 'cornet-s','resnet'
# 'cornet-r','densenet'
# 'cornet-z','vgg'

import sys, os
import seaborn as sns
# os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import numpy as np
from collections import OrderedDict

RESULTS_DIR = '/ihome/cara/cvu_thesis/results/'
FIG_DIR = '/ihome/cara/cvu_thesis/figures/'

rois = OrderedDict([('V1',[1,2]), ('V2',[3,4]), ('V3',[5,6]), ('V4',[7]), ('LO',[14,15]), ('V3b',[16]), ('VT',[26]), ('V3a',[17]), ('MT',[13])])
LAYER_NAMES = ['block_1', 'block_2', 'block_3', 'block_4']

model_layers = {'hyper':rois, 'anat':rois, \
                'resnet':OrderedDict([(l, []) for l in LAYER_NAMES]), 'densenet':OrderedDict([(l, []) for l in LAYER_NAMES]), \
                'vgg':OrderedDict([(l, []) for l in LAYER_NAMES]), 'squeezenet':OrderedDict([(l, []) for l in LAYER_NAMES]), \
                'cornet-s':OrderedDict([(l, []) for l in LAYER_NAMES]), 'cornet-r':OrderedDict([(l, []) for l in LAYER_NAMES]), \
                'cornet-z':OrderedDict([(l, []) for l in LAYER_NAMES])}

similarities = []
for model in pairs:
    second_rdms_sq = np.load(RESULTS_DIR + 'rsa_{0}_{1}_RDM.npy'.format(model[0], model[1]))
    layer_names = [model_layers[model[0]], model_layers[model[1]]]
    if model[0] == model[1]:
        similarities.append(second_rdms_sq[0,:])
    print(np.min(second_rdms_sq), np.max(second_rdms_sq))

    plt.figure()
    ax = plt.axes()
    # name axes with ROIs
    corr_heatmap = sns.heatmap(second_rdms_sq, vmin=0, cmap = 'magma', xticklabels=layer_names[1], yticklabels=layer_names[0], annot=True)
    ax.set_xlabel(model[1])
    ax.set_ylabel(model[0])
    ax.set_title('RSA (correlation) for {0} and {1}'.format(model[0], model[1]))

    # ax.set_xlabel('Layers')
    # ax.set_ylabel('Layers')
    # ax.set_title('RSA (correlation) for {} blocks'.format(model_names[0]))

    # ax.set_xlabel('ROIs')
    # ax.set_ylabel('ROIs')
    # ax.set_title('RSA (correlation) for hyperaligned ROIs')
    corr_heatmap.figure.savefig(FIG_DIR + 'rsa_{0}_{1}_RDM.png')

np.save('layer_similarities.npy', np.stack(similarities))
