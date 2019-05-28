# RDMs for RSA analysis
# want two sets of RDMs:
#   subject-wise for hyperalignment sanity check comparing before and after hyperalignment- average across time points
#   timepoint-wise for comparing DNN and brain - average across subjects
#   voxel-wise for comparing DNN and brain - average across subjects
# so have data per subject per voxel per timepoint - how to average?
# CVU 2019

import mvpa2.suite as mv
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import KernelPCA
# from rdm import *
import sys, os
os.environ['QT_QPA_PLATFORM']='offscreen'

import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy.stats import gamma, zscore, mode

import pdb
# import mvpa2.suite as mv

# import vtk_utils
# import warnings
# warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=DeprecationWarning)

MODEL_DATA_DIR = '/idata/DBIC/cara/thesis/image_activations/'

ANAT_DATA_DIR = '/idata/DBIC/fma/id_studies/home/preprocess/final/raiders-32ch/'
HYPER_DATA_DIR = '/ihome/cara/cvu_thesis/hyper_raiders_brain_data/'
REDUCED_MODEL_DATA_DIR = '/idata/DBIC/cara/thesis/reduced_activations/'
RESULTS_DIR = '/ihome/cara/cvu_thesis/rdm_results/'
FIG_DIR = '/ihome/cara/cvu_thesis/rdm_figures/'
hemispheres = OrderedDict([('lh',np.load('/ihome/cara/cvu_thesis/thesis/surface/fsaverage_lh_mask.npy')), ('rh',np.load('/ihome/cara/cvu_thesis/thesis/surface/fsaverage_rh_mask.npy'))])
#
# MODEL_DATA_DIR = '/Users/caravanuden/thesis/image_activations/'
# REDUCED_MODEL_DATA_DIR = '/Users/caravanuden/thesis/reduced_activations/'
# RESULTS_DIR = '/Users/caravanuden/thesis/rsa_results/'
# FIG_DIR = '/Users/caravanuden/thesis/figures/'

voxels = {'lh':9372, 'rh':9370}
# subjects = ['sub-rid000001', 'sub-rid000008', 'sub-rid000009', 'sub-rid000012', 'sub-rid000013', \
#             'sub-rid000016', 'sub-rid000017', 'sub-rid000018', 'sub-rid000019', 'sub-rid000021', \
#             'sub-rid000022', 'sub-rid000024', 'sub-rid000025', 'sub-rid000026', 'sub-rid000027', \
#             'sub-rid000031', 'sub-rid000032', 'sub-rid000036', 'sub-rid000037', 'sub-rid000041']

subjects = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
            'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
            'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
            'sub-rid000037','sub-rid000038','sub-rid000041']

# RUNS_LEN = OrderedDict([(1, 336), (2,344), (3,344), (4,326), (5,339), (6,344), (7,344), (8,340)])
# RUNS = list(RUNS_LEN.keys())

# NOTE: run 5 of the movie is 2s (1 TR) shorter than the MRI data
# RUNS_LEN = OrderedDict([(5,340), (6,344), (7,344), (8,340)])
# RUNS = list(RUNS_LEN.keys())

RUNS_LEN = OrderedDict([(1,369), (2,341), (3,372), (4,406)])
# RUNS = OrderedDict([(1,374), (2,346), (3,377), (4,412)])
RUNS = OrderedDict([(1,374), (2,346), (3,377)])
BRAIN_MODELS = ['hyper', 'anat']
tr_fmri = {1:374, 2:346, 3:377, 4:412}

sam_data_dir = '/idata/DBIC/snastase/life'
mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'

# rois = OrderedDict([('V1',[1, 2]), ('V2',[3, 4]), ('V3',[5, 6]), ('V3a',[17]), ('V3b',[16]), ('V4',[7]), ('MT',[13]), ('LO',[14,15]), ('VO',[8,9])])
# rois = OrderedDict([('V1',[1, 2]), ('V2',[3, 4]), ('V3',[5, 6]), ('V3a',[17]), ('V4',[7]), ('VO',[8,9]), ('V3b',[16]), ('LO',[14,15]), ('MT',[13])])
rois = OrderedDict([('V1',[1,2]), ('V2',[3,4]), ('V3',[5,6]), ('V3a',[17]), ('V3b',[16]), ('V4',[7]), ('VO',[8,9]), ('PHC',[10,11]), ('LO',[14,15]), ('MT',[13]), ('VT',[26])])
layer_names = ['block_0', 'block_1', 'block_2', 'block_3', 'block_4']

model_layers = {'hyper':rois, 'anat':rois, \
                'resnet':OrderedDict([(l, []) for l in layer_names]), 'densenet':OrderedDict([(l, []) for l in layer_names]), \
                'vgg':OrderedDict([(l, []) for l in layer_names]), \
                'cornet-s':OrderedDict([(l, []) for l in layer_names]), 'cornet-r':OrderedDict([(l, []) for l in layer_names]), \
                'cornet-z':OrderedDict([(l, []) for l in layer_names])}

SUBJECTS = ['sub-rid000001', 'sub-rid000008', 'sub-rid000009', 'sub-rid000012', 'sub-rid000013', \
            'sub-rid000016', 'sub-rid000017', 'sub-rid000018', 'sub-rid000019', 'sub-rid000021', \
            'sub-rid000022', 'sub-rid000024', 'sub-rid000025', 'sub-rid000026', 'sub-rid000027', \
            'sub-rid000031', 'sub-rid000032', 'sub-rid000036', 'sub-rid000037', 'sub-rid000041']

RUNS = {'5-6-7-8':range(5,9)}
RUNS_LEN = OrderedDict([(5,340), (6,344), (7,344), (8,340)])
HEMIS = ['lh', 'rh']

LOAD_DIR = '/idata/DBIC/fma/id_studies/home/preprocess/aligned/raiders-32ch/'
SAVE_DIR = '/idata/DBIC/cara/thesis/hyper_raiders/'

def process_runs(resp):
    ind = 0
    data = []
    for run in RUNS_LEN.keys():
        # print(ind, ind+RUNS_LEN[run])
        run_data = resp[ind:ind+RUNS_LEN[run],:]
        # run_data = run_data[2:,:]
        if run == 5:
            run_data = run_data[:-1,:]
        run_data = zscore(run_data, axis=0)
        data.append(run_data)
        ind += RUNS_LEN[run]

    return data

def load_brain_data(dataset):
    if dataset == 'hyper':
        runs = RUNS.keys()
    else:
        runs = RUNS.values()
    brain_list = []
    for subj in SUBJECTS:
        run_list = []
        for run in runs:
            hemi_list = []
            for hemi in HEMIS:
                if dataset == 'hyper':
                    resp = np.load(SAVE_DIR + '{0}_runs-{1}_{2}.npy'.format(subj, run, hemi))
                    resp_run_list = process_runs(resp)
                    resp = np.concatenate(resp_run_list,axis=0)
                    # print(resp.shape)
                    # resp = np.load('/ihome/cara/cvu_thesis/hyperaligned_life/hyperaligned_{}_{}_{}.npy'.format(subj, run, hemi))
                else:
                    resp = np.load('/idata/DBIC/fma/id_studies/home/preprocess/final/raiders-32ch/{0}/{0}_task-raiders_run-0{1}_{2}_freq32.npy'.format(subj, run, hemi))
                    # resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(subj, tr_fmri[run], run, hemi))).samples
                hemi_list.append(resp)
            run_list.append(np.concatenate(hemi_list, axis=1))
        subj_resp = np.concatenate(run_list, axis=0)
        brain_list.append(subj_resp)

    # for isc
    brain_data = np.stack(brain_list,axis=0)
    print(brain_data.shape)
    return brain_data

def get_max_norm_features(dat):
    max_dat = np.zeros([dat.shape[0],dat.shape[2],dat.shape[3]])
    for i in range(dat.shape[0]):
        m = np.linalg.norm(dat[i,:,:,:], axis=(1,2))
        max_dat[i,:,:] = dat[i,np.argmax(m),:,:]
    return max_dat

def load_layer_data(model_name, layer_names, PATH):
    reduced_data_dir = REDUCED_MODEL_DATA_DIR + model_name

    print(layer_names)
    print(reduced_data_dir)

    layers = []
    for layer in layer_names:
        if not os.path.exists(reduced_data_dir + '/processed_{0}.npy'.format(layer)):
            if not os.path.exists(reduced_data_dir):
                os.makedirs(reduced_data_dir)
            curr_layer = []
            for run in RUNS_LEN.keys():
                # dat = np.load(PATH + 'test_activations_{1}.npy'.format(run, layer))
                dat = np.load(PATH + 'part_{0}_activations_{1}.npy'.format(run, layer))

                flat = dat.reshape([dat.shape[0], dat.shape[1]*dat.shape[2]*dat.shape[3]])
                # print(flattened.shape)
                N = 5
                averaged = []
                for slice in range(0,flat.shape[0],N):
                    # averaged.append(np.mean(flat[slice:slice+N,:],axis=0))
                    averaged.append(np.linalg.norm(flat[slice:slice+N,:],axis=0))

                averaged = np.stack(averaged)
                convolved_out = np.apply_along_axis(hrf_convolve, 0, averaged)
                # print(convolved_out.shape, out.shape)
                out = convolved_out[:averaged.shape[0],:]
                out = out[8:,:]
                out = zscore(out,axis=0)
                print(run, out.shape)

                # print(run, out.shape)
                # movies overlap by 8 TRs
                curr_layer.append(out)
            curr_layer = np.concatenate(curr_layer,axis=0)
            curr_layer = curr_layer[:, ~np.isnan(curr_layer).any(axis=0)]
            print(curr_layer.shape)
            layers.append(curr_layer)
            np.save(reduced_data_dir + '/processed_{0}.npy'.format(layer), curr_layer)

        else:
            layer = np.load(reduced_data_dir + '/processed_{0}.npy'.format(layer))
            print(layer.shape)
            layers.append(layer)

    return layers

def first_order_ISC(data, name):
    mat = np.zeros((data.shape[2]))
    print(data.shape[2])
    # for timepoint in range(brain_data.shape[1]):
    print('Starting first order ISC per voxel')
    for voxel in range(data.shape[2]):
        mat[voxel] = np.mean(1-pdist(data[:,:, voxel], metric='correlation'))

    np.save(RESULTS_DIR + 'first_order_ISC_lh_{0}.npy'.format(name), mat[:voxels['lh']])
    np.save(RESULTS_DIR + 'first_order_ISC_rh_{0}.npy'.format(name), mat[voxels['lh']:])

    # do this locally bc issues with vtk_utils
    # vtk_utils.plot([mat[:voxels['lh']],  mat[voxels['lh']:]], 1, 'first_order_ISC_{0}.png'.format(name))

def first_order_RDM(data):
    rdms = []
    # for timepoint in range(brain_data.shape[1]):
    print('Starting first order RDM per timepoint')
    for roi in data:
        # print('mean and std across time: {0}, {1}'.format(np.mean(roi), np.std(roi)))
        rdm = 1-pdist(roi, metric='correlation')
        # print(rdm.shape)
        rdms.append(rdm)

    # np.save(RESULTS_DIR + 'first_order_RDM_{0}.npy'.format(name), mat)

    # plt.figure()
    # # name axes with ROIs
    # corr_heatmap = sns.heatmap(second_mat_sq, cmap = 'magma', square = True)
    # corr_heatmap.figure.savefig('first_order_RDM_{0}.npy'.format(name))

    return rdms

def second_order_RDM(model_names, layer_names, data):
    savefile = 'rsa_{0}'.format('_'.join(model_names))
    print('Starting second order RDM per ROI/layer')
    rdms = first_order_RDM(data[0]+data[1])

    rdms = np.stack(rdms, axis=0)
    print(rdms.shape)
    print('Starting second order RDM across ROIs/layers')
    second_rdms = 1-pdist(rdms, metric='correlation')
    second_rdms_sq = squareform(second_rdms)
    second_rdms_sq = second_rdms_sq[:len(layer_names[0]), len(layer_names[0]):]
    print(second_rdms_sq.shape)
    print(second_rdms_sq)

    np.save(RESULTS_DIR + '{0}_RDM.npy'.format(savefile), second_rdms_sq)
    # second_rdms_sq = np.load(RESULTS_DIR + '{0}_RDM.npy'.format(savefile))

    print(np.min(second_rdms_sq), np.max(second_rdms_sq))

    plt.figure()
    ax = plt.axes()
    # name axes with ROIs
    corr_heatmap = sns.heatmap(second_rdms_sq, vmin=0, cmap = 'magma', xticklabels=layer_names[1], yticklabels=layer_names[0], annot=True)
    ax.set_xlabel(model_names[1])
    ax.set_ylabel(model_names[0])
    ax.set_title('RSA (correlation) for {0} and {1}'.format(model_names[0], model_names[1]))

    # ax.set_xlabel('Layers')
    # ax.set_ylabel('Layers')
    # ax.set_title('RSA (correlation) for {} blocks'.format(model_names[0]))

    # ax.set_xlabel('ROIs')
    # ax.set_ylabel('ROIs')
    # ax.set_title('RSA (correlation) for hyperaligned ROIs')
    corr_heatmap.figure.savefig(FIG_DIR + '{0}_RDM.png'.format(savefile))


def cka(dataList1, dataList2):
    """
    This function computes the RV matrix correlation coefficients between pairs
    of arrays. The number and order of objects (rows) for the two arrays must
    match. The number of variables in each array may vary.
    INPUT
    ----------
    dataList : list
    A list holding numpy arrays for which the RV coefficient will be computed.
    RETURNS
    -------
    numpy array
    A numpy array holding RV coefficients for pairs of numpy arrays. The
    diagonal in the result array holds ones, since RV is computed on
    identical arrays, i.e. first array in ``dataList`` against frist array
    in
    """
    print('doing CKA')
    # First compute the scalar product matrices for each data set X
    scalArrList1 = []
    for arr in dataList1:
        # center the data
        arr -= np.mean(arr, axis=0)
        scalArr = np.dot(arr, np.transpose(arr))
        scalArrList1.append(scalArr)

    # First compute the scalar product matrices for each data set X
    scalArrList2 = []
    for arr in dataList2:
        # center the data
        arr -= np.mean(arr, axis=0)
        scalArr = np.dot(arr, np.transpose(arr))
        scalArrList2.append(scalArr)
    # Now compute the 'between study cosine matrix' C
    C = np.zeros((len(dataList1), len(dataList2)), float)
    for index, element in np.ndenumerate(C):
        nom = np.trace(np.dot(np.transpose(scalArrList1[index[0]]),
                            scalArrList2[index[1]]))
        denom1 = np.trace(np.dot(np.transpose(scalArrList1[index[0]]),
                               scalArrList1[index[0]]))
        denom2 = np.trace(np.dot(np.transpose(scalArrList2[index[1]]),
                               scalArrList2[index[1]]))
        Rv = nom / np.sqrt(np.dot(denom1, denom2))
        C[index[0], index[1]] = Rv
    return C

def return_rois(brain_data):
    roi_names = list(rois.keys())

    lh_roi = np.load('/dartfs-hpc/scratch/f001693/wang_atlas/maxprob_fsaverage6_lh.npy')[:10242]
    lh_roi = lh_roi[hemispheres['lh'][:10242]]
    rh_roi = np.load('/dartfs-hpc/scratch/f001693/wang_atlas/maxprob_fsaverage6_rh.npy')[:10242]
    rh_roi = rh_roi[hemispheres['rh'][:10242]]

    roi_mask = np.concatenate([lh_roi, rh_roi])
    roi_brain_data = []

    for roi in roi_names:
        if roi == 'VT':
            lh_roi = mv.niml.read('/ihome/cara/cvu_thesis/lh.mask_VT.niml.dset').samples[:,:10242]
            lh_roi = lh_roi[:,hemispheres['lh'][:10242]]
            rh_roi = mv.niml.read('/ihome/cara/cvu_thesis/rh.mask_VT.niml.dset').samples[:,:10242]
            rh_roi = rh_roi[:, hemispheres['rh'][:10242]]

            roi_mask_VT = np.concatenate([lh_roi, rh_roi],axis=1)
            print(brain_data.shape, roi_mask_VT.shape)
            dat = brain_data[:,np.where(roi_mask_VT[0,:] == 1)]
        else:
            dat = brain_data[:,np.where(np.isin(roi_mask, rois[roi]))]
        dat = np.reshape(dat, [dat.shape[0], dat.shape[2]])
        print(roi, dat.shape)
        roi_brain_data.append(dat)

    return roi_brain_data

def load_brain_roi_data(align, roi_names, average=True):
    roi_brain_data = []
    brain_data = load_brain_data(align)

    if average:
        avg_brain_data = np.mean(brain_data, axis=0)
        roi_brain_data = return_rois(avg_brain_data)
    else:
        roi_brain_data = [return_rois(brain_data[subj]) for subj in range(brain_data.shape[0])]

    return roi_brain_data

def get_data_lists(model_names, mydict, average):
    layers = []
    for m in model_names:
        # for k,v in mydict.iteritems():
        # python 3
        for k,v in mydict.items():
            if k == m:
                layers.append(v)
    # print(layers)
    compare_brain = any([m in BRAIN_MODELS for m in model_names])

    layer_names = []
    data = []
    if model_names[0] == model_names[1]:
        curr_layer_names = list(layers[0].keys())
        layer_names.append(curr_layer_names)
        layer_names.append(curr_layer_names)
        if model_names[0] in BRAIN_MODELS:
            curr_data = load_brain_roi_data(model_names[0], curr_layer_names, average)
        else:
            curr_data = load_layer_data(model_names[0], curr_layer_names, MODEL_DATA_DIR + '{0}/'.format(model_names[0]))
        data.append(curr_data)
        data.append(curr_data)
    else:
        for i, curr_model_name in enumerate(model_names):
            curr_layer_names = list(layers[i].keys())
            layer_names.append(curr_layer_names)
            if curr_model_name in BRAIN_MODELS:
                data.append(load_brain_roi_data(curr_model_name, curr_layer_names, average))
            else:
                data.append(load_layer_data(curr_model_name, curr_layer_names, MODEL_DATA_DIR + '{0}/'.format(curr_model_name)))

    return (layer_names, data)

def run_cka(model_names, layer_names, data):
    savefile = 'cka_{0}'.format('_'.join(model_names))

    # compute CKA (RV matrix correlation) coefficients on mean centered data
    print(len(data))
    cka_results = cka(data[0], data[1])
    print(cka_results.shape)
    print(cka_results)

    np.save(RESULTS_DIR + '{0}.npy'.format(savefile), cka_results)
    # cka_results = np.load(RESULTS_DIR + '{0}.npy'.format(savefile))

    plt.figure()
    ax = plt.axes()
    # name axes with ROIs

    corr_heatmap = sns.heatmap(cka_results,cmap = 'magma', vmin=0, xticklabels=layer_names[1], yticklabels=layer_names[0], annot=True)
    ax.set_xlabel(model_names[1])
    ax.set_ylabel(model_names[0])
    ax.set_title('RSA (CKA) for {0} and {1}'.format(model_names[0], model_names[1]))

    # ax.set_xlabel('Layers')
    # ax.set_ylabel('Layers')
    # ax.set_title('RSA (CKA) for {} layers'.format(model_names[0]))

    # ax.set_xlabel('ROIs')
    # ax.set_ylabel('ROIs')
    # ax.set_title('RSA (CKA) for hyperaligned ROIs')
    corr_heatmap.figure.savefig(FIG_DIR + '{0}_RDM.png'.format(savefile))

def get_cka_sim(model_names, mydict):
    print('getting CKA max ROI sim for each block across subjects')
    savefile = 'cka_similarity_{0}'.format('_'.join(model_names))

    max_cka_list = []
    layer_names, data = get_data_lists(model_names, mydict, average=False)

    for subj in range(len(SUBJECTS)):
        # compute CKA (RV matrix correlation) coefficients on mean centered data
        cka_results = cka(data[0][subj], data[1])
        max_cka = np.argmax(cka_results, axis=0)
        print(max_cka, max_cka.shape)
        # np.save(RESULTS_DIR + '{0}.npy'.format(savefile), cka_results)
        max_cka_list.append(max_cka)
    max_cka = np.stack(max_cka_list)
    modes = mode(max_cka,axis=0).mode
    print(max_cka, max_cka.shape)
    dist = pdist(max_cka, 'jaccard')
    print(dist.shape, squareform(dist.shape))
    similarity = np.mean(dist)
    return (similarity, modes)

def hrf(times):
    """ Return values for HRF at given times """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

def hrf_convolve(data):
    TR = 2.5
    tr_times = np.arange(0, 20, TR)
    hrf_at_trs = hrf(tr_times)

    return(np.convolve(data, hrf_at_trs))

def plot_isc():
    import vtk_utils
    import numpy as np

    vmin=0
    vmax = .6
    lh = np.load('cornet-z_all_avg_lh_corrs.npy')
    rh = np.load('cornet-z_all_avg_rh_corrs.npy')
    vtk_utils.plot(values_li=[lh, rh], vmin=vmin, vmax=vmax,out_fn='cornet-z_all_ridge.png', cmap='magma')

    lh = np.load('results/first_order_ISC_lh_anat.npy')
    rh = np.load('results/first_order_ISC_rh_anat.npy')
    vtk_utils.plot(values_li=[lh, rh], vmin=vmin, vmax=vmax, out_fn='first_order_ISC_anat.png', cmap='magma')

# pairs = [['hyper','cornet-r'],['vgg','vgg'],['resnet','resnet'],['densenet','densenet'],['cornet-s','cornet-s'],['cornet-r','cornet-r'],['cornet-z','cornet-z']]
pairs = [['hyper','hyper'],['hyper','resnet'],['hyper','densenet'],['hyper','vgg'],['hyper','cornet-s'],['hyper','cornet-z'],['hyper','cornet-r']]
# pairs = [['vgg','vgg'],['resnet','resnet'],['densenet','densenet'],['cornet-s','cornet-s'],['cornet-r','cornet-r'],['cornet-z','cornet-z']]
# pair = [sys.argv[1], sys.argv[2]]

for pair in pairs:
    print(pair)

    layer_names, data = get_data_lists(pair, model_layers, average=True)

    # second_order_RDM(pair, layer_names, data)
    run_cka(pair, layer_names, data)
    # if pair[0] == 'hyper' and pair[0] != pair[1]:
    #     similarities[pair[1]] = get_cka_sim(pair, model_layers)

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# from collections import OrderedDict
#
# # rois = OrderedDict([('V1',[1,2]), ('V2',[3,4]), ('V3',[5,6]), ('V3a',[17]), ('V3b',[16]), ('V4',[7]), ('VO',[8,9]), ('PHC',[10,11]), ('LO',[14,15]), ('MT',[13]), ('VT',[26])])
# rois = OrderedDict([('V1',[1,2]), ('V2',[3,4]), ('V3',[5,6]), ('V3a',[17]), ('V3b',[16]), ('V4',[7]), ('LO',[14,15]), ('MT',[13]), ('VT',[26])])
# layer_names = ['block_1', 'block_2', 'block_3', 'block_4']
# longer_layer_names = ['block_0', 'block_1', 'block_2', 'block_3', 'block_4']
#
# model_layers = {'hyper':rois, 'anat':rois, \
#                 'resnet':OrderedDict([(l, []) for l in layer_names]), 'densenet':OrderedDict([(l, []) for l in layer_names]), \
#                 'vgg':OrderedDict([(l, []) for l in layer_names]), 'squeezenet':OrderedDict([(l, []) for l in layer_names]), \
#                 'cornet-s':OrderedDict([(l, []) for l in layer_names]), 'cornet-r':OrderedDict([(l, []) for l in layer_names]), \
#                 'cornet-z':OrderedDict([(l, []) for l in layer_names])}
#
# cka = []
# layer_assign = np.zeros((9,6))
# ind=0
# for model in ['cornet-z','cornet-s', 'cornet-r','vgg','resnet', 'densenet']:
#     savefile = 'cka_hyper_{0}'.format(model)
#     cka_results = np.load(savefile + '.npy')
#     print(model, np.mean(np.max(cka_results,axis=1)))
#     cka.append(np.mean(np.max(cka_results,axis=1)))
#     layer_assign[:,ind] = np.argmax(cka_results,axis=1) + 1
#     plt.figure()
#     ax = plt.axes()
#     # name axes with ROIs
#
#     corr_heatmap = sns.heatmap(cka_results,cmap = 'magma', vmin=0, vmax = .2, xticklabels=model_layers[model], yticklabels=model_layers['hyper'], annot=True)
#     ax.set_xlabel(model)
#     ax.set_ylabel('hyper')
#     ax.set_title('RSA (CKA) for hyper and {0}'.format(model))
#     corr_heatmap.figure.savefig('{0}_RDM.png'.format(savefile))
#     plt.show()
#     ind+=1
#
# plt.figure()
# ax = plt.axes()
# # name axes with ROIs
#
# corr_heatmap = sns.heatmap(layer_assign,cmap = 'magma', vmin=1, xticklabels=['cornet-z','cornet-s', 'cornet-r','vgg','resnet', 'densenet'], yticklabels=model_layers['hyper'], annot=True)
# ax.set_xlabel('CNN Models')
# ax.set_ylabel('Brain ROIs')
# ax.set_title('RSA layer assignment by brain ROI')
# corr_heatmap.figure.savefig('rsa_layer_assign.png')
# plt.show()
#
# columns=['model', 'imagenet_top1', 'whole_brain_pred', 'ROI_pred', 'layer_similarity']
#
# df = pd.DataFrame([['cornet-z', 0.48, 0.14, 0.27, 0.851], \
#                    ['cornet-s', 0.74, 0.16, 0.32, 0.732], \
#                    ['cornet-r', 0.56, 0.17, 0.33, 0.795], \
#                    ['vgg', 0.724, .16, 0.32, 0.813], \
#                    ['resnet', 0.783, 0.19, 0.35, 0.628], \
#                    ['densenet', 0.777, 0.13, 0.27, 0.597]], \
#                    columns=columns)
#
# df['cka'] = np.asarray(cka)
# df['layer_dissimilarity'] = 1- df['layer_similarity']
# plt.figure()
# p1 = sns.scatterplot('imagenet_top1', # Horizontal axis
#        'cka', # Vertical axis
#        data=df, # Data source
#        legend=False)
#
# for line in range(0,df.shape[0]):
#      p1.text(df.imagenet_top1[line]-.01, df.cka[line]+.001,
#      df.model[line], horizontalalignment='left',
#      size='small', color='black')
#
# plt.title('ImageNet Top-1 vs CKA Similarity Score')
# # Set x-axis label
# plt.xlabel('ImageNet Top-1')
# # Set y-axis label
# plt.ylabel('CKA Score Across ROIs')
# p1.figure.savefig('imagenet_top1_vs_cka.png')
# plt.show()
#
# p1 = sns.scatterplot('layer_dissimilarity', # Horizontal axis
#        'cka', # Vertical axis
#        data=df, # Data source
#        legend=False)
#
# for line in range(0,df.shape[0]):
#      p1.text(df.layer_dissimilarity[line]-.01, df.cka[line]+.001,
#      df.model[line], horizontalalignment='left',
#      size='small', color='black')
#
# plt.title('Layer Dissimilarity vs CKA Similarity Score')
# # Set x-axis label
# plt.xlabel('Layer Dissimilarity')
# # Set y-axis label
# plt.ylabel('CKA Score Across ROIs')
# p1.figure.savefig('similarity_vs_cka.png')
# plt.show()
#
# plt.figure()
# p1 = sns.scatterplot('layer_dissimilarity', # Horizontal axis
#        'imagenet_top1', # Vertical axis
#        data=df, # Data source
#        legend=False)
#
# for line in range(0,df.shape[0]):
#      p1.text(df.layer_dissimilarity[line]-.01, df.imagenet_top1[line]+.001,
#      df.model[line], horizontalalignment='left',
#      size='small', color='black')
#
# plt.title('Layer Dissimilarity vs ImageNet-Top1 Performance')
# # Set x-axis label
# plt.xlabel('Layer Dissimilarity')
# # Set y-axis label
# plt.ylabel('ImageNet Top-1')
# p1.figure.savefig('similarity_vs_imagenet_top1.png')
# plt.show()
#
# plt.figure()
# p1 = sns.scatterplot('layer_dissimilarity', # Horizontal axis
#        'ROI_pred', # Vertical axis
#        data=df, # Data source
#        legend=False)
#
# for line in range(0,df.shape[0]):
#      p1.text(df.layer_dissimilarity[line]-.01, df.ROI_pred[line]+.001,
#      df.model[line], horizontalalignment='left',
#      size='small', color='black')
#
# plt.title('Layer Dissimilarity vs ROI Forward Encoding Performance')
# # Set x-axis label
# plt.xlabel('Layer Dissimilarity')
# # Set y-axis label
# plt.ylabel('ROI Forward Encoding Performance')
# p1.figure.savefig('similarity_vs_roi.png')
# plt.show()
#
# p1 = sns.scatterplot('ROI_pred', # Horizontal axis
#        'cka', # Vertical axis
#        data=df, # Data source
#        legend=False)
#
# for line in range(0,df.shape[0]):
#      p1.text(df.ROI_pred[line]+.005, df.cka[line]+.001,
#      df.model[line], horizontalalignment='right',
#      size='small', color='black')
#
# plt.title('Forward Encoding Performance vs CKA Similarity Score')
# # Set x-axis label
# plt.xlabel('Forward Encoding Performance Across ROIs')
# # Set y-axis label
# plt.ylabel('CKA Score Across ROIs')
# p1.figure.savefig('fe_vs_cka.png')
# plt.show()
