# python life.py | tee logs/other_$(date +"%F-T%H%M%S").log
# also try single_alpha=False

import seaborn as sns
# os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# import mvpa2.suite as mv
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.decomposition import KernelPCA

from ridge import bootstrap_ridge
from scipy.stats import gamma, zscore

import sys, os, time, csv, gzip



VIDEO_DATA_DIR = '/ihome/cara/cvu_thesis/processed_raiders_videos/'
RESPONSE_DATA_DIR = '/ihome/cara/cvu_thesis/hyper_raiders_brain_data/'
HYPERALIGN_DIR = '/idata/DBIC/fma/id_studies/home/preprocess/aligned/raiders-32ch/'
MODEL_DATA_DIR = '/Users/caravanuden/thesis/image_activations/'

REDUCED_MODEL_DATA_DIR = '/idata/DBIC/cara/thesis/reduced_activations/'
RIDGE_RESULTS_DIR = '/ihome/cara/cvu_thesis/ridge_results/'

NP_DATA_DIR = '/ihome/cara/cvu_thesis/hyperaligned_raiders_brain_data'

sam_data_dir = '/idata/DBIC/snastase/life'
mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'

# SUBJECTS = ['sub-rid000001', 'sub-rid000008', 'sub-rid000009', 'sub-rid000012', 'sub-rid000013', \
#             'sub-rid000016', 'sub-rid000017', 'sub-rid000018', 'sub-rid000019', 'sub-rid000021', \
#             'sub-rid000022', 'sub-rid000024', 'sub-rid000025', 'sub-rid000026', 'sub-rid000027', \
#             'sub-rid000031', 'sub-rid000032', 'sub-rid000036', 'sub-rid000037', 'sub-rid000041']

SUBJECTS = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
            'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
            'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
            'sub-rid000037','sub-rid000038','sub-rid000041']
HEMIS = ['lh', 'rh']

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
RUNS_HALF = 683
HEMIS = ['lh', 'rh']
tr_movie = {1:369, 2:341, 3:372, 4:406}
tr_fmri = {1:374, 2:346, 3:377, 4:412}
TR = 2.5
N_VERTICES = 40962
N_MEDIAL = {'lh': 3486, 'rh': 3491}
# RUNS = {1: '5-6-7-8', 2: '5-6-7-8', 3: '5-6-7-8', 4: '5-6-7-8', \
#         5: '1-2-3-4', 6: '1-2-3-4', 7: '1-2-3-4', 8: '1-2-3-4', }
#
# RUNS = OrderedDict([(1,374), (2,346), (3,377), (4,412)])
# # RUNS = OrderedDict([(1,374), (2,346), (3,377)])
# RUNS = OrderedDict([(1,370), (2,342), (3,373)])
#


SUBJECTS = ['sub-rid000001', 'sub-rid000008', 'sub-rid000009', 'sub-rid000012', 'sub-rid000013', \
            'sub-rid000016', 'sub-rid000017', 'sub-rid000018', 'sub-rid000019', 'sub-rid000021', \
            'sub-rid000022', 'sub-rid000024', 'sub-rid000025', 'sub-rid000026', 'sub-rid000027', \
            'sub-rid000031', 'sub-rid000032', 'sub-rid000036', 'sub-rid000037', 'sub-rid000041']

# RUNS = ['1-2-3-4', '5-6-7-8']
HEMIS = ['lh', 'rh']

LOAD_DIR = '/idata/DBIC/fma/id_studies/home/preprocess/aligned/raiders-32ch/'
SAVE_DIR = '/idata/DBIC/cara/thesis/hyper_raiders/'

def process_runs(resp, remove, z):
    print('resp shape:', resp.shape)
    ind = 0
    data = []
    for run in RUNS_LEN.keys():
        run_data = resp[ind:ind+RUNS_LEN[run]-8,:]
        if remove and run == 5:
            run_data = run_data[:-1,:]
        if z:
            run_data = zscore(run_data, axis=0)
        data.append(run_data)
        # print(run, run_data.shape)

        ind += RUNS_LEN[run]-8

    return data

# get the fMRI data
def get_resp(test_run, hemi):
    run_data = []
    for subj in SUBJECTS:
        print(subj, hemi)
        curr_resp = np.load(SAVE_DIR + '{0}_runs-5-6-7-8_{1}.npy'.format(subj, hemi))
        resp_run_list = process_runs(curr_resp, remove=True, z=True)
        run_data.append(np.concatenate(resp_run_list))

    print(np.stack(run_data).shape)
    run_data = np.mean(np.stack(run_data),axis=0)
    print(run_data.shape)
    # data_list = process_runs(run_data, remove=False, z=False)
    data_list = []
    data_list.append(run_data[:331,:])
    data_list.append(run_data[331:667,:])
    data_list.append(run_data[667:1003,:])
    data_list.append(run_data[1003:,:])

    train = []
    for i in range(len(data_list)):
        if i != int(test_run)-5:
            train.append(data_list[i])
        else:
            test = data_list[i]

    return train, test

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

def load_layer_data(model_name, layer_names, test_run):
    PATH = '/idata/DBIC/cara/thesis/image_activations/'+ model_name + '/'
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
                    averaged.append(np.linalg.norm(flat[slice:slice+N,:],axis=0))
                    # averaged.append(np.mean(flat[slice:slice+N,:],axis=0))

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
            layer_data = np.concatenate(curr_layer,axis=0)
            print(layer_data.shape)
            np.save(reduced_data_dir + '/processed_{0}.npy'.format(layer), layer_data)

        else:
            layer_data = np.load(reduced_data_dir + '/processed_{0}.npy'.format(layer))
            print(layer_data.shape)

        layer_data = layer_data[:, ~np.isnan(layer_data).any(axis=0)]

        selector = KernelPCA()
        reduced_layer = selector.fit_transform(layer_data)
        np.save(reduced_data_dir + '/reduced_{0}.npy'.format(layer), reduced_layer)
        print('Reduced layer shape for layer {0}: {1}'.format(layer, reduced_layer.shape))
        layers.append(reduced_layer)

    layers = np.concatenate(layers, axis=1)
    print('Reduced layer stim shape: {1}'.format(test_run, layers.shape))
    data_list = []
    data_list.append(layers[:331,:])
    data_list.append(layers[331:667,:])
    data_list.append(layers[667:1003,:])
    data_list.append(layers[1003:,:])

    train = []
    for i in range(len(data_list)):
        print(data_list[i].shape)
        if i != int(test_run)-5:
            train.append(data_list[i])
        else:
            test = data_list[i]

    return train, test


def run_ridge(train_stim, train_resp, test_stim, test_resp, alphas, nruns, prefix):
    print('Training bootstrap ridge regression.')
    print([s.shape for s in train_stim])
    print([s.shape for s in train_resp])
    print(test_stim.shape, test_resp.shape)
    wt, corrs, alphas, _, _ = bootstrap_ridge(train_stim, train_resp, test_stim, test_resp, alphas, nruns, single_alpha=True, return_wt=False)

    print('Finished training ridge regression, writing to file.')

    # no weights to save bc return_wt=False
    # save the corrs as npy and niml.dset
    np.save(prefix + 'corrs.npy', corrs)
    print(np.min(corrs), np.max(corrs))
    # out = get_full_surface(corrs)
    # mv.niml.write(prefix + 'corrs.{0}.niml.dset'.format(hemi), corrs[None,:])

    # save the alphas
    np.save(prefix + 'alphas.npy', alphas)

    print('\nFinished writing corrs and alphas to {0}'.format(prefix))
    return corrs

def forward_encoding(run, hemi, curr_model_name, curr_layer_names):
    print('Getting image/video activations.')
    # train_stim, test_stim = get_stim(run)
    train_stim, test_stim = load_layer_data(curr_model_name, curr_layer_names, run)

    alphas = np.logspace(0, 3, 20)
    nruns = len(RUNS_LEN.keys()) - 1
    chunklen = 15
    nchunks = 15

    train_resp, test_resp = get_resp(run, hemi)
    # train_resp, test_resp = load_brain_data(run, 'hyper')
    directory = os.path.join(RIDGE_RESULTS_DIR, '{0}/{0}_all_run-{1}_{2}_'.format(curr_model_name, run, hemi))
    corrs = run_ridge(train_stim, train_resp, test_stim, test_resp, alphas, nruns, directory)


    for layer in curr_layer_names:
        print('Getting image/video activations for layer {0}'.format(layer))
        train_stim, test_stim = load_layer_data(curr_model_name, [layer], run)
        directory = os.path.join(RIDGE_RESULTS_DIR, '{0}/{0}_{1}_run-{2}_{3}_'.format(curr_model_name, layer, run, hemi))
        corrs = run_ridge(train_stim, train_resp, test_stim, test_resp, alphas, nruns, directory)

def plot_map():
    for model_name in ['cornet-z', 'cornet-r', 'cornet-s', 'vgg', 'resnet', 'densenet']:
        import vtk_utils
        import numpy as np

        vmin= -0.3
        vmax = 0.7
        lh = np.load('ridge_results/{0}_all_avg_lh_corrs.npy'.format(model_name))
        rh = np.load('ridge_results/{0}_all_avg_rh_corrs.npy'.format(model_name))

        # lh[np.where(lh==np.max(lh))] = 500
        # rh[np.where(rh==np.max(rh))] = 500

        vtk_utils.plot(values_li=[lh, rh], vmin=vmin, vmax=vmax,out_fn='ridge_results/figures/find_best_{0}_ridge.png'.format(model_name), cmap='coolwarm')
    #
    # vmin= 0
    # vmax = 3
    # lh = np.load('ridge_results/{0}_voxel_assign_lh.npy'.format(model_name))
    # print(lh)
    # rh = np.load('ridge_results/{0}_voxel_assign_rh.npy'.format(model_name))
    # print(rh)
    # print(lh.shape, rh.shape)
    # print(len([lh, rh]))
    # vtk_utils.plot(values_li=[lh, rh], vmin=vmin, vmax=vmax,out_fn='ridge_results/figures/{0}_voxel_assign.png'.format(model_name), cmap='accent')


for model_name in ['cornet-z', 'cornet-r', 'cornet-s', 'vgg', 'resnet', 'densenet']:
# model_name = sys.argv[1]

    for hemi in HEMIS:
        corrs_list = []
        for run in RUNS_LEN.keys():
            print('Model: {2}, run: {0}, hemi: {1}'.format(run, hemi, model_name))
            forward_encoding(run, hemi, model_name, layer_names)

    for hemi in HEMIS:
        for layer in ['all'] + layer_names:
        # for layer in ['all']:
            corrs_list = []
            for run in RUNS_LEN.keys():
                filename = os.path.join(RIDGE_RESULTS_DIR, '{0}/{0}_{1}_run-{2}_{3}_corrs.npy'.format(model_name, layer, run, hemi))
                curr = np.load(filename)
                print(np.min(curr), np.max(curr))
                corrs_list.append(curr)
            corrs = np.mean(np.stack(corrs_list),axis=0)
            print(np.min(corrs), np.max(corrs))
            np.save('/ihome/cara/cvu_thesis/thesis/ridge_results/{0}/{0}_{1}_avg_{2}_corrs.npy'.format(model_name, layer, hemi), corrs)
            print('Saved avg corrs for layer {0}, hemi {1}'.format(layer, hemi))

    for hemi in HEMIS:
        corrs_list = []
        for layer in layer_names:
            print('Model: {0}, hemi: {1}'.format(model_name, hemi))
            filename = '/ihome/cara/cvu_thesis/thesis/ridge_results/{0}/{0}_{1}_avg_{2}_corrs.npy'.format(model_name, layer, hemi)
            corrs_list.append(np.load(filename))
        corrs = np.stack(corrs_list)
        print(corrs.shape)
        maxes = np.argmax(corrs, axis=0)
        print(maxes.shape)
        np.save('/ihome/cara/cvu_thesis/thesis/ridge_results/{0}/{0}_voxel_assign_{1}.npy'.format(model_name, hemi), maxes)

print('done!')

# rois = OrderedDict([('V1',[1,2]), ('V2',[3,4]), ('V3',[5,6]), ('V3a',[17]), ('V3b',[16]), ('V4',[7]), ('VO',[8,9]), ('PHC',[10,11]), ('LO',[14,15]), ('MT',[13]), ('VT',[26])])
# hemispheres = OrderedDict([('lh',np.load('/ihome/cara/cvu_thesis/thesis/surface/fsaverage_lh_mask.npy')), ('rh',np.load('/ihome/cara/cvu_thesis/thesis/surface/fsaverage_rh_mask.npy'))])
# from scipy.stats import mode
# def return_rois(brain_data, roi_means, ind):
#     roi_names = list(rois.keys())
#
#     lh_roi = np.load('/ihome/cara/cvu_thesis/fsaverage6/roi_mask_lh.npy')[:10242]
#     lh_roi = lh_roi[hemispheres['lh'][:10242]]
#     rh_roi = np.load('/ihome/cara/cvu_thesis/fsaverage6/roi_mask_lh.npy')[:10242]
#     rh_roi = rh_roi[hemispheres['rh'][:10242]]
#
#     roi_mask = np.concatenate([lh_roi, rh_roi])
#     roi_brain_data = OrderedDict()
#     print(np.max(brain_data))
#     roi_corrs = []
#     # print(np.mean(brain_data), np.std(brain_data))
#     for i,roi in enumerate(roi_names):
#         dat = brain_data[np.where(np.isin(roi_mask, rois[roi])),]
#         if np.max(dat) == np.max(brain_data):
#             print(roi, np.max(dat))
#         # roi_corrs.extend(dat.tolist())
#         # roi_means[i][ind] = np.mean(dat)
#         # roi_means[i][ind] = mode(dat,axis=1).mode +1
#
#     # roi_corrs = [item for sublist in roi_corrs for item in sublist]
#     # print(np.mean(roi_corrs), np.std(roi_corrs))
#     # return roi_brain_data
#
# model_names = ['cornet-z', 'cornet-s', 'cornet-r', 'vgg', 'resnet', 'densenet']
# ind=0
# roi_means = np.zeros((9,6))
# for model_name in model_names:
#     # print('\n')
#     print(model_name)
#     lh = np.load('/ihome/cara/cvu_thesis/thesis/ridge_results/{0}/{0}_all_avg_lh_corrs.npy'.format(model_name))
#     rh = np.load('/ihome/cara/cvu_thesis/thesis/ridge_results/{0}/{0}_all_avg_rh_corrs.npy'.format(model_name))
#     # lh = np.load('/ihome/cara/cvu_thesis/thesis/ridge_results/{0}/{0}_voxel_assign_lh.npy'.format(model_name))
#     # rh = np.load('/ihome/cara/cvu_thesis/thesis/ridge_results/{0}/{0}_voxel_assign_rh.npy'.format(model_name))
#     data = np.concatenate([lh,rh])
#     # print('mean: {0}, std: {1}'.format(np.mean(data), np.std(data)))
#
#     return_rois(data, roi_means, ind)
#     ind+=1
#
# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline
#
# model_names = ['cornet-z', 'cornet-s', 'cornet-r', 'vgg', 'resnet', 'densenet']
# for model in model_names:
#     cka_results = np.load('/ihome/cara/cvu_thesis/results/cka_{0}_{0}.npy'.format(model))
#     print(cka_results.shape)
#     print(model, np.mean(cka_results[np.triu_indices(4, k = 1)]))
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
# plt.figure()
# p1 = sns.scatterplot('imagenet_top1', # Horizontal axis
#        'whole_brain_pred', # Vertical axis
#        data=df, # Data source
#        legend=False)
#
# for line in range(0,df.shape[0]):
#      p1.text(df.imagenet_top1[line]-.01, df.whole_brain_pred[line]+.001,
#      df.model[line], horizontalalignment='left',
#      size='small', color='black')
#
# plt.title('ImageNet Top-1 vs Whole-Brain Forward Encoding Performance')
# # Set x-axis label
# plt.xlabel('ImageNet Top-1')
# # Set y-axis label
# plt.ylabel('Whole-Brain Forward Encoding')
# p1.figure.savefig('imagenet_top1_vs_wholebrain.png')
# plt.show()
#
# plt.figure()
# p1 = sns.scatterplot('imagenet_top1', # Horizontal axis
#        'ROI_pred', # Vertical axis
#        data=df, # Data source
#        legend=False)
#
# for line in range(0,df.shape[0]):
#      p1.text(df.imagenet_top1[line]-.01, df.ROI_pred[line]+.001,
#      df.model[line], horizontalalignment='left',
#      size='small', color='black')
#
# plt.title('ImageNet Top-1 vs ROI Forward Encoding Performance')
# # Set x-axis label
# plt.xlabel('ImageNet Top-1')
# # Set y-axis label
# plt.ylabel('ROI Forward Encoding')
# p1.figure.savefig('imagenet_top1_vs_roi.png')
# plt.show()
#
# plt.figure()
# p1 = sns.scatterplot('layer_similarity', # Horizontal axis
#        'imagenet_top1', # Vertical axis
#        data=df, # Data source
#        legend=False)
#
# for line in range(0,df.shape[0]):
#      p1.text(df.layer_similarity[line]-.01, df.imagenet_top1[line]+.001,
#      df.model[line], horizontalalignment='left',
#      size='small', color='black')
#
# plt.title('Layer Similarity vs ImageNet-Top1 Performance')
# # Set x-axis label
# plt.xlabel('Layer Similarity')
# # Set y-axis label
# plt.ylabel('ImageNet Top-1')
# p1.figure.savefig('similarity_vs_imagenet_top1.png')
# plt.show()
#
# plt.figure()
# p1 = sns.scatterplot('layer_similarity', # Horizontal axis
#        'ROI_pred', # Vertical axis
#        data=df, # Data source
#        legend=False)
#
# for line in range(0,df.shape[0]):
#      p1.text(df.layer_similarity[line]-.01, df.ROI_pred[line]+.001,
#      df.model[line], horizontalalignment='left',
#      size='small', color='black')
#
# plt.title('Layer Similarity vs ROI Forward Encoding Performance')
# # Set x-axis label
# plt.xlabel('Layer Similarity')
# # Set y-axis label
# plt.ylabel('ROI Forward Encoding Performance')
# p1.figure.savefig('similarity_vs_roi.png')
# plt.show()


# plt.figure()
# ax = plt.axes()
# roi_names = list(rois.keys())
#
# # name axes with ROIs
# mean_heatmap = sns.heatmap(roi_means, vmin=1,cmap = 'magma', xticklabels=model_names, yticklabels=roi_names, annot=True)
# ax.set_xlabel('CNN Models')
# ax.set_ylabel('Brain ROIs')
# # ax.set_title('Average ridge regression performance by brain ROI')
# # mean_heatmap.figure.savefig('mean_ridge_roi_model.png')
# ax.set_title('Ridge regression layer assignment by brain ROI')
# mean_heatmap.figure.savefig('layer_assign_ridge_roi_model.png')
# else:
#     model_name = sys.argv[1]
#     run = sys.argv[2]
#     hemi = sys.argv[3]
#     forward_encoding(run, hemi, model_name, list(model_layers[model_name].keys()))
