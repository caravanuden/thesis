# python life.py | tee logs/other_$(date +"%F-T%H%M%S").log
# also try single_alpha=False

import matplotlib; matplotlib.use('agg')

import mvpa2.suite as mv
import numpy as np
import pandas as pd
from ridge import ridge
import sys, os, time, csv

mvpa_dir = '/idata/DBIC/cara/life/pymvpa/'
sam_data_dir = '/idata/DBIC/snastase/life'
ridge_dir = '/idata/DBIC/cara/life/ridge'
cara_data_dir = '/idata/DBIC/cara/life/data'
npy_dir = '/idata/DBIC/cara/w2v/w2v_features'

subjects = ['sub-rid000001','sub-rid000005','sub-rid000006','sub-rid000009','sub-rid000012',\
                'sub-rid000014','sub-rid000017','sub-rid000019','sub-rid000024','sub-rid000027',\
                'sub-rid000031','sub-rid000032','sub-rid000033','sub-rid000034','sub-rid000036',\
                'sub-rid000037','sub-rid000038','sub-rid000041']

hemispheres = ['lh', 'rh']

tr_movie = {1:369, 2:341, 3:372, 4:406}
tr_fmri = {1:374, 2:346, 3:377, 4:412}
tr_length = 2.5
n_samples = 1509
n_vertices = 40962
n_proc = 32     # how many cores do we have?
n_medial = {'lh': 3486, 'rh': 3491}

# get ridge regression inputs
# here are extracted CNN activations in response to video
def get_stim(input_file, train_runs, test_run):
    # TODO: load CNN activations as np array

    stim = []
    idx = 0

    # loop thru all movie parts, get activations, and incorporate hemodynamic lag
    for run in movie_trs.keys():
        idx += movie_trs[movie_part]
        movie_activations = activations[:idx,:]

        # incorporate hemodynamic delay/lag
        lagged_activations = np.concatenate((movie_activations[3:,:], movie_activations[2:-1,:], movie_activations[1:-2,:], movie_activations[:-3,:]), axis=1)
        stim.append(lagged_activations)
        print('Shape of stim for run {0}: {1}'.format(run, lagged_activations.shape))

        # split into train and test
    train_stim = [stim[i] for i in np.subtract(train_runs, 1)]
    test_stim = stim[test_run-1]

    return train_stim, test_stim

# gets the fMRI data
def get_resp(test_subj, mappers, train_runs, test_run, hemi):
    train_p = subjects[:].remove(test_subj)

    print('\nLoading fMRI GIFTI data in hyperaligned common space, using {0} as test participant...'.format(test_subj))
    train_resp = []
    for run in train_runs:
        avg = []
        for participant in train_p:
            if run == 4:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr_fmri[run], run, hemi))).samples[4:-5,:]
            else:
                resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(participant, tr_fmri[run], run, hemi))).samples[4:-4,:]

            mv.zscore(resp, chunks_attr=None)
            resp = mappers[participant].forward(resp)
            resp = resp[:,cortical_vertices[hemi] == 1]
            mv.zscore(resp, chunks_attr=None)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('Shape of resp for run {0}: {1}'.format(run, avg.shape))
        train_resp.append(avg)

    if test_run == 4:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_subj, tr_fmri[test_run], test_run, hemi))).samples[4:-5,:]
    else:
        test_resp = mv.gifti_dataset(os.path.join(sam_data_dir, '{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format(test_subj, tr_fmri[test_run], test_run, hemi))).samples[4:-4,:]

    mv.zscore(test_resp, chunks_attr=None)
    test_resp = mappers[participant].forward(test_resp)
    test_resp = test_resp[:,cortical_vertices[hemi] == 1]
    mv.zscore(test_resp, chunks_attr=None)

    print('Shape of resp for run {0}: {1}'.format(test_run, test_resp.shape))

    return train_resp, test_resp

model = sys.argv[1]
align = sys.argv[2]
stimfile = sys.argv[3]
fold = int(sys.argv[4])
test_run = fold+1

hemi = sys.argv[5]

train_runs = [1,2,3,4]
train_runs.remove(test_run)

# First let's create mask of cortical vertices excluding medial wall
cortical_vertices = {}
for half in ['lh', 'rh']:
    test_ds = mv.niml.read('/dartfs-hpc/scratch/cara/models/niml/ws/ws_run1_singlealpha.{0}.niml.dset'.format(half))
    cortical_vertices[half] = np.ones((n_vertices))
    cortical_vertices[half][np.sum(test_ds.samples[1:, :] != 0, axis=0) == 0] = 0

print('Model: {0}\nStim file: {1}\nHemi: {2}\nRuns in training: {3}\nRun in test: {4}\n'.format(model, stimfile, hemi, train_runs, test_run))

train_stim, test_stim = get_stim('{0}_{1}'.format(model, stimfile), test_run, train_runs)
print('\nLoading hyperaligned mappers...')
mappers = mv.h5load(os.path.join(mvpa_dir, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}.hdf5'.format(hemi, test_run)))

for test_subj in subjects:
    train_resp, test_resp = get_resp(test_subj, mappers, test_run, train_runs, hemi)

    alphas = np.logspace(0, 3, 20)
    nboots = len(train_runs)
    chunklen = 15
    nchunks = 15

    wt, corrs, alphas, bootstrap_corrs, valinds = ridge.bootstrap_ridge(train_stim, train_resp, test_stim, test_resp, alphas, nboots, single_alpha=True, return_wt=False)

    print('\nFinished training ridge regression, writing to file...')
    directory = os.path.join('/dartfs-hpc/scratch/cara/models', '{0}/{1}/{2}/leftout_run_{3}'.format(align, model, stimfile, test_run), hemi, test_subj)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # no weights to save bc return_wt=False
    # save the corrs
    np.save(os.path.join(directory, 'corrs.npy', corrs)
    med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
    out = np.zeros((corrs.shape[0]+med_wall_ind.shape[0]),dtype= corrs.dtype)
    out[cortical_vertices[hemi] == 1] = corrs
    mv.niml.write(os.path.join(directory, 'corrs.{0}.niml.dset'.format(hemi)), out[None,:])

    # save the alphas, bootstrap corrs, and valinds
    np.save(os.path.join(directory, 'alphas.npy'), alphas)
    np.save(os.path.join(directory, 'bootstrap_corrs.npy'), bootstrap_corrs)
    np.save(os.path.join(directory, 'valinds.npy'), valinds)

    print('\nFinished writing to {0}.'.format(directory))
print('All done!')
