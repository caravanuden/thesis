# python life.py | tee logs/other_$(date +"%F-T%H%M%S").log
# also try single_alpha=False

import matplotlib; matplotlib.use('agg')

import mvpa2.suite as mv
import numpy as np
import pandas as pd
from ridge import bootstrap_ridge
import sys, os, time, csv

VIDEO_DATA_DIR = '/ihome/cara/thesis/processed_raiders_videos/'
RESPONSE_DATA_DIR = '/idata/DBIC/fma/id_studies/home/preprocess/final/raiders-32ch/'
HYPERALIGNMENT_DIR = '/idata/DBIC/fma/id_studies/home/preprocess/aligned/raiders-32ch/'

SUBJECTS = ['sub-rid000005','sub-rid000011','sub-rid000014','sub-rid000015','sub-rid000020',\
            'sub-rid000028','sub-rid000029','sub-rid000033','sub-rid000038','sub-rid000042', 'sub-rid000043']
HEMIS = ['lh', 'rh']

tr_movie = {1:369, 2:341, 3:372, 4:406}
tr_fmri = {1:374, 2:346, 3:377, 4:412}
TR = 2.5
N_VERTICES = 40962
N_MEDIAL = {'lh': 3486, 'rh': 3491}

cortical_vertices = create_cortical_vertex_mask()

# /idata/DBIC/fma/id_studies/home/preprocess/final/raiders-32ch/sub-rid000001/sub-rid000001_task-raiders_run-01_lh_freq32.npy
# create mask of cortical vertices excluding medial wall
def create_cortical_vertex_mask():
    cortical_vertices = {}
    for half in HEMIS:
        test_ds = mv.niml.read('/dartfs-hpc/scratch/cara/models/niml/ws/ws_run1_singlealpha.{0}.niml.dset'.format(half))
        cortical_vertices[half] = np.ones((N_VERTICES))
        cortical_vertices[half][np.sum(test_ds.samples[1:, :] != 0, axis=0) == 0] = 0
    return cortical_vertices

# include medial wall again for saving to niml dset
def get_full_surface(npy_data):
    med_wall_ind = np.where(cortical_vertices[hemi] == 0)[0]
    out = np.zeros((npy_data.shape[0]+med_wall_ind.shape[0]),dtype=npy_data.dtype)
    out[cortical_vertices[hemi] == 1] = npy_data

    return out

# get ridge regression inputs
# here are extracted CNN activations in response to video
def get_stim(train_runs, test_run):
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

def load_and_normalize_resp(subj_mapper, subj, run, hemi):
    resp = np.load(os.path.join(RESPONSE_DATA_DIR, '{0}/{0}_task-raiders_run-0{1}_{2}_freq32.npy'.format(subj, run, hemi)))

    mv.zscore(resp, chunks_attr=None)
    test_resp = subj_mapper.forward(resp)
    resp = resp[:,cortical_vertices[hemi] == 1]
    mv.zscore(resp, chunks_attr=None)

    return resp

# get the fMRI data
def get_resp(test_subj, mappers, train_runs, test_run, hemi):
    train_subj = SUBJECTS[:].remove(test_subj)
    train_resp = []

    for run in train_runs:
        avg = []
        for subj in train_subj:
            resp = load_and_normalize_resp(mappers[subj], subj, run, hemi)
            avg.append(resp)

        avg = np.mean(avg, axis=0)
        mv.zscore(avg, chunks_attr=None)
        print('Shape of resp for run {0}: {1}'.format(run, avg.shape))
        train_resp.append(avg)

    test_resp = load_and_normalize_resp(mappers[test_subj], test_subj, test_run, hemi)

    print('Shape of resp for run {0}: {1}'.format(test_run, test_resp.shape))

    return train_resp, test_resp

test_run = sys.argv[1]
train_runs = range(1,9)
train_runs.remove(test_run)

hemi = sys.argv[2]

train_stim, test_stim = get_stim(test_run, train_runs)
print('\nLoading hyperaligned mappers...')

# sub-rid000021_rh_qhyper-to-raiders-8ch_ico32_z_r20.0_sl-avg_reflection_non-scaling_non-norm-row_runs5-6-7-8.hdf5.gz
mappers = mv.h5load(os.path.join(HYPERALIGNMENT_DIR, 'search_hyper_mappers_life_mask_nofsel_{0}_leftout_{1}.hdf5'.format(hemi, test_run)))

for test_subj in SUBJECTS:
    train_resp, test_resp = get_resp(test_subj, mappers, test_run, train_runs, hemi)

    alphas = np.logspace(0, 3, 20)
    nboots = len(train_runs)
    chunklen = 15
    nchunks = 15

    wt, corrs, alphas, bootstrap_corrs, valinds = bootstrap_ridge(train_stim, train_resp, test_stim, test_resp, alphas, nboots, single_alpha=True, return_wt=False)

    print('\nFinished training ridge regression, writing to file...')
    directory = os.path.join('/dartfs-hpc/scratch/cara/models', '{0}/{1}/{2}/leftout_run_{3}'.format(align, model, stimfile, test_run), hemi, test_subj)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # no weights to save bc return_wt=False
    # save the corrs
    np.save(os.path.join(directory, 'corrs.npy', corrs)
    out = get_full_surface(corrs)
    mv.niml.write(os.path.join(directory, 'corrs.{0}.niml.dset'.format(hemi)), out[None,:])

    # save the alphas, don't need the bootstrap corrs and valinds
    np.save(os.path.join(directory, 'alphas.npy'), alphas)

    print('\nFinished writing to {0}.'.format(directory))
print('All done!')
