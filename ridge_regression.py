# python life.py | tee logs/other_$(date +"%F-T%H%M%S").log
# also try single_alpha=False

import matplotlib; matplotlib.use('agg')

import mvpa2.suite as mv
import numpy as np
import pandas as pd
from ridge import bootstrap_ridge
import sys, os, time, csv, gzip

VIDEO_DATA_DIR = '/ihome/cara/thesis/processed_raiders_videos/'
RESPONSE_DATA_DIR = '/idata/DBIC/fma/id_studies/home/preprocess/final/raiders-32ch/'
HYPERALIGN_DIR = '/idata/DBIC/fma/id_studies/home/preprocess/aligned/raiders-32ch/'
RIDGE_RESULTS_DIR = '/ihome/cara/thesis/ridge_results/'

SUBJECTS = ['sub-rid000005','sub-rid000011','sub-rid000014','sub-rid000015','sub-rid000020',\
            'sub-rid000028','sub-rid000029','sub-rid000033','sub-rid000038','sub-rid000042', 'sub-rid000043']
HEMIS = ['lh', 'rh']
RUNS = range(1,9)

tr_movie = {1:369, 2:341, 3:372, 4:406}
tr_fmri = {1:374, 2:346, 3:377, 4:412}
TR = 2.5
N_VERTICES = 40962
N_MEDIAL = {'lh': 3486, 'rh': 3491}
HYPERALIGN_MOVIE_HALF = {1: '5-6-7-8', 2: '5-6-7-8', 3: '5-6-7-8', 4: '5-6-7-8', \
                         5: '1-2-3-4', 6: '1-2-3-4', 7: '1-2-3-4', 8: '1-2-3-4', }
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
# here are extracted CNN activations    0 in response to video
def get_stim(test_runs):
    train_runs = RUNS
    train_runs.remove(test_run)

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

def load_and_normalize_resp(subj, run, hemi):
    resp = np.load(os.path.join(RESPONSE_DATA_DIR, '{0}/{0}_task-raiders_run-0{1}_{2}_freq32.npy'.format(subj, run, hemi)))

    # unzip hyperalignmemt mapper
    filename = '{0}_{1}_qhyper-to-raiders-8ch_ico32_z_r20.0_sl-avg_reflection_non-scaling_non-norm-row_runs1-2-3-4.hdf5.gz'.format(subj, hemi, HYPERALIGN_MOVIE_HALF[run])
    with gzip.open(filename, 'rb') as f:
        mapper = mv.h5load(os.path.join(HYPERALIGN_DIR, f))

    mv.zscore(resp, chunks_attr=None)
    test_resp = mapper.forward(resp)
    resp = resp[:,cortical_vertices[hemi] == 1]
    mv.zscore(resp, chunks_attr=None)

    return resp

# get the fMRI data
def get_resp(test_subj, test_run, hemi):
    train_runs = RUNS
    train_runs.remove(test_run)

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
hemi = sys.argv[2]

print('Getting video activations.')
train_stim, test_stim = get_stim(test_run)

for test_subj in SUBJECTS:
    print('Getting fMRI data for test subj {0}.'.format(test_subj))
    train_resp, test_resp = get_resp(test_subj, test_run, hemi)

    alphas = np.logspace(0, 3, 20)
    nruns = len(RUNS) - 1
    chunklen = 15
    nchunks = 15

    print('Training bootstrap ridge regression.')
    wt, corrs, alphas, _, _ = bootstrap_ridge(train_stim, train_resp, test_stim, test_resp, alphas, nruns, single_alpha=True, return_wt=False)

    print('\nFinished training ridge regression, writing to file.')
    directory = os.path.join(RIDGE_RESULTS_DIR, '{0}/{1}/{2}/'.format(hemi, test_run, test_subj))
    if not os.path.exists(directory):
        os.makedirs(directory)

    # no weights to save bc return_wt=False
    # save the corrs as npy and niml.dset
    np.save(os.path.join(directory, 'corrs.npy', corrs)
    out = get_full_surface(corrs)
    mv.niml.write(os.path.join(directory, 'corrs.{0}.niml.dset'.format(hemi)), out[None,:])

    # save the alphas
    np.save(os.path.join(directory, 'alphas.npy'), alphas)

    print('\nFinished writing corrs and alphas to {0}.'.format(directory))
print('All done!')
