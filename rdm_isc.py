# RDMs for RSA analysis
# want two sets of RDMs:
#   subject-wise for hyperalignment sanity check comparing before and after hyperalignment- average across time points
#   timepoint-wise for comparing DNN and brain - average across subjects
#   voxel-wise for comparing DNN and brain - average across subjects
# so have data per subject per voxel per timepoint - how to average?
# CVU 2019

from scipy.spatial.distance import pdist
import sys
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

DATA_DIR = sys.argv[1]
DATASET = sys.argv[2]
ANAT_DATA_DIR = '/idata/DBIC/fma/id_studies/home/preprocess/aligned/raiders-32ch/'
HYPER_DATA_DIR = '/ihome/cara/cvu_thesis/hyper_raiders_brain_data/'
FIGURE_DIR = '/ihome/cara/cvu_thesis/figures/'

# subjects = ['sub-rid000001', 'sub-rid000008']

subjects = ['sub-rid000001', 'sub-rid000008', 'sub-rid000009', 'sub-rid000012', 'sub-rid000013', \
            'sub-rid000016', 'sub-rid000017', 'sub-rid000018', 'sub-rid000019', 'sub-rid000021', \
            'sub-rid000022', 'sub-rid000024', 'sub-rid000025', 'sub-rid000026', 'sub-rid000027', \
            'sub-rid000031', 'sub-rid000032', 'sub-rid000036', 'sub-rid000037', 'sub-rid000041']
hemispheres = ['lh', 'rh']
runs = ['1-2-3-4', '5-6-7-8']

isc_data = {}
for subj in subjects:
    run_dat = []
    for run in runs:
        hemi_dat = []
        for hemi in hemispheres:
            print(subj, run, hemi)
            # load numpy array
            dat = np.load(DATA_DIR + '{0}_{1}_runs{2}_hyperalign.npy'.format(subj, hemi, run))
            hemi_dat.append(dat)
        whole_brain_run = np.concatenate(hemi_dat, axis=1)
        run_dat.append(whole_brain_run)
    isc_data[subj] = np.mean(np.concatenate(run_dat, axis=0),axis=0)

# for isc
brain_data = np.stack(list(isc_data.values()),axis=0)
print(brain_data.shape)
C_mat = np.corrcoef(brain_data)
# C_mat = pdist(brain_data, metric='correlation')
print(C_mat.shape)
plt.figure(figsize = (15,15))
corr_heatmap = sns.heatmap(C_mat, vmax = .8, square = True)
corr_heatmap.figure.savefig(FIGURE_DIR + 'isc_{0}.png'.format(DATASET))

# for timepoint rdm
