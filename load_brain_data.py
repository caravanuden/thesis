import os
import numpy as np
from mvpa2.suite import h5load

DATA_DIR = '/idata/DBIC/fma/id_studies/home/preprocess/aligned/raiders-32ch/'
NP_DATA_DIR = '/ihome/cara/cvu_thesis/hyper_raiders_brain_data/'

subjects = ['sub-rid000001', 'sub-rid000008', 'sub-rid000009', 'sub-rid000012', 'sub-rid000013', \
            'sub-rid000016', 'sub-rid000017', 'sub-rid000018', 'sub-rid000019', 'sub-rid000021', \
            'sub-rid000022', 'sub-rid000024', 'sub-rid000025', 'sub-rid000026', 'sub-rid000027', \
            'sub-rid000031', 'sub-rid000032', 'sub-rid000036', 'sub-rid000037', 'sub-rid000041']
hemispheres = ['lh', 'rh']
runs = ['1-2-3-4', '5-6-7-8']

for subj in subjects:
    for hemi in hemispheres:
        for run in runs:
            print('{0}_{1}_runs{2}_hyperalign'.format(subj, hemi, run))
            filename = DATA_DIR + '{0}_{1}_qhyper-to-raiders-8ch_ico32_z_r20.0_sl-avg_reflection_scaling_non-norm-row_runs{2}.hdf5.gz'.format(subj, hemi, run)
            # get pymvpa dataset
            ds = h5load(filename)
            # save numpy array
            np.save(NP_DATA_DIR + '{0}_{1}_runs{2}_hyperalign.npy'.format(subj, hemi, run), ds.samples)
