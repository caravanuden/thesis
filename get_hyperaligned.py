import numpy as np
import matplotlib; matplotlib.use('agg')
import mvpa2.suite as mv
from mvpa2.base.hdf5 import h5load

SUBJECTS = ['sub-rid000001', 'sub-rid000008', 'sub-rid000009', 'sub-rid000012', 'sub-rid000013', \
            'sub-rid000016', 'sub-rid000017', 'sub-rid000018', 'sub-rid000019', 'sub-rid000021', \
            'sub-rid000022', 'sub-rid000024', 'sub-rid000025', 'sub-rid000026', 'sub-rid000027', \
            'sub-rid000031', 'sub-rid000032', 'sub-rid000036', 'sub-rid000037', 'sub-rid000041']

RUNS = ['1-2-3-4', '5-6-7-8']
HEMIS = ['lh', 'rh']

LOAD_DIR = '/idata/DBIC/fma/id_studies/home/preprocess/aligned/raiders-32ch/'
SAVE_DIR = '/idata/DBIC/cara/thesis/hyper_raiders/'
for subj in SUBJECTS:
    for run in RUNS:
        for hemi in HEMIS:
            print(subj, hemi, run)
            data = h5load(LOAD_DIR + '{0}_{1}_qhyper-to-raiders-8ch_ico32_z_r20.0_sl-avg_reflection_non-scaling_non-norm-row_runs{2}.hdf5.gz'.format(subj, hemi, run)).samples
            np.save(SAVE_DIR + '{0}_runs-{1}_{2}.npy'.format(subj, run, hemi), data)
