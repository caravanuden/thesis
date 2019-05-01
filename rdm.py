# RDMs for RSA analysis
# want two sets of RDMs:
#   subject-wise for hyperalignment sanity check comparing before and after hyperalignment
#   timepoint-wise for comparing DNN and brain
# so have data per subject per voxel per timepoint - how to average?
# CVU 2019

from scipy.spatial.distance import pdist
