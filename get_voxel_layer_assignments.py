import numpy as np
import matplotlib; matplotlib.use('agg')
import mvpa2.suite as mv

for hemi in ['lh', 'rh']:
    roi = np.load('maxprob_fsaverage6_{0}.npy'.format(hemi))
    VT = mv.niml.read('{0}.mask_VT.niml.dset'.format(hemi)).samples
    print(sum(np.where(VT[0,:] == 1)))
    roi[np.where(VT[0,:] == 1)] = 26
    np.save('roi_mask_{0}.npy'.format(hemi), roi)
# roi_mask = np.concatenate([lh_roi, rh_roi])
#
# brain_data = load_brain_data(align)
# # avg_brain_data = brain_data[1,:,:]
# avg_brain_data = np.mean(brain_data, axis=0)
# np.save('avg_brain_data_{0}.npy'.format(align), avg_brain_data)
#
# # avg_brain_data = np.load('avg_brain_data.npy')
# print(avg_brain_data.shape)
# roi_brain_data = []
#
# for roi in roi_names:
#     print(roi)
#     dat = avg_brain_data[:,np.where(np.isin(roi_mask, rois[roi]))]
#     dat = np.reshape(dat, [dat.shape[0], dat.shape[2]])
#     print(dat.shape)
#     roi_brain_data.append(dat)

DATA_DIR = '/ihome/cara/cvu_thesis/thesis/ridge_results/'

for hemi in ['lh', 'rh']:
    for model in ['cornet-z', 'cornet-r', 'cornet-s']:
        rois = []
        for roi in ['V1', 'V2', 'V4', 'IT']:
            data = mv.niml.read(DATA_DIR + '{0}/{1}_avg_corrs.{2}.niml.dset'.format(model, roi, hemi))
            print(data.shape)
            data = np.mean(data, axis=0)
            print(data.shape)
            rois.append(data)
        rois = np.stack(rois)
        maxes = np.argmax(rois, axis=0)
        print(maxes.shape)
        mv.niml.write('/ihome/cara/{0}_maxes.{1}.niml.dset'.format(model, hemi), maxes)
