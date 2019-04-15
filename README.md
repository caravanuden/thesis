# thesis

Code for Cara Van Uden's senior honors thesis in computer science titled "Scene, object, and action representations learned by R(2+1)D CNN and the brain."

TODO (CNN side):
1. Scale and crop Raiders video clips to 112x112 clips needed by R(2+1)D model (incorporate eye tracking (congruency map across subj) for crop to get salient crop?)
2. Get singularity/docker working for installing dependencies (opencv, ffmpeg, pytorch) needed by model
3. Get model (pre)trained on SOA dataset, both on full dataset and on S,O,A individually (do steps 5-8 for each of these 4 models) - need to contact LT about this and #4 (ask grad student?)
4. Figure out correct striding, input length, etc for input video clips and feed into model
5. Extract activations for each layer of model based on these Raiders clips
6. Add x,y of eye tracking for nuisance regressor (ask Andy)- other nuisance regressors?

TODO (neuro side):
1. Get hyperaligned Raiders responses (work with surface just like Life) and get info about the dataset
2. Get most relevant voxels (mask) (voxels that have most correlation with voxels w other subjects)
3. Use the extracted activations from #5 to predict (ridge regression) fMRI BOLD response of hyperaligned Raiders data

TODO (combining the two):
1. Assign voxels to layers based on layer-wise prediction accuracy
2. Compare centered vs eye-tracked clips
3. Variance accounted for by SOA, S, O, A - show that prediction localizes to dorsal stream, ventral stream, and PPA
