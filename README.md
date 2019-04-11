# thesis

Code for Cara Van Uden's senior honors thesis in computer science titled "Modeling scene, object, and action representations learned by R(2+1)D CNN and the brain."

TODO:
1. Scale and crop Raiders video clips to 112x112 clips needed by R(2+1)D model (incorporate eye tracking for crop to get salient crop?)
2. Get singularity/docker working for installing dependencies (opencv, ffmpeg, pytorch) needed by model
3. Get model trained on SOA dataset, both on full dataset and on S,O,A individually (do steps 5-8 for each of these 4 models)
4. Figure out correct striding, input length, etc for input video clips and feed into model
5. Extract activations for each layer of model based on these Raiders clips
6. Use each of these activations to predict (ridge regression) fMRI BOLD response of hyperaligned Raiders data
7. Assign voxels to layers based on layer-wise prediction accuracy
8. Make figures
