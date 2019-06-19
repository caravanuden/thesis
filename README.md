# thesis

Code for Cara Van Uden's senior honors thesis in computer science titled "Comparing brain-like representations learned by vanilla, residual, and recurrent CNN architectures."

Though it has been hypothesized that state-of-the art residual networks approximate the recurrent visual system, it is yet to be seen if the representations learned by these ”biologically inspired” CNNs actually have closer representations to neural data. It is likely that CNNs and DNNs that are most functionally similar to the brain will contain mechanisms that are most like those used by the brain. In this thesis, we investigate how different CNN architectures approximate the representations learned through the ventral—object recognition and processing—stream of the brain. We specifically evaluate how recent approximations of biological neural recurrence—such as residual connections, dense residual connections, and a biologically-inspired implemen- tation of recurrence—affect the representations learned by each CNN. We first investigate the representations learned by layers throughout a few state-of-the-art CNNs—VGG-19 (vanilla CNN), ResNet-152 (CNN with res!
 idual connections), and DenseNet-161 (CNN with dense connections). To control for differences in model depth, we then extend this analysis to the CORnet family of biologically-inspired CNN models with matching high-level architectures. The CORnet family has three models: a vanilla CNN (CORnet-Z), a CNN with biologically-valid recurrent dynamics (CORnet-R), and a CNN with both recurrent and residual connections (CORnet-S).

We compare the representations of these six models to functionally aligned (with hyperalignment) fMRI brain data acquired during a naturalistic visual task. We take two approaches to comparing these CNN and brain representations. We first use forward encoding, a predictive approach that uses CNN features to predict neural responses across the whole brain. We next use representational similarity analysis (RSA) and centered kernel alignment (CKA) to measure the similarities in representation within CNN layers and specific brain ROIs. We show that, compared to vanilla CNNs, CNNs with residual and recurrent connections exhibit representations that are even more similar to those learned by the human ventral visual stream. We also achieve state-of-the-art forward encoding and RSA performance with the residual and recurrent CNN models.

### TODONE:
- Get anat and hyperaligned Raiders responses (work with surface just like Life) and get info about the dataset
- Get singularity/docker working for installing dependencies (opencv, ffmpeg, pytorch) needed by model

- Scale and crop Raiders video clips to 224x224 clips needed by ImageNet models
- Get images from video (every half sec)
- Get/choose correct activation layers for CORnet-{Z,R,S}, ResNet, DenseNet, VGG (pretrained on ImageNet)
- Get image activations for each layer for each model

- Compare layers within-model for each model
- Compare ROIs within brain
- Use the extracted activations from models to predict (ridge regression) fMRI BOLD response of hyperaligned Raiders data
- Get ROI-wise pred acc
- Use the extracted activations from models to make CKA RDMs with ROIs from fMRI BOLD response of hyperaligned Raiders data
- Assign voxels to layers based on layer-wise prediction accuracy and RDM similarity
- Assign ROIs to layers based on layer-wise prediction accuracy and RDM similarity (Jaccard similarity)

### TODO:
- Incorporate eye tracking (congruency map across subj) for cropping images to get salient crop?
- ?
