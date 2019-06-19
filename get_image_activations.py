# get activations from conv and fc layers of ResNet34- for image/video comparison
# CVU 2019

import sys
from PIL import Image
from glob import glob
import numpy as np
import torch
# import cornet
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms

# model_name = 'cornet'
IMG_DIR = '/Users/caravanuden/thesis/raiders/images/'
# IMG_DIR = '/Users/caravanuden/thesis/life/images/'
MODEL_DATA_DIR = '/Users/caravanuden/thesis/image_activations/'
runs = range(6,9)

# a simple hook class that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

# Once we have an img, we need to preprocess it.
# We need to:
#       * resize the img, nope resized already.
#       * normalize it, as noted in the PyTorch pretrained models doc,
#         with, mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
#       * convert it to a PyTorch Tensor.
#
# We can do all this preprocessing using a transform pipeline.
transform_pipeline = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

# loads image, returns cuda tensor
def image_loader(image_name):
    image = Image.open(image_name)
    image = transform_pipeline(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.cpu()  #assumes that you're using CPU

for model_name in ['vgg']:
# for model_name in ['cornet-r', 'cornet-z', 'vgg', 'densenet', 'resnet', 'squeezenet']:
    if model_name == 'vgg':
        model = models.vgg19(pretrained=True)

        selected_layers = {'block_0': model.features[4], 'block_1':  model.features[9], \
                           'block_2':  model.features[18], 'block_3':  model.features[27], \
                           'block_4':  model.features[36]}

    elif model_name == 'densenet':
        # model = models.densenet161(pretrained=True)
        model = models.densenet169(pretrained=True)

        # selected_layers = {'block_0': model.features.pool0, 'block_1': model.features.denseblock1.denselayer6.conv2, \
        #                    'block_2': model.features.denseblock2.denselayer12.conv2, 'block_3': model.features.denseblock3.denselayer36.conv2, \
        #                    'block_4': model.features.denseblock4.denselayer24.conv2}

        selected_layers = {'block_0': model.features.pool0, 'block_1': model.features.denseblock1.denselayer6.conv2, \
                           'block_2': model.features.denseblock2.denselayer12.conv2, 'block_3': model.features.denseblock3.denselayer32.conv2, \
                           'block_4': model.features.denseblock4.denselayer32.conv2}

    elif model_name == 'resnet':
        model = models.resnet152(pretrained=True)

        # selected_layers = {'block_1': model.layer1[2].conv2, 'block_2': model.layer2[7].conv2, \
        #                    'block_3': model.layer3[35].conv2, 'block_4': model.layer4[2].conv2}

        selected_layers = {'block_0': model.maxpool}

    elif model_name == 'squeezenet':
        model = models.squeezenet1_1(pretrained=True)

        selected_layers = {'block_1': model.features[4].expand3x3_activation, 'block_2': model.features[7].expand3x3_activation, \
                           'block_3': model.features[10].expand3x3_activation, 'block_4': model.features[12].expand3x3_activation}

    elif model_name == 'cornet-s':
        model = cornet.cornet_s(pretrained=True, map_location='cpu')

        # selected_layers = {'block_1': model.module.V1.output, 'block_2': model.module.V2.output, \
        #                    'block_3': model.module.V4.output, 'block_4': model.module.IT.output}

        selected_layers = {'block_0': model.module.V1.pool}

    elif model_name == 'cornet-r':
        model = cornet.cornet_r(pretrained=True, map_location='cpu')

        # selected_layers = {'block_1': model.module.V1.output, 'block_2': model.module.V2.output, \
        #                    'block_3': model.module.V4.output, 'block_4': model.module.IT.output}

        selected_layers = {'block_0': model.module.V1.nonlin_input}


    elif model_name == 'cornet-z':
        model = cornet.cornet_z(pretrained=True, map_location='cpu')

        # selected_layers = {'block_1': model.module.V1.output, 'block_2': model.module.V2.output, \
        #                    'block_3': model.module.V4.output, 'block_4': model.module.IT.output}

        selected_layers = {'block_0': model.module.V1.nonlin}
    # need to put resnet in eval mode
    model.eval()

    for run in runs:
        # register hooks on each layer
        hookF = {}
        outputs = {}
        for key,val in selected_layers.items():
            hookF[key] = Hook(val)
            outputs[key] = []

        # preprocess and convert all images
        files = glob(IMG_DIR + 'part_{0}/*'.format(run))
        files.sort(key=lambda x:int(x[:-4].split('_')[-1]))
        print('Extracting activations from {0} images from part {1}'.format(len(files), run))
        for filename in files:
            img = image_loader(filename)
            out = model(img)
            for key, hook in hookF.items():
                outputs[key].append((hook.output.data).cpu().numpy())

        for key, hook in hookF.items():
            hook_output = np.concatenate(outputs[key], axis=0)
            np.save(MODEL_DATA_DIR + '{0}/part_{1}_activations_{2}.npy'.format(model_name, run, key), hook_output)
            print(hook_output.shape)
            hook.close()
