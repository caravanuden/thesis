from vids.models.resnet_3D import resnet18
import pickle
import torch

# pprint([(layer, weights['blobs'][layer].shape) for layer in layer_names if layer.split('_')[1] == '0'])
with open('cnn_models/r2.5d_d34_l32_ft_sports1m.pkl', 'rb') as fp:
# with open('/checkpoint/bkorbar/caffe2_models/r3d_d18_l16.pkl', 'rb') as fp:
    weights = pickle.load(fp, encoding='latin1')

def get_blob(weights, name):
    return torch.from_numpy(weights['blobs'][name])



model = resnet18(
    sample_size=112,
    sample_duration=8,
    num_classes=400
)

print(model.conv1.weight.data[0][0])
model.conv1.weight.data = get_blob(weights, "conv1_w")
print(model.conv1.weight.data[0][0])

model.bn1.weight.data = get_blob(weights, 'conv1_spatbn_relu_s')
model.bn1.running_mean.data = get_blob(weights, 'conv1_spatbn_relu_rm')
model.bn1.running_var.data = get_blob(weights,  'conv1_spatbn_relu_riv')
model.bn1.bias.data = get_blob(weights, 'conv1_spatbn_relu_b')

for i in range(0,2):
    layer_counter = 0
    model.layer1[i % 2].conv1.weight.data = get_blob(weights,'comp_{}_conv_1_w'.format(layer_counter+i))
    model.layer1[i % 2].bn1.weight.data = get_blob(weights, 'comp_{}_spatbn_1_s'.format(layer_counter+i))
    model.layer1[i % 2].bn1.bias.data = get_blob(weights, 'comp_{}_spatbn_1_b'.format(layer_counter+i))
    model.layer1[i % 2].bn1.running_mean.data = get_blob(weights, 'comp_{}_spatbn_1_rm'.format(layer_counter+i))
    model.layer1[i % 2].bn1.running_var.data = get_blob(weights, 'comp_{}_spatbn_1_riv'.format(layer_counter+i))

    model.layer1[i % 2].conv2.weight.data = get_blob(weights, 'comp_{}_conv_2_w'.format(layer_counter+i))
    model.layer1[i % 2].bn2.weight.data = get_blob(weights, 'comp_{}_spatbn_2_s'.format(layer_counter+i))
    model.layer1[i % 2].bn2.bias.data = get_blob(weights, 'comp_{}_spatbn_2_b'.format(layer_counter+i))
    model.layer1[i % 2].bn2.running_mean.data = get_blob(weights, 'comp_{}_spatbn_2_rm'.format(layer_counter+i))
    model.layer1[i % 2].bn2.running_var.data = get_blob(weights, 'comp_{}_spatbn_2_riv'.format(layer_counter+i))
    layer_counter += 2

    model.layer2[i % 2].conv1.weight.data = get_blob(weights,'comp_{}_conv_1_w'.format(layer_counter+i))
    model.layer2[i % 2].bn1.weight.data = get_blob(weights, 'comp_{}_spatbn_1_s'.format(layer_counter+i))
    model.layer2[i % 2].bn1.bias.data = get_blob(weights, 'comp_{}_spatbn_1_b'.format(layer_counter+i))
    model.layer2[i % 2].bn1.running_mean.data = get_blob(weights, 'comp_{}_spatbn_1_rm'.format(layer_counter+i))
    model.layer2[i % 2].bn1.running_var.data = get_blob(weights, 'comp_{}_spatbn_1_riv'.format(layer_counter+i))

    model.layer2[i % 2].conv2.weight.data = get_blob(weights, 'comp_{}_conv_2_w'.format(layer_counter+i))
    model.layer2[i % 2].bn2.weight.data = get_blob(weights,'comp_{}_spatbn_2_s'.format(layer_counter+i))
    model.layer2[i % 2].bn2.bias.data = get_blob(weights, 'comp_{}_spatbn_2_b'.format(layer_counter+i))
    model.layer2[i % 2].bn2.running_mean.data = get_blob(weights, 'comp_{}_spatbn_2_rm'.format(layer_counter+i))
    model.layer2[i % 2].bn2.running_var.data = get_blob(weights, 'comp_{}_spatbn_2_riv'.format(layer_counter+i))
    layer_counter += 2

    model.layer3[i % 2].conv1.weight.data = get_blob(weights,'comp_{}_conv_1_w'.format(layer_counter+i))
    model.layer3[i % 2].bn1.weight.data = get_blob(weights,'comp_{}_spatbn_1_s'.format(layer_counter+i))
    model.layer3[i % 2].bn1.bias.data = get_blob(weights, 'comp_{}_spatbn_1_b'.format(layer_counter+i))
    model.layer3[i % 2].bn1.running_mean.data = get_blob(weights, 'comp_{}_spatbn_1_rm'.format(layer_counter+i))
    model.layer3[i % 2].bn1.running_var.data = get_blob(weights, 'comp_{}_spatbn_1_riv'.format(layer_counter+i))

    model.layer3[i % 2].conv2.weight.data = get_blob(weights, 'comp_{}_conv_2_w'.format(layer_counter+i))
    model.layer3[i % 2].bn2.weight.data = get_blob(weights, 'comp_{}_spatbn_2_s'.format(layer_counter+i))
    model.layer3[i % 2].bn2.bias.data = get_blob(weights, 'comp_{}_spatbn_2_b'.format(layer_counter+i))
    model.layer3[i % 2].bn2.running_mean.data = get_blob(weights, 'comp_{}_spatbn_2_rm'.format(layer_counter+i))
    model.layer3[i % 2].bn2.running_var.data = get_blob(weights, 'comp_{}_spatbn_2_riv'.format(layer_counter+i))
    layer_counter += 2

    model.layer4[i % 2].conv1.weight.data = get_blob(weights,'comp_{}_conv_1_w'.format(layer_counter+i))
    model.layer4[i % 2].bn1.weight.data = get_blob(weights,'comp_{}_spatbn_1_s'.format(layer_counter+i))
    model.layer4[i % 2].bn1.bias.data = get_blob(weights, 'comp_{}_spatbn_1_b'.format(layer_counter+i))
    model.layer4[i % 2].bn1.running_mean.data = get_blob(weights, 'comp_{}_spatbn_1_rm'.format(layer_counter+i))
    model.layer4[i % 2].bn1.running_var.data = get_blob(weights, 'comp_{}_spatbn_1_riv'.format(layer_counter+i))

    model.layer4[i % 2].conv2.weight.data = get_blob(weights, 'comp_{}_conv_2_w'.format(layer_counter+i))
    model.layer4[i % 2].bn2.weight.data = get_blob(weights,'comp_{}_spatbn_2_s'.format(layer_counter+i))
    model.layer4[i % 2].bn2.bias.data = get_blob(weights, 'comp_{}_spatbn_2_b'.format(layer_counter+i))
    model.layer4[i % 2].bn2.running_mean.data = get_blob(weights, 'comp_{}_spatbn_2_rm'.format(layer_counter+i))
    model.layer4[i % 2].bn2.running_var.data = get_blob(weights, 'comp_{}_spatbn_2_riv'.format(layer_counter+i))

model.layer2[0].downsample[0].weight.data = get_blob(weights,'shortcut_projection_2_w')
model.layer2[0].downsample[1].weight.data = get_blob(weights, 'shortcut_projection_2_spatbn_s')
model.layer2[0].downsample[1].bias.data = get_blob(weights, 'shortcut_projection_2_spatbn_b')
model.layer2[0].downsample[1].running_mean.data = get_blob(weights, 'shortcut_projection_2_spatbn_rm')
model.layer2[0].downsample[1].running_var.data = get_blob(weights, 'shortcut_projection_2_spatbn_riv')

model.layer3[0].downsample[0].weight.data = get_blob(weights,'shortcut_projection_4_w')
model.layer3[0].downsample[1].weight.data = get_blob(weights, 'shortcut_projection_4_spatbn_s')
model.layer3[0].downsample[1].bias.data = get_blob(weights, 'shortcut_projection_4_spatbn_b')
model.layer3[0].downsample[1].running_mean.data = get_blob(weights, 'shortcut_projection_4_spatbn_rm')
model.layer3[0].downsample[1].running_var.data = get_blob(weights, 'shortcut_projection_4_spatbn_riv')

model.layer4[0].downsample[0].weight.data = get_blob(weights,'shortcut_projection_6_w')
model.layer4[0].downsample[1].weight.data = get_blob(weights, 'shortcut_projection_6_spatbn_s')
model.layer4[0].downsample[1].bias.data = get_blob(weights, 'shortcut_projection_6_spatbn_b')
model.layer4[0].downsample[1].running_mean.data = get_blob(weights, 'shortcut_projection_6_spatbn_rm')
model.layer4[0].downsample[1].running_var.data = get_blob(weights, 'shortcut_projection_6_spatbn_riv')


model.fc.weight.data = get_blob(weights, 'last_out_L400_w')
model.fc.bias.data = get_blob(weights, 'last_out_L400_b')


print(model)
from examples.classification.utils.checkpointer import Checkpointer
checkpointer = Checkpointer(model, save_dir=".", save_to_disk=True)
checkpointer.save(45, "r3d_18_l16_c2", None)
