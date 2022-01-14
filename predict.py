import argparse
import sys

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import image_dataset
import utils
from architectures import resnet50, inception_v3, cornet_z, cornet_s, resnet18, efficientnet_b0, vit

###################################################################################################################
# Parse user input
###################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--arch', help='<Required> Which model to use', required=True,
                    choices=['resnet18', 'resnet50', 'inception', 'efficientnet', 'cornet_z', 'cornet_s',
                             'vision_transformer'])
parser.add_argument('--path', required=True)
parser.add_argument('--blended', action='store_true')
parser.add_argument('--attention', choices=['cbam', 'aacn', 'none'], default='none')
parser.add_argument('--test_folder', required=True)


args = parser.parse_args()

if not args.arch or not args.test_folder:
    parser.print_help()
    sys.exit()

###################################################################################################################
# Load test data
###################################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.blended:
    test_csv = pd.read_csv('../csv/blended/test.csv')
else:
    test_csv = pd.read_csv('../csv/reg/test.csv')

classes = test_csv.columns.values[1:]

if args.arch.lower() == 'inception':
    img_size = 299
else:
    img_size = 224

test_set = image_dataset.ImageDataset(csv=test_csv, directory=args.test_folder, set_type='test', img_size=img_size)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

###################################################################################################################
# Initialize the model architecture
###################################################################################################################
arch = None

if args.arch.lower() == 'resnet18':
    arch = resnet18.ResNet18(path=args.path, train_from_scratch=False, attention=args.attention)
elif args.arch.lower() == 'resnet50':
    arch = resnet50.ResNet50(path=args.path, train_from_scratch=False, attention=args.attention)
elif args.arch.lower() == 'inception':
    arch = inception_v3.Inception(path=args.path, train_from_scratch=False, attention=args.attention)
elif args.arch.lower() == 'cornet_z':
    arch = cornet_z.CORnet(path=args.path, train_from_scratch=False, attention=args.attention)
elif args.arch.lower() == 'cornet_s':
    arch = cornet_s.CORnet(path=args.path, train_from_scratch=False, attention=args.attention)
elif args.arch.lower() == 'efficientnet':
    arch = efficientnet_b0.EfficientNetB0(path=args.path, train_from_scratch=False, attention=args.attention)
elif args.arch.lower() == 'vision_transformer':
    arch = vit.ViT(path=args.path, train_from_scratch=False)

arch = arch.get_model().to(device)

###################################################################################################################
# Visualize and save filters
###################################################################################################################
model_weights = []  # we will save the conv layer weights in this list
conv_layers = []  # we will save the 49 conv layers in this list

# get all the model children as list
model_children = list(arch.children())

# counter to keep count of the conv layers
counter = 0
# append all the conv layers and their respective weights to the list
# for i in range(len(model_children)):
#     if type(model_children[i]) == nn.Conv2d:
#         counter += 1
#         model_weights.append(model_children[i].weight)
#         conv_layers.append(model_children[i])
#     elif type(model_children[i]) == nn.Sequential:
#         for j in range(len(model_children[i])):
#             for child in model_children[i][j].children():
#                 if type(child) == nn.Conv2d:
#                     counter += 1
#                     model_weights.append(child.weight)
#                     conv_layers.append(child)
# print(f"Total convolutional layers: {counter}")
#
# # take a look at the conv layers and the respective weights
# for weight, conv in zip(model_weights, conv_layers):
#     print(len(conv_layers))
#     print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
#
# plt.figure(figsize=(20, 17))
# for i, filter in enumerate(model_weights[args.layer]):
#     plt.subplot(args.fsize, args.fsize, i + 1)
#     plt.imshow(filter[0, :, :].detach().cpu().numpy(), cmap='viridis')
#     plt.axis('off')
#     plt.savefig(f'filters/{args.arch.lower()}_layer_{args.layer}.png')
# plt.show()

# Visualize conv filter
kernels = arch.conv1.weight.detach()
fig, axarr = plt.subplots(kernels.size(0))
#for idx in range(kernels.size(0)):
#    axarr[idx].imshow(kernels[idx].squeeze())


kernels = arch.conv1.weight.detach().cpu()
print(kernels.size())
utils.visualize_tensor(kernels, args.arch.lower(), ch=0, allkernels=False)

###################################################################################################################
# Start prediction and process results
###################################################################################################################

both_correct = 0
one_correct = 0
incorrect = 0

FILENAME = 'buffer.txt'
hyperclass_dict = {'ManMade-ManMade': [0, 0, 0, (0, 0)], 'ManMade-Nature': [0, 0, 0, (0, 0)],
                   'Human-ManMade': [0, 0, 0, (0, 0)],
                   'Nature-Nature': [0, 0, 0, (0, 0)], 'Human-Nature': [0, 0, 0, (0, 0)]}

with torch.no_grad():
    for counter, data in tqdm(enumerate(test_loader), total=int(len(test_set) / test_loader.batch_size)):
        image, target = data['image'].to(device), data['label'].to(device)
        # get all the index positions where value == 1
        target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]

        # get the predictions by passing the image through the model
        prediction = arch(image)

        prediction = torch.sigmoid(prediction)
        prediction = prediction.detach().cpu()
        prediction_array = prediction[0].numpy()

        # get the indices of the 2 highest percentages in the list
        prediction_indices = sorted(range(len(prediction_array)), key=lambda i: prediction_array[i])[-2:]

        res = utils.check_if_both_one_or_none_correct(classes, target_indices, prediction_indices)
        which_class = None
        if res == 'both_correct':
            both_correct += 1
        elif res == 'one_correct':
            one_correct += 1
            which_class = utils.which_correct(classes, target_indices, prediction_indices)
        elif res == 'incorrect':
            incorrect += 1

        if len(target_indices) > 1:
            hyperclass_flag = False
            class_names = []
            for i in range(len(target_indices)):
                class_names.append(classes[target_indices[i]])

            utils.update_hyperclass_dict(hyperclass_dict, res, which_class, class_names[0], class_names[1])

        f = open(FILENAME, "a")
        f.write(res + "\n")
        f.close()

    total = both_correct + one_correct + incorrect
    print(f"Correct: {both_correct}, total: {total}, validation accuracy: {(100 * both_correct / total):.4f}")

utils.print_results(both_correct, one_correct, incorrect, hyperclass_dict)

utils.save_hyperclass_results(model_name=args.arch,
                              attention=args.attention,
                              both_correct=both_correct,
                              one_correct=one_correct,
                              none_correct=incorrect,
                              hyperclass_dict=hyperclass_dict)
