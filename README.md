# Evaluating Attention in Convolutional Neural Networks for Blended Images

## Abstract
In neuroscientific experiments, blended images are used to examine how attention mechanisms in the human brain work. They are particularly suited for this research area, as a subject needs to focus on particular features in an image to be able to classify superimposed objects. As Convolutional Neural Networks (CNNs) take some inspiration from the mammalian visual system -- such as the hierarchical structure where different levels of abstraction are processed on different network layers -- we examine how CNNs perform on this task. More specifically, we evaluate the performance of four popular CNN architectures (ResNet18, ResNet50, CORnet-Z, and Inception V3) on the classification of objects in blended images. Since humans can rather easily solve this task by applying object-based attention, we also augment all architectures with a multi-headed self-attention mechanism to examine its effect on performance. Lastly, we analyse if there is a correlation between the similarity of a network architecture's structure to the human visual system and its ability to correctly classify objects in blended images. Our findings showed that adding a self-attention mechanism reliably increases the similarity to the V4 area of the human ventral stream, an area where attention has a large influence on the processing of visual stimuli.


## Sources
The AACN code is (modified) from https://github.com/MartinGer/Attention-Augmented-Convolutional-Networks.

## Requirements

See requirements.txt

## Data

The corresponding dataset (including csv-files) can be found at https://www.kaggle.com/andreaport/blendeduibk

## Usage

Folder structure:
 
``` 
main folder
│   README.md
└─── code
│   │   main.py
│   │   predict.py
|   |   ...
│   │
│   └───architectures
│       │   ...
│   
└─── blended_dataset
└─── blended_test
└─── reg_dataset
└─── reg_test
└─── csv
    │   bl_test_labels.csv
    │   teset_labels.csv
    └───blended
    |      train.csv
    |      val.csv
    └───reg
           train.csv
           val.csv        
```
### Training a network from scratch
```
python3 main.py [-h] --arch {resnet18,resnet50,inception,cornet_z} [--attention {aacn,none}] [--heads NUM_HEADS] [--blended] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR]

```
### Predicting
```
python3 predict.py [-h] --arch {resnet18,resnet50,inception,cornet_z} --path PATH [--blended] [--attention {aacn,none}] [--heads NUM_HEADS] --test_folder TEST_FOLDER

```
