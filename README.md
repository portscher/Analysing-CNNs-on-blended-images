# Evaluating Attention in Convolutional Neural Networks for Blended Images

## Abstract

In neuroscience, blended images have oftentimes been used to examine the way in which attention mechanisms in the human brain work. They are particularly suited for experiments in this research area, as a subject needs to focus on particular features on an image in order to be able to classify the superimposed objects.
Since there are many parallels between the human visual system and Convolutional Neural Networks -- such as the hierarchical structure where different levels of abstraction are processed on different network layers -- we examine, how CNNs perform on this task.
In this paper, we evaluate the performance of selected CNN architectures (ResNet18, ResNet50, EfficientNet B0, CORnet-Z, CORnet-S and Inception V3) as well as a Vision Transformer on the classification of objects in blended images. 
We first trained them with regular, non-blended images to see how well the networks can abstract to classifying blended images. 
Second, we trained them with blended images to find out how their performance changes when they are primed for this specific use case.
Since humans solve this task by applying selective attention, we also measured the impact an attention mechanism: the Attention Augmented Convolutional Network (AACN), using multi-headed self-attention. 
Since the task can rather easily be solved by humans we examined whether a correlation between the similarity of a network architecture's structure to the human visual system and its ability to correctly classify objects on blended images could be found. 


## Sources

The basic EfficientNet definition (without attention) is from https://github.com/zsef123/EfficientNets-PyTorch.
The Vision Transformer definition is from https://github.com/rwightman/pytorch-image-models.
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
python3 main.py [-h] --arch {resnet18,resnet50,inception,efficientnet,cornet_z,cornet_s,vision_transformer} [--attention {aacn,none}] [--heads NUM_HEADS] [--blended] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR]

```
### Predicting
```
python3 predict.py [-h] --arch {resnet18,resnet50,inception,efficientnet,cornet_z,cornet_s,vision_transformer} --path PATH [--blended] [--attention {aacn,none}] [--heads NUM_HEADS] --test_folder TEST_FOLDER

```
