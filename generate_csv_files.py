import argparse
import os.path
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

"""
Script for generating a CSV file from a folder of images.
Each image name must contain the name of all relevant object classes depicted on the image.

Example: a blended image of a dog could be named dog_123.jpg
"""
def generate_label_csv(folder, class_names, file_name):
    files = [e for e in os.listdir(folder)]
    df = pd.DataFrame({'image': files})

    for class_name in class_names:
        df[class_name] = np.where(df['image'].str.contains(class_name), 1, 0)

    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)
    df.to_csv(file_name, index=False)


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--blended', action='store_true')

args = parser.parse_args()

classes = ['boat', 'car', 'cat', 'chair', 'dog', 'face', 'flower', 'plane']

if args.blended:
    base = '../blended_dataset'
else:
    base = '../reg_dataset'

generate_label_csv(f'{base}/', classes, '../csv/all_labels.csv')
generate_label_csv(f'{base}/val/', classes, '../csv/val.csv')
generate_label_csv(f'{base}/train/', classes, '../csv/train.csv')
