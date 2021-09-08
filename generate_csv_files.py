import argparse
import os.path
import pandas as pd
from sklearn.utils import shuffle
import numpy as np


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

# generate_label_csv(f'{base}/', classes, '../csv/all_labels.csv')
# generate_label_csv(f'{base}/val/', classes, '../csv/bl_val.csv')
# generate_label_csv(f'{base}/train/', classes, '../csv/bl_train.csv')
generate_label_csv(f'../blended_test/', classes, '../csv/bl_test_labels.csv')
