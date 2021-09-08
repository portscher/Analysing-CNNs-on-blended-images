"""
Script to divide the set of images into training (75%) and validation sets (25%)
"""
import pandas as pd

df_all = pd.read_csv("../csv/all_labels.csv")
classes = ['boat', 'car', 'cat', 'chair', 'dog', 'face', 'flower', 'plane']

dfs = []
for ctr, c in enumerate(classes):
    dfs.append(df_all[df_all['image'].str.contains(c)])

training_dfs = []
validation_dfs = []

# shuffle and move 75% to training and 25% to validation
for df in dfs:
    df.sample(frac=1).reset_index(drop=True)

    split_at = int(len(df) * 0.8)

    train = df.iloc[:split_at, :]
    training_dfs.append(train)

    val = df.iloc[split_at:, :]
    validation_dfs.append(val)

train_csv = pd.concat(training_dfs)
train_csv.to_csv('../csv/train.csv', index=False)
val_csv = pd.concat(validation_dfs)
val_csv.to_csv('../csv/val.csv', index=False)

print("all: {}, train: {} , val: {}".format(df_all.shape, train_csv.shape, val_csv.shape))
