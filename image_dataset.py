import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

mean = [0.0449, -0.0222, 0.0572]
std = [0.9864, 0.9709, 0.9740]


class ImageDataset(Dataset):
    def __init__(self, csv, directory, set_type, img_size=224):
        self.csv = csv
        self.set_type = set_type
        self.all_image_names = self.csv[:]['image']
        self.all_labels = np.array(self.csv.drop('image', axis=1))
        self.dir = directory
        self.img_size = img_size

        print(f"Number of {set_type} images: {len(self.all_labels)}")
        self.image_names = list(self.all_image_names)
        self.labels = list(self.all_labels)

        if self.set_type == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        elif self.set_type == 'val' or self.set_type == 'test':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = cv2.imread(self.dir + self.image_names[index])
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]

        return {
            'image': torch.as_tensor(image, dtype=torch.float32),
            'label': torch.as_tensor(targets, dtype=torch.float32)
        }
