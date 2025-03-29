import os
import pandas as pd
import argparse
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#dataset loader
class PerturbedDataloader(Dataset):
    def __init__(self, root_dir, transform=None, csv_file='./labels.csv'):
        self.root_dir = root_dir
        self.transform = transform
        self.csv = pd.read_csv(csv_file)
        
        self.csv['Perturbed'] = self.csv['Perturbed'].map(lambda x: 1 if x == 'Yes' else 0)
        
        #store filenames and labels
        self.labels = self.csv['Perturbed'].values

        #filenames in dataset dir
        self.file_names = self.csv['Path'].values

    def __len__(self):
        """
        Returns the total number of files in the directory.
        """
        return len(self.file_names)

    def __getitem__(self, i):
        file_name = self.file_names[i]
        file_path = os.path.join(self.root_dir, file_name)
        
        if not file_name.endswith(".pt"):
            image = Image.open(file_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        else:
            image = torch.load(file_path).squeeze(0)

        # Get the label
        label = torch.tensor(self.labels[i], dtype=torch.float32)

        if torch.cuda.is_available(): 
            image = image.cuda()
            label = label.cuda()
        return image, label
        