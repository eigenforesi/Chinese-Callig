# Imports
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchvision
import torch.nn as nn
from IPython.display import Image
import os



# LOAD DATA
# Traverse through the 20 directories in the training and testing data directories, and extract images from each directory
train_features = []
test_features = []
train_targets = []
test_targets = []
base_dir = 'archive/data/data'

# Traverse through the training data directory
for dir_name in os.listdir(base_dir):
    if dir_name == 'train' or dir_name == 'test':
        # Enter training data directory
        dir_path = os.path.join(base_dir, dir_name)
        category = dir_name # 'train' or 'test'
        for sub_dir in os.listdir(dir_path):
            # Enter each sub-directory of the training data - corresponds to different calligraphers
            sub_dir_path = os.path.join(dir_path, sub_dir)
            if os.path.isdir(sub_dir_path): # Check if it is a directory
                for file in os.listdir(sub_dir_path): # Traverse through all files in the sub-directory
                    if file.endswith('.jpg'):
                        img_path = os.path.join(sub_dir_path, file) # Get the full path of the image file
                        # img = torchvision.io.read_image(img_path)
                        if category == 'train':
                            # train_features.append(img.numpy())
                            train_features.append(np.array(img_path))
                            train_targets.append(sub_dir)
                        elif category == 'test':
                            # test_features.append(img.numpy())
                            test_features.append(np.array(img_path))
                            test_targets.append(sub_dir)
        print("Extracted images from directory:", dir_name)

print(f'Training data features shape: {train_features.shape}')
print(f'Training data targets shape: {train_targets.shape}')
print(f'Test data features shape: {test_features.shape}')
print(f'Test data targets shape: {test_targets.shape}')

# Print dimensions of an individual image
print(f'Individual image dimensions: {train_features[0].shape if train_features else "No images found"}')
