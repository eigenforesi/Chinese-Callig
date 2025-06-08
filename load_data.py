# Imports
import numpy as np
import os
import torchvision.io as io
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

# This class handles loading data from a specified directory of images
class CalligData:
    # Initialize the class with the base directory for data
    def __init__(self, base_dir='archive/data/data'):
        # Initialize base directory and the training and testing features and targets
        self.base_dir = base_dir
        self.train_features = []
        self.test_features = []
        self.train_targets = []
        self.test_targets = []

    # Load data from the specified directory by traversing through the training and testing directories
    # If only_test is True, it will only load the test data
    def load_data(self, only_test=False):
        self.train_features, self.test_features, self.train_targets, self.test_targets = [], [], [], []

        # Traverse through the training and testing data directories
        for dir_name in os.listdir(self.base_dir):
            if ((dir_name == 'train' or dir_name == 'test') and only_test == False) or (dir_name == 'test' and only_test == True):
                # Enter training data directory
                dir_path = os.path.join(self.base_dir, dir_name)
                category = dir_name # 'train' or 'test'
                for sub_dir in os.listdir(dir_path):
                    # Enter each sub-directory of the training data - corresponds to different calligraphers
                    sub_dir_path = os.path.join(dir_path, sub_dir)
                    if os.path.isdir(sub_dir_path): # Check if it is a directory
                        for file in os.listdir(sub_dir_path): # Traverse through all files in the sub-directory
                            if file.endswith('.jpg'):
                                img_path = os.path.join(sub_dir_path, file) # Get the full path of the image file
                                image_pil = Image.open(img_path) # Open the image file
                                image = np.array(image_pil) # Convert the image to a numpy array
                                if category == 'train': # Append the image and its corresponding sub-directory name to the training features and targets
                                    self.train_features.append(image)
                                    self.train_targets.append(sub_dir)
                                elif category == 'test': # Append the image and its corresponding sub-directory name to the testing features and targets
                                    self.test_features.append(image)
                                    self.test_targets.append(sub_dir)
                    print(f'Extracted images from {sub_dir_path} in {dir_name} directory')
                print("Extracted images from directory:", dir_name)

        # Print dimensions of an individual image
        print(f'Individual image dimensions: {self.train_features[0].shape if self.train_features else "No images found"}') 
        # Individual image dimensions: (64, 64, 3) Training data features shape: (64, 64, 3)

    def get_train_features(self): # Returns the training features as a numpy array
        return np.array(self.train_features)

    def get_test_features(self): # Returns the testing features as a numpy array
        return np.array(self.test_features)

    def get_train_targets(self): # Returns the training targets as a numpy array
        return np.array(self.train_targets)

    def get_test_targets(self): # Returns the testing targets as a numpy array
        return np.array(self.test_targets)

