# Imports
import numpy as np
import os
import torchvision.io as io
from PIL import Image

class CalligData:
    # Initialize the class with the base directory for data
    def __init__(self, base_dir='archive/data/data'):
        self.base_dir = base_dir
        self.train_features = []
        self.test_features = []
        self.train_targets = []
        self.test_targets = []

    # Load data from the specified directory
    def load_data(self):
        self.train_features, self.test_features, self.train_targets, self.test_targets = [], [], [], []

        # Traverse through the training and testing data directories
        for dir_name in os.listdir(self.base_dir):
            if dir_name == 'train' or dir_name == 'test':
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
                                if category == 'train':
                                    self.train_features.append(image)
                                    self.train_targets.append(sub_dir)
                                elif category == 'test':
                                    self.test_features.append(image)
                                    self.test_targets.append(sub_dir)
                    print(f'Extracted images from {sub_dir_path} in {dir_name} directory')
                print("Extracted images from directory:", dir_name)
        #print(f'Training data features type: {type(self.train_features[0])}') <class 'numpy.ndarray'>
        #print(f'Training data features shape: {self.train_features[0].shape}') 

        # Print dimensions of an individual image
        print(f'Individual image dimensions: {self.train_features[0].shape if self.train_features else "No images found"}') 
        # Individual image dimensions: (64, 64, 3) Training data features shape: (64, 64, 3)

    def get_train_features(self):
        return np.array(self.train_features)

    def get_test_features(self):
        return np.array(self.test_features)

    def get_train_targets(self):
        return np.array(self.train_targets)

    def get_test_targets(self):
        return np.array(self.test_targets)
    



# TESTING
data = CalligData(base_dir='archive/data/data')
data.load_data()

