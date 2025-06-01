# Imports
import numpy as np
import os
import torchvision.io as io
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
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

    # def flatten_data(self):
    #     # Reshape the training and testing features to flatten the images
    #     train_features_flat = np.array(self.train_features)
    #     test_features_flat = np.array(self.test_features)
    #     print(f"Flattened train and test data to sizes:{train_features_flat.shape}, {test_features_flat.shape}") #(84022, 64, 64, 3) (21007, 64, 64, 3)
    #     return train_features_flat, test_features_flat

    # def train_valid_split(self, train_features, train_targets, valid_size=0.2, random_state=42):
    #     return train_test_split(
    #         train_features, train_targets, test_size=valid_size, random_state=random_state, stratify=train_targets
    #     )
    #     print("Successfully split training data into training and validation sets.")
    
    # def get_torch_tensors(self, train_features, val_features, test_features):
    #     import torch
    #     # Convert to torch tensors
    #     train_features = torch.tensor(train_features, dtype=torch.float32)
    #     test_features = torch.tensor(test_features, dtype=torch.float32)
    #     val_features = torch.tensor(val_features, dtype=torch.float32)

    #     # Reshape inputs to (B, C, H, W) format for PyTorch
    #     train_features = train_features.permute(0, 3, 1, 2) # (N, H, W, C) to (N, C, H, W)
    #     test_features = test_features.permute(0, 3, 1, 2) # (N, H, W, C) to (N, C, H, W)
    #     val_features = val_features.permute(0, 3, 1, 2) # (N, H, W, C) to (N, C, H, W)

    #     # Print shapes of the datasets
    #     print("Successfully converted data to torch tensors.")
    #     print(f"Train features shape: {train_features.shape}, Train targets shape: {len(self.train_targets)}")
    #     print(f"Validation features shape: {val_features.shape}, Validation targets shape: {len(self.train_targets)}")
    #     print(f"Test features shape: {test_features.shape}, Test targets shape: {len(self.test_targets)}")
    #     return train_features, val_features, test_features

    # def get_data_loaders(self, train_dataset, val_dataset, test_dataset, batch_size=32):
    #     # Create DataLoader for training, validation, and testing datasets
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #     print("Successfully created DataLoaders for training, validation, and testing datasets.")
    #     return train_loader, val_loader, test_loader

    # def get_tensor_dataset(self, features, targets):
    #     # Convert targets to torch tensors
    #     targets = torch.tensor(targets, dtype=torch.long)
    #     return TensorDataset(features, targets)

    # def get_tensor_datasets(self, train_features, train_targets, val_features, val_targets, test_features, test_targets):
    #     # Convert training and validation datasets to TensorDataset
    #     train_dataset = self.get_tensor_dataset(train_features, train_targets)
    #     val_dataset = self.get_tensor_dataset(val_features, val_targets)
    #     test_dataset = self.get_tensor_dataset(test_features, test_targets)
    #     print("Successfully created TensorDatasets for training, validation, and testing datasets.")
    #     return train_dataset, val_dataset, test_dataset

    # # Method to prepare data for the model (assuming the load_data method has already been called)
    # def PrepareData(self, valid_size=0.2, batch_size=32):
    #     # Flatten the data
    #     train_features, test_features_np = self.flatten_data()

    #     # Split training data into training and validation sets
    #     train_features_np, val_features_np, train_targets, val_targets = self.train_valid_split(valid_size, train_features, self.train_targets)

    #     # Convert to torch tensors
    #     train_features, val_features, test_features = self.get_torch_tensors(train_features_np, val_features_np, test_features_np)

    #     # Convert string labels to integer class indices
    #     from sklearn.preprocessing import LabelEncoder
    #     le = LabelEncoder()
    #     train_targets = le.fit_transform(train_targets)
    #     test_targets = le.transform(test_targets)
    #     val_targets = le.transform(val_targets)

    #     # Get tensor datasets
    #     train_dataset, val_dataset, test_dataset = self.get_tensor_datasets(train_features, train_targets, val_features, val_targets, test_features, test_targets)

    #     # Get DataLoaders
    #     train_loader, val_loader, test_loader = self.get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)
    #     print("Data preparation complete.")
    #     return train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset

# TESTING
# data = CalligData(base_dir='archive/data/data')
# data.load_data()

