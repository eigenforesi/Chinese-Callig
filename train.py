#Imports
import numpy as np
#import matplotlib.pyplot as plt
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








### Model Architecture
"""Input size: (64, 64, 3) - from the images in the dataset​
2D Convolutional layer 1​
3x3 filter, stride 2, padding 1, 16 output channels, output size (32, 32, 16)​
2D Convolutional layer 2​
3x3 filter, stride 2, padding 1, 32 output channels, output size (16, 16, 32)​
2D Convolutional layer 3​
3x3 filter, stride 2, padding 1, 64 output channels, output size (8, 8, 64)​
Max Pooling layer (2x2)​
2x2, stride 1, padding 0, output size (7, 7, 64)​
Fully Connected Layer​
Output size 20​
ReLU Activation Function​"""

### Model Definition
class Callclassifier(nn.Module):
    def __init__(self):
        super(Callclassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.fc = nn.Linear(in_features=64 * 7 * 7, out_features=20)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        return x
    
###Define Hyperparameters
model = Callclassifier(inputdim=(64, 64, 3), num_classes=20)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


### Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
