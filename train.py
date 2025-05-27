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
from load_data import CalligData


# Load data
data = CalligData(base_dir='archive/data/data')
data.load_data()
# Get training and testing data
train_features = data.get_train_features()
train_targets = data.get_train_targets()
test_features = data.get_test_features()
test_targets = data.get_test_targets()


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
