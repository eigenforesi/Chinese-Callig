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

### Flatten the data 
# Reshape the training and testing features to flatten the images
train_features = np.array(train_features)
test_features = np.array(test_features)
print(train_features.shape, test_features.shape) #(84022, 64, 64, 3) (21007, 64, 64, 3)

# Training data split - validation split
from sklearn.model_selection import train_test_split
train_features, val_features, train_targets, val_targets = train_test_split(
    train_features, train_targets, test_size=0.2, random_state=42, stratify=train_targets
)

# Convert to torch tensors
train_features = torch.tensor(train_features, dtype=torch.float32)
test_features = torch.tensor(test_features, dtype=torch.float32)
val_features = torch.tensor(val_features, dtype=torch.float32)

# Convert string labels to integer class indices
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_targets = le.fit_transform(train_targets)
test_targets = le.transform(test_targets)
val_targets = le.transform(val_targets)

# Convert targets to torch tensors
train_targets = torch.tensor(train_targets, dtype=torch.long)
test_targets = torch.tensor(test_targets, dtype=torch.long)
val_targets = torch.tensor(val_targets, dtype=torch.long)

# Reshape inputs to (B, C, H, W) format for PyTorch
train_features = train_features.permute(0, 3, 1, 2)  # (N, H, W, C) to (N, C, H, W)
test_features = test_features.permute(0, 3, 1, 2)  # (N, H, W, C) to (N, C, H, W)
val_features = val_features.permute(0, 3, 1, 2)  # (N, H, W, C) to (N, C, H, W)
# Print shapes of the datasets
print(f"Train features shape: {train_features.shape}, Train targets shape: {train_targets.shape}")
print(f"Validation features shape: {val_features.shape}, Validation targets shape: {val_targets.shape}")
print(f"Test features shape: {test_features.shape}, Test targets shape: {test_targets.shape}")

# Create DataLoader for training and validation datasets
from torch.utils.data import DataLoader, TensorDataset
train_dataset = TensorDataset(train_features, train_targets)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(val_features, val_targets)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = TensorDataset(test_features, test_targets)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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
model = Callclassifier()  # Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Identify Tracked Values
train_loss_list = []             
validation_accuracy_list = [] 

### Training Loop
import tqdm
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    #Training loop
    for xb, yb in train_loader:  # Iterate over batches of training data
        
        output = model(xb)  # Forward pass: compute model predictions
        loss = loss_func(output, yb) # Compute the loss between predictions and true labels
        
        optimizer.zero_grad()  # Clear gradients from the previous step
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters using the optimizer
        
        
        train_loss_list.append(loss.item()) # Store the training loss for this batch
    
    # Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # Disable gradient computation for validation
        for xb, yb in val_loader:
            
            output = model(xb)  # Forward pass: compute model predictions
            loss = loss_func(output, yb)  # Compute the loss between predictions and true labels
            val_loss += loss.item()  # Accumulate validation loss
    
    val_loss /= len(val_loader)  # Average validation loss

    # Calculate validation accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            outputs = model(xb)
            _, predicted = torch.max(outputs.data, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    val_accuracy = 100 * correct / total if total > 0 else 0

    validation_accuracy_list.append(val_loss)  # Store the validation loss for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {np.mean(train_loss_list):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")