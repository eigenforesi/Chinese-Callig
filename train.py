### IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchvision
import torch.nn as nn
from IPython.display import Image
import os
from load_data import CalligData
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import tqdm
import random


### MOMDEL ARCHITECTURE
"""Input size: (64, 64, 3) - from the images in the dataset​
2D Convolutional layer 1​
3x3 filter, stride 2, padding 1, 16 output channels, output size (32, 32, 16)​
Batch Normalization layer 1​
2D Convolutional layer 2​
3x3 filter, stride 2, padding 1, 32 output channels, output size (16, 16, 32)​
Batch Normalization layer 2​
2D Convolutional layer 3​
3x3 filter, stride 2, padding 1, 64 output channels, output size (8, 8, 64)​
Batch Normalization layer 3​
Max Pooling layer (2x2)​
2x2, stride 1, padding 0, output size (7, 7, 64)​
Dropout layer with 30% dropout rate​
Fully Connected Layer​ 1
Output size 128
ReLU Activation Function​
Fully Connected Layer​ 2
Output size 20 (number of classes)"""

### MODEL CLASS DEFINITION
class Callclassifier(nn.Module):
    # Define layers
    def __init__(self):
        super(Callclassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1) # Convolutional layer 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1) # Convolutional layer 2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1) # Convolutional layer 3
        self.bn1 = nn.BatchNorm2d(16) # Batch Normalization layer 1
        self.bn2 = nn.BatchNorm2d(32) # Batch Normalization layer 2
        self.bn3 = nn.BatchNorm2d(64) # Batch Normalization layer 3
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0) # Max Pooling layer
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128) # Fully Connected Layer 1
        self.fc2 = nn.Linear(in_features=128, out_features=20) # Fully Connected Layer 2 (output layer)
        self.relu = nn.ReLU() # ReLU Activation Function
        self.dropout = nn.Dropout(0.3) # Dropout layer with 30% dropout rate

    # Define the forward pass
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) # Feed through Conv1, BatchNorm1, and ReLU
        x = self.relu(self.bn2(self.conv2(x))) # Feed through Conv2, BatchNorm2, and ReLU
        x = self.relu(self.bn3(self.conv3(x))) # Feed through Conv3, BatchNorm3, and ReLU
        x = self.pool(x) # Apply Max Pooling
        x = x.view(-1, 64 * 7 * 7) # Flatten the output for the fully connected layers
        x = self.dropout(x) # Apply Dropout
        x = self.fc1(x) # Feed through Fully Connected Layer 1
        x = self.fc2(self.relu(x)) # Apply ReLU and feed through Fully Connected Layer 2
        return x # Result is the output
    
# Main function of the script (only runs when this script is run directly)
def main():
    # LOAD AND PREPARE DATA
    # Load data
    data = CalligData(base_dir='archive/data/data') # Initialize the CalligData class (see load_data.py)
    data.load_data() # Search through all the directories and generate lists of images and labels

    # Get features and targets for both the training and testing datasets from the CalligData object
    train_features = data.get_train_features()
    train_targets = data.get_train_targets()
    test_features = data.get_test_features()
    test_targets = data.get_test_targets()

    # Flatten the data 
    # Convert the lists of images to numpy arrays
    train_features = np.array(train_features)
    test_features = np.array(test_features)
    print(train_features.shape, test_features.shape) # (84022, 64, 64, 3) (21007, 64, 64, 3) expected

    # Split training data into training and validation sets (80% train, 20% validation)
    train_features, val_features, train_targets, val_targets = train_test_split(
        train_features, train_targets, test_size=0.2, random_state=42, stratify=train_targets
    )

    # Convert train, test, and validation features to torch tensors
    train_features = torch.tensor(train_features, dtype=torch.float32)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    val_features = torch.tensor(val_features, dtype=torch.float32)

    # Convert string labels to integer class indices
    le = LabelEncoder() # Initialize the LabelEncoder
    train_targets = le.fit_transform(train_targets) # Fit and transform training targets
    test_targets = le.transform(test_targets) # Transform test targets
    val_targets = le.transform(val_targets) # Transform validation targets

    # Convert train, test, and validation targets to torch tensors
    train_targets = torch.tensor(train_targets, dtype=torch.long)
    test_targets = torch.tensor(test_targets, dtype=torch.long)
    val_targets = torch.tensor(val_targets, dtype=torch.long)

    # Reshape inputs to (B, C, H, W) format for PyTorch (B=batch size, C=channels, H=height, W=width)
    train_features = train_features.permute(0, 3, 1, 2)  # (B, H, W, C) to (B, C, H, W)
    test_features = test_features.permute(0, 3, 1, 2)  # (B, H, W, C) to (B, C, H, W)
    val_features = val_features.permute(0, 3, 1, 2)  # (B, H, W, C) to (B, C, H, W)
    # Print shapes of the datasets for verification
    print(f"Train features shape: {train_features.shape}, Train targets shape: {train_targets.shape}")
    print(f"Validation features shape: {val_features.shape}, Validation targets shape: {val_targets.shape}")
    print(f"Test features shape: {test_features.shape}, Test targets shape: {test_targets.shape}")
    
    # Create TensorDataset for training, validation, and testing datasets (using features and targets)
    # Use the TensorDataset objects to create DataLoader objects for training, validation, and testing datasets
    train_dataset = TensorDataset(train_features, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(val_features, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataset = TensorDataset(test_features, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    ### TRAINING THE MODEL
    # Create the model and define hyperparameters
    model = Callclassifier()  # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available and set device accordingly
    model.to(device) # Move the model to the device (GPU or CPU)
    loss_func = nn.CrossEntropyLoss() # Define the loss function (CrossEntropyLoss)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Initialize the optimizer (Adam) with the model parameters and learning rate

    # Identify tracked values, creating lists in which to store values
    # NOTE: These lists store values after every BATCH, not only every EPOCH
    train_loss_list = [] # training loss             
    val_acc_list = [] # validation accuracy
    val_loss_list = [] # validation loss

    # Number of epochs for training
    num_epochs = 20

    # Learning Rate Scheduler (adjusts learning rate during training to improve convergence)
    # Reduces the learning rate by a factor of 0.5 every 3 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    ### TRAINING LOOP
    # Loop over number of epochs
    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        running_train_loss = 0.0       # ← initialize before the batch loop
        num_train_batches = 0          # ← initialize before the batch loop

        # Loop over batches of training data
        for xb, yb in train_loader:
            
            output = model(xb)  # Forward pass: compute model predictions
            loss = loss_func(output, yb) # Compute the loss between predictions and true labels
            
            optimizer.zero_grad()  # Clear gradients from the previous step
            loss.backward()  # Backward pass: compute gradients
            optimizer.step()  # Update model parameters using the optimizer

            running_train_loss += loss.item()  # Get the training loss for this batch
            num_train_batches += 1  # Increment the number of training batches

            # Compute average training loss for this epoch
            avg_train_loss = running_train_loss / num_train_batches  # Total training loss SO FAR divided by number of batches SO FAR
            train_loss_list.append(avg_train_loss)  # Append average training loss to the list
        
        
        # Prepares the model for validation, initializing variables for validation loss and accuracy
        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0 # ← initialize before the batch loop
        correct = 0 # Initialize correct predictions counter
        total = 0 # Initialize total predictions counter
        num_val_batches = 0 # Counter for number of validation batches

        # For every epoch, loop over batches of validation data
        with torch.no_grad(): # Do not compute gradients during validation
            # Loop over batches of validation data
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device) # Move data to the device (GPU or CPU)
                outputs = model(xb) # Forward pass: compute model predictions
                loss = loss_func(outputs, yb) # Compute the loss between predictions and true labels

                running_val_loss += loss.item() # Add the validation loss for this batch to the running total
                num_val_batches += 1 # Increment the number of validation batches

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1) # Get the predicted class indices
                total += yb.size(0) # Update total number of samples
                correct += (predicted == yb).sum().item() # Add prediction to correct total if it is correct

        # Compute average validation loss and accuracy
        avg_val_loss = running_val_loss / num_val_batches  # Total validation loss SO FAR divided by number of batches SO FAR
        val_accuracy = 100 * correct / total if total > 0 else 0 # Calculate validation accuracy as a percentage, 0 if no samples (to avoid division by zero)

        val_loss_list.append(avg_val_loss) # Append average validation loss to the list
        val_acc_list.append(val_accuracy) # Append validation accuracy to the list

        scheduler.step()  # Step the learning rate scheduler to adjust the learning rate

        # Print training and validation statistics for the current epoch before moving to the next epoch
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )


    # Save the trained model in a .pth file
    torch.save(model.state_dict(), 'call_classifier.pth')



    ### TESTING THE MODEL
    # Test the model on the testing dataset
    model.eval()  # Set the model to evaluation mode
    correct = 0 # Initialize correct predictions counter
    total = 0 # Initialize total predictions counter
    with torch.no_grad():  # Disable gradient computation for testing
        for xb, yb in test_loader:
            outputs = model(xb)  # Forward pass: compute model predictions
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class indices
            total += yb.size(0)  # Update total number of samples
            correct += (predicted == yb).sum().item()  # Count correct predictions
    test_accuracy = 100 * correct / total if total > 0 else 0  # Calculate test accuracy
    print(f"Test Accuracy: {test_accuracy:.2f}%")  # Print the test accuracy



    ############ Show examples of model predictions #################################
    model.eval()
    num_examples = 5

    # random examples, can sample random indices:
    sample_ids = random.sample(range(len(test_dataset)), num_examples)

    images = [] # List to store sampled images
    true_labels = [] # List to store true labels of sampled images
    # Loop through the sampled indices to get images and true labels
    for idx in sample_ids:
        img, true_idx = test_dataset[idx]
        images.append(img)               # img shape: (3, 64, 64)
        true_labels.append(true_idx.item())
    
    # Run these images through the model to get predicted IDs
    imgs_tensor = torch.stack(images).to(device)  # create torch tensor for images, shape (N, 3, 64, 64)
    with torch.no_grad(): # Disable gradient computation
        outputs = model(imgs_tensor) # feed images tensor into the model to generate predictions, shape (N, 20)
        _, pred_ids = torch.max(outputs, dim=1) # # Get the predicted class indices (IDs) from the model outputs, shape (N,)

    pred_ids = pred_ids.cpu().numpy().tolist() # Convert predictions to a list

    # Convert indices back to calligrapher names, using inverse transform of the LabelEncoder created earlier
    true_names = le.inverse_transform(true_labels) # turn true integers into an array of strings
    pred_names = le.inverse_transform(pred_ids) # turn predicted integers into an array of strings

    # Plot each of the sampled images with titles, showing true and predicted names
    plt.figure(figsize=(12, 6)) # create figure
    for i in range(num_examples):  # Loop through the number of examples
        ax = plt.subplot(1, num_examples, i+1) # Create a subplot for each image
        img = images[i].cpu().permute(1, 2, 0).numpy() # Convert image tensor to numpy array and permute dimensions to (H, W, C)
        ax.imshow(img.astype(np.uint8)) # Display the image
        ax.axis("off") # Disable axis
        ax.set_title(f"True: {true_names[i]}\nPred: {pred_names[i]}", fontsize=10) # Titles
    plt.tight_layout()
    plt.show() # Display the plot

    ####################################################
    # Visualize training loss and validation accuracy  #
    ####################################################

    plt.figure(figsize=(15, 9)) # create figure

    # Plot training loss of the model versus number of BATCHES
    plt.subplot(2, 1, 1)
    plt.plot(train_loss_list, linewidth=3)
    plt.ylabel("Training Loss")
    plt.xlabel("Batch")
    plt.title("Average Training Loss per Batch")
    sns.despine() # Remove top and right spines from the plot for better aesthetics

    # Plot validation accuracy of the model versus number of EPOCHS
    plt.subplot(2, 1, 2)
    plt.plot(val_acc_list, linewidth=3)
    plt.ylabel("Validation Accuracy (%)")
    plt.xlabel("Epoch")
    plt.title("Validation Accuracy per Epoch")
    sns.despine() # Same as previous plot

    plt.tight_layout()
    plt.savefig('training_validation_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Plot saved as 'training_validation_plot.png'.")

   
if __name__ == "__main__":
    main()  # Run the main function to prepare data, train the model, and visualize results


