#Imports
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
# from torchview import draw_graph


### Model Architecture
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

### Model Definition
class Callclassifier(nn.Module):
    def __init__(self):
        super(Callclassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(self.relu(x))
        return x
    

def main():
    # LOAD AND PREPARE DATA
    # Load data
    data = CalligData(base_dir='archive/data/data')
    data.load_data()
    # train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset = data.PrepareData(valid_size=0.2, batch_size=32)

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


    ## TRAINING THE MODEL
    ###Define Hyperparameters
    model = Callclassifier()  # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Vizualize the model architecture
    
    # model_graph = draw_graph(model, input_size=[(3, 64, 64)],  # adjust size if needed
    #                      expand_nested=True,
    #                      device='cpu')  # or 'cuda' if using GPU
    # model_graph.render("callclassifier_diagram", format="png")

    # Identify Tracked Values
    train_loss_list = []             
    validation_accuracy_list = [] 

    ### Training Loop
    import tqdm
    num_epochs = 5

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        #Training loop
        for xb, yb in train_loader:  # Iterate over batches of training data
            
            output = model(xb)  # Forward pass: compute model predictions
            loss = loss_func(output, yb) # Compute the loss between predictions and true labels
            
            optimizer.zero_grad()  # Clear gradients from the previous step
            loss.backward()  # Backward pass: compute gradients
            optimizer.step()  # Update model parameters using the optimizer

        # Get the training loss for this epoch
        train_loss_list.append(loss.item())  # Store the training loss for this batch
            # train_loss_list.append(loss.item()) # Store the training loss for this batch
        
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

        scheduler.step()  # Step the learning rate scheduler

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {np.mean(train_loss_list):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")


    # Save the trained model
    torch.save(model.state_dict(), 'call_classifier.pth')  # Save the model state dictionary


    # Test the model on the testing dataset
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation for testing
        for xb, yb in test_loader:
            outputs = model(xb)  # Forward pass: compute model predictions
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class indices
            total += yb.size(0)  # Update total number of samples
            correct += (predicted == yb).sum().item()  # Count correct predictions
    test_accuracy = 100 * correct / total if total > 0 else 0  # Calculate test accuracy
    print(f"Test Accuracy: {test_accuracy:.2f}%")  # Print the test accuracy

    ############ Show example of the model #################################
    model.eval()
    num_examples = 5

    # random examples, can sample random indices:
    import random
    sample_ids = random.sample(range(len(test_dataset)), num_examples)

    images = []
    true_labels = []
    for idx in sample_ids:
        img, true_idx = test_dataset[idx]
        images.append(img)               # img shape: (3, 64, 64)
        true_labels.append(true_idx.item())
    
    # 2. Run them through the model to get predicted IDs

    imgs_tensor = torch.stack(images).to(device)            # shape (N, 3, 64, 64)
    with torch.no_grad():
        outputs = model(imgs_tensor)                        # shape (N, 20)
        _, pred_ids = torch.max(outputs, dim=1)             # shape (N,)

    pred_ids = pred_ids.cpu().numpy().tolist()

    # 3. Convert indices back to calligrapher names
    true_names = le.inverse_transform(true_labels)           # array of strings
    pred_names = le.inverse_transform(pred_ids)              # array of strings

    # 4. Plot each of the sampled images with titles
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    for i in range(num_examples):
        ax = plt.subplot(1, num_examples, i+1)
        img = images[i].cpu().permute(1, 2, 0).numpy()      # (64, 64, 3)
        ax.imshow(img.astype(np.uint8))
        ax.axis("off")
        ax.set_title(f"True: {true_names[i]}\nPred: {pred_names[i]}", fontsize=10)
    plt.tight_layout()
    plt.show()

    ####################################################

    # Visualize training loss and validation accuracy

    plt.figure(figsize=(15, 9))

    # Training loss
    plt.subplot(2, 1, 1)
    plt.plot(train_loss_list, linewidth=3)
    plt.ylabel("Training Loss")
    plt.xlabel("Epochs")
    plt.title("Training Loss per Epoch")
    sns.despine()

    # Validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(validation_accuracy_list, linewidth=3, color='gold')
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.title("Validation Accuracy per Epoch")
    sns.despine()

    plt.tight_layout()
    plt.savefig('training_validation_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Plot saved as 'training_validation_plot.png'.")


if __name__ == "__main__":
    main()  # Run the main function to train the model


