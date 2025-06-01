import numpy as np
import torch
from IPython.display import Image
import os
import seaborn as sns
from load_data import CalligData
from train import Callclassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import tqdm

def main():
    # Load model
    model = Callclassifier()
    model.load_state_dict(torch.load("call_classifier.pth"))
    model.eval()
    print("Model loaded successfully.")

    # Load testing data
    data = CalligData(base_dir='archive/data/data')
    data.load_data(only_test=True)  # Load only test data
    # Get test features and targets
    test_features = data.get_test_features()
    test_targets = data.get_test_targets()

    test_features = np.array(test_features) # reshape the testing features to flatten the images
    test_features = torch.tensor(test_features, dtype=torch.float32)
    le = LabelEncoder()
    test_targets = le.transform(test_targets)
    test_targets = torch.tensor(test_targets, dtype=torch.long)
    test_features = test_features.permute(0, 3, 1, 2) # Reshape to (N, C, H, W) format for PyTorch
    test_dataset = TensorDataset(test_features, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Run model on test data

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

    # Visualize training loss and validation accuracy

    plt.figure(figsize = (15, 9))

    plt.subplot(2, 1, 1)
    plt.plot(train_loss_list, linewidth = 3)
    plt.ylabel("training loss")
    plt.xlabel("epochs")
    sns.despine()

    plt.subplot(2, 1, 2)
    plt.plot(validation_accuracy_list, linewidth = 3, color = 'gold')
    plt.ylabel("validation accuracy")
    plt.xlabel("epochs")
    sns.despine()

    # Export the plot to a file
    plt.tight_layout()
    plt.show()
    plt.savefig('training_validation_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'training_validation_plot.png'.")

if __name__ == "__main__":
    main()  # Run the main function to test the model