"""
This script loads the necessary data and models, evaluates the testing accuracy, and generates visualizations for specified images.
It provides ak analysis of the model's performance and displays visuaizations of the model's accuracies through heatmaps
and overlayed images.

Preconditions:
The CIFAR-10 dataset has been preprocessed and saved in a pickled file named "cifar_2class_py2.p".
The trained model's checkpoint has been saved in a file named "trained_model.pt".
The training accuracies for each epoch have been saved in a numpy array named "train_accuracies.npy".
The RGB images for visualization have been saved in a numpy array named "rgb_images.npy".
The heatmaps generated for the images have been saved in a numpy array named "heatmaps.npy".
The overlayed images (original images with heatmaps) have been saved in a numpy array named "overlayed_images.npy".

Postconditions:
The testing accuracy of the trained model on the CIFAR-2 test dataset is printed.
A plot is displayed showing the training and testing accuracies over the epochs.
Visualizations are displayed for a specified of images: the original image, heatmap, & overlayed image for each specified index.
"""


import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from neuralnetwork import CIFARClassifier
from CIFAR2Class import CIFAR2ClassDataset
import os
import plotting

# Loading data
folder_name = "artifacts"
data = pickle.load(
    open(os.path.join(folder_name, "cifar_2class_py2.p"), "rb"), encoding="bytes"
)

test_data = data[b"test_data"]
test_labels = data[b"test_labels"]

# Reshape into 32x32, 3 -> RGB
test_data = test_data.reshape(-1, 3, 32, 32)

# Convert numpy array into PyTorch tensors
test_data = torch.from_numpy(test_data).float()
test_labels = torch.from_numpy(test_labels[:, 0]).long()

# Load the trained model
model = CIFARClassifier()
model.load_state_dict(torch.load(os.path.join(folder_name, "trained_model.pt")))

train_accuracies = np.load(os.path.join(folder_name, "train_accuracies.npy"))
rgb_images = np.load(os.path.join(folder_name, "rgb_images.npy"))
heatmaps = np.load(os.path.join(folder_name, "heatmaps.npy"))
overlayed_images = np.load(os.path.join(folder_name, "overlayed_images.npy"))

# Create custom dataset that is compatible with PyTorch's DataLoader
test_dataset = CIFAR2ClassDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)


# Testing loop
def test_step(test_loader, model):
    test_total = 0
    test_correct = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * test_correct / test_total
    print(f"Testing Accuracy: {test_accuracy:.2f}%")
    return test_accuracy


test_accuracy = test_step(test_loader, model)
plotting.plot_train_test_accuracies(train_accuracies, test_accuracy)

# Select the subset of images to visualize - heatmaps
sample_indices = [1, 2, 3]
plotting.plot_heatmap_samples(rgb_images, heatmaps, overlayed_images, sample_indices)
