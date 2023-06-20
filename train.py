"""
This script trains a neural network model on the CIFAR-2 dataset, saves the trained model's state dictionary.
It generates GradCAM visualizations for a specific image in the dataset, and saves the visualizations as numpy arrays. 
It also calculates and saves the training accuracies for each epoch in an numpy array.

Preconditions:

The cifar_2class_py2.p file exists and contains the necessary data dictionary with the following keys: b'train_data', b'train_labels', b'test_data', and b'test_labels'.
The neuralnetwork.py, gradcam.py, and CIFAR2Class.py files exist and contain the required classes (NeuralNetwork, GradCAM, CIFAR2ClassDataset) used in the code.
The required libraries (torch, cv2, numpy) are installed and accessible.

Postconditions:

The model is trained for the specified number of epochs (num_epochs) using the CIFAR-10 dataset.
The trained model's state dictionary is saved to a file named 'trained_model.pt'.
GradCAM visualizations are generated for a specific image in the test dataset (image_index = 10) and saved as numpy arrays:
The RGB images are saved as 'rgb_images.npy'.
The heatmaps are saved as 'heatmaps.npy'.
The overlayed images (RGB images with overlaid heatmaps) are saved as 'overlayed_images.npy'.
The training accuracies for each epoch are stored in a numpy array and saved as 'train_accuracies.npy'.

"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import gradcam
from neuralnetwork import CIFARClassifier
from CIFAR2Class import CIFAR2ClassDataset
import os
import config


def load_and_preprocess_data(folder_name, filename):
    data = pickle.load(
        open(os.path.join(folder_name, filename), "rb"), encoding="bytes"
    )

    train_data = data[b"train_data"]
    train_labels = data[b"train_labels"]

    # Reshape into 32x32, 3 -> RGB
    train_data = train_data.reshape(-1, 3, 32, 32)

    # Convert numpy array into PyTorch tensors
    train_data = torch.from_numpy(train_data).float()
    train_labels = torch.from_numpy(train_labels[:, 0]).long()

    return train_data, train_labels


# training loop function
def train_loop(model, optimizer, criterion, data_loader):
    model.train()
    train_loss = 0.0
    train_total = 0
    train_correct = 0

    for images, labels in data_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_loss += loss.item() * images.size(0)

    train_loss /= len(data_loader.dataset)
    train_accuracy = 100.0 * train_correct / train_total

    return train_loss, train_accuracy


def train_model(
    train_data, train_labels, model, criterion, optimizer, num_epochs, batch_size
):
    # Create custom dataset that is compatible with PyTorch's DataLoader
    train_dataset = CIFAR2ClassDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_loop(
            model, optimizer, criterion, train_loader
        )

        train_accuracies.append(train_accuracy)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%"
        )

    return train_accuracies


# Specify the folder to save the results
folder_name = "artifacts"

# Load and preprocess data
train_data, train_labels = load_and_preprocess_data(folder_name, "cifar_2class_py2.p")

# Initialize the model & define hyperparameters
model = CIFARClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=config.learning_rate, momentum=config.momentum
)
num_epochs = config.num_epochs
batch_size = config.batch_size

# Train the model
train_accuracies = train_model(
    train_data, train_labels, model, criterion, optimizer, num_epochs, batch_size
)

# Save the trained model
torch.save(model.state_dict(), os.path.join(folder_name, "trained_model.pt"))

# Generate & Save visualizations and accuracies
gradcam_vis = gradcam.GradCAM(model, model.conv1)
rgb_images, heatmaps, overlayed_images = gradcam.generate_gradcam_visualizations(
    model, train_data, gradcam_vis
)
np.save(os.path.join(folder_name, "rgb_images.npy"), np.array(rgb_images))
np.save(os.path.join(folder_name, "heatmaps.npy"), np.array(heatmaps))
np.save(os.path.join(folder_name, "overlayed_images.npy"), np.array(overlayed_images))
np.save(os.path.join(folder_name, "train_accuracies.npy"), np.array(train_accuracies))
