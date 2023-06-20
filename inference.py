"""
The script loads an image, passes it through a pre-trained model, and 
prints the predicted label, actual label, and accuracy of the model's 
inference result.

Preconditions:

The image file "plane.jpg" exists and contains the image data to be inferred.
The trained model's checkpoint has been saved in a file named "trained_model.pt".
The neuralnetwork.py file exists and contains the NeuralNetwork class used in the code.
The required libraries (cv2, numpy, torch) are installed and accessible.

Postconditions:

The image is loaded and preprocessed by resizing and converting it to the RGB color space.
The trained model is loaded from the checkpoint and its state dictionary is modified accordingly.
The image is passed through the model for inference, and the predicted label is obtained.
The predicted label, actual label, and accuracy are printed to the console.

"""


import cv2  # type: ignore
import numpy as np
import torch
from neuralnetwork import CIFARClassifier
import os


def load_and_preprocess_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (32, 32))
    inference_data = torch.from_numpy(np.transpose(resized_image, (2, 0, 1))).float()
    return inference_data


folder_name = "artifacts"
# Load and preprocess the image
inference_data = load_and_preprocess_image(os.path.join(folder_name, "plane.jpg"))

# Load the trained model
model = CIFARClassifier()
model.load_state_dict(torch.load(os.path.join(folder_name, "trained_model.pt")))

# Perform inference
output = model(inference_data.unsqueeze(0))
_, predicted_label = torch.max(output, 1)
predicted_label = predicted_label.item()

classes = ["Plane", "Ship"]

# Print the predicted label
print("Predicted Label:", classes[predicted_label])
