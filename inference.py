import subprocess

subprocess.check_call(['pip', 'install', 'boto3'])
subprocess.check_call(['pip', 'install', 'torch'])
subprocess.check_call(['pip', 'install', 'sagemaker'])

import cv2  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import json
import os
import boto3
import tarfile

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_and_preprocess_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (32, 32))
    inference_data = torch.from_numpy(np.transpose(resized_image, (2, 0, 1))).float()
    return inference_data


def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'MNIST.tar.gz')
    with tarfile.open(model_path, 'r:gz') as tar:
        tar.extract('MNIST.pth')
    model = NeuralNetwork()
    model.load_state_dict(torch.load('MNIST.pth'))
    return model

def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    s3_image_key = json.loads(request_body)['s3_key']
    s3_client = boto3.client('s3')
    s3_client.download_file('sid-test-bucket-1279', s3_image_key, s3_image_key)
    inference_data = load_and_preprocess_image(s3_image_key)
    return inference_data

def predict_fn(input_object, model):
    output = model(input_object.unsqueeze(0))
    _, predicted_label = torch.max(output, 1)
    predicted_label = predicted_label.item()
    return predicted_label


def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    print("Predicted Number:", predictions)
    payload = json.dumps({"Predicted Number:", predictions})
    return payload
