# MNIST Image Classification 🖼️

Note: ReadME needs to be updated to include DCGANs.

This repository contains code for training, evaluating, and deploying a convolutional neural network (CNN) model for image classification on the MNIST dataset using PyTorch.

## Prerequisites 🌎

- Python 3.6 or above
- PyTorch
- torchvision
- scikit-learn
- seaborn
- matplotlib
- boto3
- sagemaker

## Installation 🎩

1. Clone the repository:
```shell
git clone https://github.com/Sid1279/MNIST-CNN.git
cd mnist-cnn
```
2. Install the required Python packages:
```shell
pip install -r requirements.txt
```
3. Rename your endpoint and bucket to your desired values in the deployment notebook (deploy.ipynb).

4. Create a notebook instance in AWS SageMaker for deployment.

5. Upload the `MNIST PyTorch CNN.ipynb` notebook to SageMaker.

6. Run all cells in the notebook to deploy the trained model to Amazon SageMaker.

## How does it work 🤳🏽
This notebook demonstrates the process of training a Convolutional Neural Network (CNN) on the MNIST dataset and deploying the trained model using Amazon SageMaker.

1. Data Preparation: The script downloads the MNIST dataset and applies data transformations such as normalization and tensor conversion using torchvision.transforms. It creates custom Dataset objects and uses DataLoader to efficiently load and iterate over the training and test data.

2. Model Architecture: The script defines a CNN model using the NeuralNetwork class, which inherits from torch.nn.Module. The model consists of convolutional layers, activation functions, pooling layers, and fully connected layers. The forward pass method implements the model's computation flow.

3. Model Training: The script initializes the model, defines the loss function (cross-entropy loss), and sets up the optimizer (Adam optimizer). It then enters a loop over the specified number of epochs and performs the following steps for each epoch:
   - Iterates over the training data, computes the forward pass, calculates the loss, and performs backpropagation to update the model's parameters.
   - Computes training accuracy and loss and logs them to Tensorboard using SummaryWriter.
   - Evaluates the model on the test data, computes testing accuracy and loss, and logs them to Tensorboard.
    
4. Evaluation: After training, the script generates a confusion matrix to visualize the model's performance on the training set using sklearn.metrics.confusion_matrix and seaborn.heatmap. The confusion matrix provides insights into the model's ability to correctly classify different digits.

5. Model Deployment: The script saves the trained model's state dictionary using torch.save and creates a tar.gz archive containing the model file. It uploads the archive to an S3 bucket using the AWS SDK (boto3). It then uses the SageMaker Python SDK to create a PyTorchModel object, specifying the model data location, IAM role, and other necessary information. The model is deployed to an endpoint using deploy method, specifying the endpoint name and instance type.



