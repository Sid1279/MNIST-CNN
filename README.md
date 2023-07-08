# MNIST Image Classification

This repository contains code for training, evaluating, and deploying a convolutional neural network (CNN) model for image classification on the MNIST dataset using PyTorch.

## Prerequisites

- Python 3.6 or above
- PyTorch
- torchvision
- scikit-learn
- seaborn
- matplotlib
- boto3
- sagemaker

## Installation

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


