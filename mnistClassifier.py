import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# data preprocessing and loading
transform =transforms.Compose({
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081,)) #mnist mean and std
})

# loading datasets
def load_mnist_datasets():
    try:
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        return train_dataset, test_dataset
    
    except Exception as e:
        print(f"error downloading MNIST: {e}")
        print("Trying alternative download methods")

        try:
            datasets.MNIST.mirrors = [
                'https://ossci-datasets.s3.amazonaws.com/mnist/',
                'http://yann.lecun.com/exdb/mnist/'
            ]

            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)
            return train_dataset, test_dataset
    
        except Exception as e2:
            print(f"Alternative download failed: {e2}")
            print("Please manually download MNIST dataset or use the manual download function below")
            raise