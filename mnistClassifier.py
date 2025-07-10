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

def manual_mnist_donwload():
    import gzip
    import pickle

    os.makedirs('./data/MNIST/raw', exist_ok=True)
    urls = {
        'train-images-idx3-ubyte.gz': 'https://github.com/pytorch/vision/raw/main/torchvision/datasets/utils.py',
        'train-labels-idx1-ubyte.gz': 'https://github.com/pytorch/vision/raw/main/torchvision/datasets/utils.py',
        't10k-images-idx3-ubyte.gz': 'https://github.com/pytorch/vision/raw/main/torchvision/datasets/utils.py',
        't10k-labels-idx1-ubyte.gz': 'https://github.com/pytorch/vision/raw/main/torchvision/datasets/utils.py'
    }
    
    print("manual download")

def generate_dummy_data():
    from torch.utils.data import TensorDataset

    # dummy ddata
    train_images = torch.randn(60000, 1, 28, 8)
    train_labels = torch.randint(0, 10, (60000,))
    test_images = torch.randn(10000, 1, 28, 8)
    test_labels = torch.randint(0, 10, (10000,))

    train_datasets = TensorDataset(train_images,train_labels)
    test_datasets = TensorDataset(test_images, test_labels)

    print("dummy data for testing purposes")
    return train_datasets, test_datasets


