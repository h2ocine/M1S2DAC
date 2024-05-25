import numpy as np
import os
import gzip
import urllib.request

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz", 
        "train-labels-idx1-ubyte.gz", 
        "t10k-images-idx3-ubyte.gz", 
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    for file in files:
        print(f"Downloading {file}...")
        urllib.request.urlretrieve(base_url + file, file)
    
    print("Download complete.")
    
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            return data.reshape(-1, 28*28) / 255.0

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)
    
    train_images = load_images("train-images-idx3-ubyte.gz")
    train_labels = load_labels("train-labels-idx1-ubyte.gz")
    test_images = load_images("t10k-images-idx3-ubyte.gz")
    test_labels = load_labels("t10k-labels-idx1-ubyte.gz")
    
    return (train_images, train_labels), (test_images, test_labels)

(train_images, train_labels), (test_images, test_labels) = download_mnist()