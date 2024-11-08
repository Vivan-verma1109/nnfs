import os
import cv2
import numpy as np
import nnfs

nnfs.init()

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []
    
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
        X.append(image)
        y.append(label)
    return np.array(X), np.array(y).astype("uint8")

def create_dataset_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    return X, y, X_test, y_test

X, y, X_test, y_test = create_dataset_mnist('fashion_mnist_images')

X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

X = X.reshape(X.shape[0], -1 )
X_test = X_test.reshape(X.shape[0], -1)

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
