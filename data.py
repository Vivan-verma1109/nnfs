import os
import cv2
from zipfile import ZipFile
import matplotlib as plt
import numpy as np
import urllib
import urllib.request

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

labels = os.listdir("fashion_mnist_images/train")

X = []
y = []

for label in labels:
    for file in os.listdir(os.path.join("fashion_mnist_images", 'train', label)):
        image = cv2.imread(os.path.join("fashion_mnist_images", 'train', label), cv2.IMREAD_UNCHANGED)        
        X.append(image)
        y.append(label)