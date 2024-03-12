import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def print_mnist_samples(images, n_images):
    figure = plt.figure()
    num_of_images = n_images
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
        
