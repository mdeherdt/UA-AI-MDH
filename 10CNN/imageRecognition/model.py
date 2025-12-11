import torch
import torch.nn as nn


class MyCNN(nn.Module):
    """
    Implement a small CNN for CIFAR-10.

    - Input: 3×32×32 images
    - Output: 10 classes (linear layer)

    Example architecture:
      Conv → ReLU → Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → FC → ReLU → FC
    """

    def __init__(self):
        super().__init__()
        # TODO: define layers here


    def forward(self, x):
        # TODO: implement forward pass
        # Flatten between convolutional and fully connected layers
        # You can use x = x.view(x.size(0), -1)

        return x
