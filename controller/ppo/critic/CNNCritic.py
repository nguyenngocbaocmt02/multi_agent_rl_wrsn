import torch
import torch.nn as nn
import numpy as np
from utils import layer_init

# Define the CNN architecture
class CNNCritic(nn.Module):
    def __init__(self):
        super(CNNCritic, self).__init__()
        
        # Convolutional layer 1: input (100, 100, 3), output (48, 48, 16)
        self.conv1 = layer_init(nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=2, padding=2))
        self.relu1 = nn.ReLU(inplace=True)
        
        # Convolutional layer 2: input (48, 48, 16), output (22, 22, 32)
        self.conv2 = layer_init(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2))
        self.relu2 = nn.ReLU(inplace=True)
        
        # Convolutional layer 3: input (22, 22, 32), output (10, 10, 64)
        self.conv3 = layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2))
        self.relu3 = nn.ReLU(inplace=True)
        
        # Flatten the output of the convolutional layers
        self.flatten = nn.Flatten()
        
        # Fully connected layer: input , output 100
        self.fc1 = layer_init(nn.Linear(in_features=10816, out_features=100))
        self.relu4 = nn.ReLU(inplace=True)
        
        # Output layer: input 100, output 1
        self.fc2 = layer_init(nn.Linear(in_features=100, out_features=1), std=1.0)
        
    def forward(self, x):
        # Pass the input through the layers of the CNN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        # Concatenate the output of the CNN with the input vector
        
        # Pass the concatenated input through the fully connected layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        
        return x