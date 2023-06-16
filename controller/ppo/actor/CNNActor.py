import torch
import torch.nn as nn
from utils import layer_init
# Define the CNN architecture
class CNNActor(nn.Module):
    def __init__(self):
        super(CNNActor, self).__init__()
        
        # Convolutional layer 1: input (100, 100, 3), output (48, 48, 16)
        self.conv1 = layer_init(nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=2, padding=2))
        self.relu1 = nn.ReLU()
        
        # Convolutional layer 2: input (48, 48, 16), output (22, 22, 32)
        self.conv2 = layer_init(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2))
        self.relu2 = nn.ReLU()
        
        # Convolutional layer 3: input (22, 22, 32), output (13, 13, 64)
        self.conv3 = layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2))
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        # Fully connected layer: input (10*10*64), output 100
        self.fc1 = layer_init(nn.Linear(in_features=13*13*64, out_features=100))
        self.relu4 = nn.ReLU()
        self.fc2 = layer_init(nn.Linear(in_features=13*13*64, out_features=100))
        self.relu5 = nn.ReLU()
        self.mean_layer = layer_init(nn.Linear(100, 3), std=0.1)
        self.log_std_layer = layer_init(nn.Linear(100, 3), std = 0.1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Pass the input through the layers of the network
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)

        x1 = self.fc1(x)
        x1 = self.relu4(x1)
        mean = self.mean_layer(x1)
        mean = self.tanh(mean)

        x2 = self.fc1(x)
        x2 = self.relu4(x2)
        log_std = self.log_std_layer(x2)
        log_std = self.tanh(log_std)
        
        return mean, log_std