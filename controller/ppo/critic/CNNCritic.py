import torch
import torch.nn as nn

# Define the CNN architecture
class CNNCritic(nn.Module):
    def __init__(self):
        super(CNNCritic, self).__init__()
        
        # Convolutional layer 1: input (100, 100, 3), output (48, 48, 16)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        
        # Convolutional layer 2: input (48, 48, 16), output (22, 22, 32)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        
        # Convolutional layer 3: input (22, 22, 32), output (10, 10, 64)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.relu3 = nn.ReLU()
        
        # Flatten the output of the convolutional layers
        self.flatten = nn.Flatten()
        
        # Fully connected layer: input (10*10*64 + 3), output 100
        self.fc1 = nn.Linear(in_features=10816, out_features=100)
        self.relu4 = nn.ReLU()
        
        # Output layer: input 100, output 1
        self.fc2 = nn.Linear(in_features=100, out_features=1)
        
    def forward(self, x1):
        # Pass the input through the layers of the CNN
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv3(x1)
        x1 = self.relu3(x1)
        x1 = self.flatten(x1)
        
        # Concatenate the output of the CNN with the input vector
        
        # Pass the concatenated input through the fully connected layers
        x1 = self.fc1(x1)
        x1 = self.relu4(x1)
        x1 = self.fc2(x1)
        
        return x1