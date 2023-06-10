import torch
import torch.nn as nn

# Define the CNN architecture
class CNNActor(nn.Module):
    def __init__(self):
        super(CNNActor, self).__init__()
        
        # Convolutional layer 1: input (100, 100, 3), output (48, 48, 16)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        
        # Convolutional layer 2: input (48, 48, 16), output (22, 22, 32)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        
        # Convolutional layer 3: input (22, 22, 32), output (13, 13, 64)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.relu3 = nn.ReLU()
        
        # Fully connected layer: input (10*10*64), output 100
        self.fc1 = nn.Linear(in_features=13*13*64, out_features=100)
        self.relu4 = nn.ReLU()
        self.mean_layer = nn.Linear(100, 3)
        self.log_std_layer = nn.Linear(100, 3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Pass the input through the layers of the network
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        mean = self.mean_layer(x)
        mean = self.sigmoid(mean)
        log_std = self.log_std_layer(x)
        log_std = self.sigmoid(log_std)
        return mean, log_std