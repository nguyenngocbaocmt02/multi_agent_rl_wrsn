import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn

def draw_heatmap_state(state):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    # Iterate over each layer and plot the corresponding heatmap
    for i, ax in enumerate(axes.flatten()):
        # Get the data for the current layer
        layer_data = state[i]
        # Plot the heatmap for the current layer
        heatmap = ax.imshow(layer_data, cmap='viridis')
        ax.set_title(f'Layer {i+1}')
        fig.colorbar(heatmap, ax=ax)
        # Adjust the spacing between subplots and display the figure
    plt.tight_layout()
    plt.show()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer