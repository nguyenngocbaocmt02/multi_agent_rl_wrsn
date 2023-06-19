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

def layer_init(layer, std=0.1, bias_const=0.):
    if isinstance(layer, nn.Conv2d):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    elif isinstance(layer, nn.Module):
        for param in layer.parameters():
            if param.dim() > 1:
                nn.init.orthogonal_(param)
            else:
                nn.init.constant_(param, bias_const)
    return layer
