import torch
import torch.nn as nn

def get_activation_function(name):
    """gets activation function by its name."""

    activation_dict = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'gelu': nn.GELU(approximate='tanh'),
        'silu': nn.SiLU(),
        'softmax': nn.Softmax(dim=-1),
        'identity': nn.Identity(),
        # add more if needed
    }
    if name in activation_dict:
        return activation_dict[name]
    else:
        raise ValueError(f'Activation function {name} not found in {activation_dict.keys()}')
