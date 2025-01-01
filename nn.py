from torch import nn

def PhiNN(input_size, hidden_size, output_size):
    layers = [
        nn.utils.spectral_norm(nn.Linear(input_size, hidden_size)),
        nn.ReLU(),
        nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU(),
        nn.utils.spectral_norm(nn.Linear(hidden_size, output_size)),
    ]
    return nn.Sequential(*layers)