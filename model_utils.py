import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Utility functions for preprocessing
def parse_size(size_str):
    if 'x' in size_str:
        dimensions = size_str.strip().split('x')
        return float(dimensions[0]) * float(dimensions[1])
    elif size_str == 'Non':
        return 0
    else:
        return float(size_str)

def parse_ratio(ratio_str):
    values = ratio_str.strip().split(':')
    return float(values[0]) / float(values[1])

def normalize_feature(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def denormalize_feature(normalized_value, min_val, max_val):
    return normalized_value * (max_val - min_val) + min_val

# The PyTorch model class
class LifetimePredictionModel(nn.Module):
    def __init__(self, num_solder_types, num_led_names, num_submounts, num_pad_counts, numerical_input_size, embedding_dim=8):
        super(LifetimePredictionModel, self).__init__()
        # Embedding layers
        self.solder_embedding = nn.Embedding(num_solder_types, embedding_dim)
        self.led_embedding = nn.Embedding(num_led_names, embedding_dim)
        self.submount_embedding = nn.Embedding(num_submounts, embedding_dim)
        self.pad_count_embedding = nn.Embedding(num_pad_counts, embedding_dim)

        # Fully connected layers
        total_input_size = embedding_dim * 4 + numerical_input_size
        self.fc1 = nn.Linear(total_input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)

    def forward(self, solder_input, led_input, submount_input, pad_count_input, numerical_input):
        # Embed the categorical inputs
        solder_embedded = self.solder_embedding(solder_input).squeeze(1)
        led_embedded = self.led_embedding(led_input).squeeze(1)
        submount_embedded = self.submount_embedding(submount_input).squeeze(1)
        pad_count_embedded = self.pad_count_embedding(pad_count_input).squeeze(1)

        # Concatenate embeddings and numerical inputs
        x = torch.cat([solder_embedded, led_embedded, submount_embedded, pad_count_embedded, numerical_input], dim=1)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        output = self.output_layer(x)
        return output
