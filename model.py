import numpy as np
import os
import torch
import torch.nn as nn


# 265 == best length

folder_path = "./ExtractedFeatures/de_movingAve/"
    
feature_path = os.path.join(folder_path, "feature.npy")
label_path = os.path.join(folder_path, "label.npy")
cumulative_samples_path = os.path.join(folder_path, "cumulative.npy")

feature = np.load(feature_path)
label = np.load(label_path)
cumulative_samples = np.load(cumulative_samples_path)

print(f"Loaded data from {folder_path}:")
print(f"Feature shape: {feature.shape}")
print(f"Label shape: {label.shape}")
print(f"Cumulative samples shape: {cumulative_samples.shape}")

sample_feature = feature[cumulative_samples[8]:cumulative_samples[9], :]
sample_label = label[cumulative_samples[8]:cumulative_samples[9], :]
print(f"Feature shape: {sample_feature.shape}")
print(f"Label shape: {sample_label.shape}")

class Simple1DEncoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Simple1DEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, 
                               out_channels=output_channels, 
                               kernel_size=2, 
                               stride=2)
        self.conv2 = nn.Conv1d(in_channels=input_channels, 
                               out_channels=output_channels, 
                               kernel_size=2, 
                               stride=2)
        self.conv3 = nn.Conv1d(in_channels=input_channels, 
                               out_channels=output_channels, 
                               kernel_size=2, 
                               stride=2)
        self.relu = nn.ReLU()
        self.avgPool = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, x, position):
        x = self.conv1(x)
        x1 = self.relu(x)
        pos1 = self.avgPool(position)
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        pos2 = self.avgPool(pos1)
        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        pos3 = self.avgPool(pos2)

        return x

sample_feature = np.swapaxes(sample_feature, 0, 1).reshape(1, 310, -1)
sample_feature = torch.tensor(sample_feature, dtype=torch.float32)

sequence_length = sample_feature.shape[2]
batch_size = 1
channels = 310

positional_embedding = torch.arange(1, sequence_length + 1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
positional_embedding = positional_embedding.repeat(batch_size, channels, 1)

input_channels = 310
output_channels = 310
encoder = Simple1DEncoder(input_channels, output_channels)

# Create a dummy input tensor with batch size 8 and length 32
dummy_input = torch.randn(8, 1, 32)

# Forward pass through the encoder
output = encoder(sample_feature, positional_embedding)

print("Output shape:", output.shape)