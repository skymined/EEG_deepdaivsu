import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# print(f"Loaded data from {folder_path}:")
# print(f"Feature shape: {feature.shape}")
# print(f"Label shape: {label.shape}")
# print(f"Cumulative samples shape: {cumulative_samples.shape}")

# print(f"Feature shape: {sample_feature.shape}")
# print(f"Label shape: {sample_label.shape}")

class Conv1DEncoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv1DEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, 
                               out_channels=output_channels, 
                               kernel_size=2, 
                               stride=2)
        self.conv2 = nn.Conv1d(in_channels=output_channels, 
                               out_channels=output_channels, 
                               kernel_size=2, 
                               stride=2)
        self.conv3 = nn.Conv1d(in_channels=output_channels, 
                               out_channels=output_channels, 
                               kernel_size=2, 
                               stride=2)
        self.relu = nn.ReLU()
        self.avgPool = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, x, position):
        x = self.conv1(x)
        x1 = self.relu(x)
        pos1 = self.avgPool(position)
        embed_pos1 = pos1.repeat(x.shape[0], x1.shape[1], 1)
        
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        pos2 = self.avgPool(pos1)
        embed_pos2 = pos2.repeat(x.shape[0], x2.shape[1], 1)

        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        pos3 = self.avgPool(pos2)
        embed_pos3 = pos3.repeat(x.shape[0], x3.shape[1], 1)

        return x1+embed_pos1, x2+embed_pos2, x3+embed_pos3

class Embed_CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, channel):
        super(Embed_CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.mha2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.relu = nn.ReLU()
        self.avgPool = nn.AvgPool1d(kernel_size=channel, stride=channel)
    
    def forward(self, x1, x2, x3):
        x1_pad = x1.transpose(1, 2)
        x1_pad = F.pad(x1_pad, (0, 0, 0, x1_pad.shape[1]))
        x1_pad = x1_pad.transpose(0, 1)

        x2_pad = x2.transpose(1, 2).transpose(0, 1)
        x3_pad = x3.transpose(1, 2).transpose(0, 1)

        att_output1, _ = self.mha(x2_pad, x1_pad, x1_pad)

        att_output1 = att_output1.transpose(0, 1)
        att_output1 = F.pad(att_output1, (0, 0, 0, att_output1.shape[1]))
        att_output1 = att_output1.transpose(0, 1)

        att_output2, _ = self.mha2(x3_pad, att_output1, att_output1)

        att_output2 = att_output2.transpose(1, 2)
        output = self.avgPool(att_output2)

        return output

class Classifier_per_time(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(Classifier_per_time, self).__init__()
        self.linear1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim // 4, kernel_size=1)
        self.linear2 = nn.Conv1d(in_channels=embed_dim // 4, out_channels=3, kernel_size=1)
    
    def forward(self, x):
        output1 = x.permute(1, 2, 0)

        output2 = self.linear1(output1)

        output3 = self.linear2(output2)

        output3 = output3.transpose(1, 2)
        output = F.softmax(output3, dim=-1) # after top k

        return output

# 5, 64, 62, 3, 8, 25
class Baseline_model(nn.Module):
    def __init__(self, in_channel, out_channel, input_dim, final_dim, num_heads, topk):
        super(Baseline_model, self).__init__()
        self.encoder = Conv1DEncoder(in_channel, out_channel)
        self.eca = Embed_CrossAttention(out_channel, num_heads, input_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=out_channel, num_heads=num_heads)
        self.classifier = Classifier_per_time(embed_dim=out_channel, output_dim=final_dim)

    def forward(self, x):
        positional_embedding = torch.arange(1, x.shape[2] + 1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], -1)
        x = x.transpose(1, 2)
        
        embed1, embed2, embed3 = self.encoder(x, positional_embedding)
        final_embed = self.eca(embed3, embed2, embed1)

        final_embed = final_embed.transpose(1, 2)
        sf_attention_output, _ = self.self_attention(final_embed, final_embed, final_embed)

        class_output = self.classifier(sf_attention_output)

        max_values, _ = class_output.max(dim=-1)
        topk_values, topk_indices = torch.topk(max_values, 25, dim=1)
        top_25_vectors = class_output[:, topk_indices[0]]
        final_output = top_25_vectors.mean(dim=1)

        return final_output
