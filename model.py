import numpy as np
import os
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.dropout(self.pe[:x.size(0)].to(x.device))
    
class MCA(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.u = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(-1)
        self.num_head = num_heads
        self.dropout = nn.Dropout(0.1)

    def forward(self, x1, x2, x3, m2=None):

        T1, B1, C1 = x1.shape  ## (66, 1984, 64)
        T2, B2, C2 = x2.shape  ## (66, 1984, 64)

        x1 = x1.transpose(0, 1)  ## (1984, 66, 64)
        x2 = x2.transpose(0, 1)  ## (1984, 66, 64)

        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)
        attn = q @ k.transpose(-1, -2) / np.sqrt(C1)
        if m2 is not None:
            m2 = m2.reshape(B2, 1, 1, 1, T2)
            m2 = torch.tile(m2, (1, N2, 1, 1, 1))
            m2 = m2.reshape(B2 * N2, 1, 1, T2)
            attn = attn.masked_fill(m2 == 0, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        x = attn @ v

        x = self.u(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)
        return x


class Conv1DEncoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv1DEncoder, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv3 = nn.Conv1d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=2,
            stride=2,
        )
        self.relu = nn.ReLU()
        self.avgPool = nn.AvgPool1d(kernel_size=2, stride=2)

    
    def forward(self, x, position, device):
        x = self.conv1(x)
        x1 = self.relu(x)
        pos1 = self.avgPool(position)
        embed_pos1 = pos1.repeat(x.shape[0], 1, 1).to(device) 
        
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        pos2 = self.avgPool(pos1)
        embed_pos2 = pos2.repeat(x.shape[0], 1, 1).to(device) 

        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        pos3 = self.avgPool(pos2)
        embed_pos3 = pos3.repeat(x.shape[0], 1, 1).to(device) 

        return x1 + embed_pos1, x2 + embed_pos2, x3 + embed_pos3


class MCA(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.u = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(-1)
        self.num_head = num_heads
        self.dropout = nn.Dropout(0.1)

    ## x3 넣어준 이유는 Embed_CrossAttention의 코드변경 최소화하기 위해서
    def forward(self, x1, x2, x3, m2=None):

        # B1, T1, N1, C1 = x1.shape
        # B2, T2, N2, C2 = x2.shape

        T1, B1, C1 = x1.shape  ## (66, 1984, 64)
        T2, B2, C2 = x2.shape  ## (66, 1984, 64)

        ## x1 = x1.transpose(1, 2).reshape(B1 * N1, T1, C1)
        ## x2 = x2.transpose(1, 2).reshape(B2 * N2, T2, C2)

        x1 = x1.transpose(0, 1)  ## (1984, 66, 64)
        x2 = x2.transpose(0, 1)  ## (1984, 66, 64)

        # q = self.q(x1).reshape(B1 * N1, T1, self.num_head, -1).transpose(1, 2)
        # k = self.k(x2).reshape(B2 * N2, T2, self.num_head, -1).transpose(1, 2)
        # v = self.v(x2).reshape(B2 * N2, T2, self.num_head, -1).transpose(1, 2)

        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)
        attn = q @ k.transpose(-1, -2) / np.sqrt(C1)
        if m2 is not None:
            m2 = m2.reshape(B2, 1, 1, 1, T2)
            m2 = torch.tile(m2, (1, N2, 1, 1, 1))
            m2 = m2.reshape(B2 * N2, 1, 1, T2)
            attn = attn.masked_fill(m2 == 0, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # x = (attn @ v).transpose(1, 2)
        x = attn @ v

        # x = x.reshape(B1, N1, T1, C1).transpose(1, 2)

        x = self.u(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)
        return x, None  ## att_output_weight는 더미값으로 줌 (코드 변경 최소화하려고)


class Embed_CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, channel):
        super(Embed_CrossAttention, self).__init__()
        self.mha = MCA(embed_dim=embed_dim, num_heads=num_heads)
        self.mha2 = MCA(embed_dim=embed_dim, num_heads=num_heads)
        self.relu = nn.ReLU()
        self.avgPool = nn.AvgPool1d(kernel_size=channel, stride=channel)

    def forward(self, x1, x2, x3):
        x1_pad = x1.transpose(1, 2)
        x1_pad = x1_pad.transpose(0, 1)

        x2_pad = x2.transpose(1, 2).transpose(0, 1)
        x3_pad = x3.transpose(1, 2).transpose(0, 1)

        att_output1 = self.mha(x2_pad, x1_pad, x1_pad)
        att_output1 = self.relu(att_output1)

        att_output2 = self.mha2(x3_pad, att_output1, att_output1)
        att_output2 = self.relu(att_output2)

        att_output2 = att_output2.transpose(1, 2)
        output = self.avgPool(att_output2)

        return output


class Classifier_per_time(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(Classifier_per_time, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)
        self.linear1 = nn.Linear(in_features=embed_dim, out_features=3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        output1 = x.permute(1, 2, 0)

        output2 = self.conv1(output1)
        output2 = F.relu(output2)
        output2 = output2.transpose(1, 2)

        output3 = self.linear1(output2)

        return output3


# 5, 64, 62, 3, 8, 25
class Baseline_model(nn.Module):
    def __init__(self, in_channel, out_channel, input_dim, final_dim, num_heads, topk):
        super(Baseline_model, self).__init__()
        self.pos_encoder = PositionalEncoding(out_channel, 0.1, 300)
        self.encoder = Conv1DEncoder(in_channel, out_channel)
        self.eca = Embed_CrossAttention(out_channel, num_heads, input_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=out_channel, num_heads=num_heads)
        self.classifier = Classifier_per_time(embed_dim=out_channel, output_dim=final_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=132, out_features=1)
        self.maxPool = nn.MaxPool1d(kernel_size=12, stride=12)

    def forward(self, x, device):
        batch_size = x.shape[0]
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], -1)
        x = x.transpose(0, 1)
        pos_x = self.pos_encoder(x).to(device)
        x = x.permute(1, 2, 0)
        pos_x = pos_x.permute(1, 2, 0)

        embed1, embed2, embed3 = self.encoder(x, pos_x, device)
        final_embed = self.eca(embed3, embed2, embed1)

        final_embed = final_embed.transpose(1, 2)
        sf_attention_output, _ = self.self_attention(final_embed, final_embed, final_embed)
        sf_attention_output = self.relu(sf_attention_output)

        class_output = self.classifier(sf_attention_output)
        
        max_values, _ = class_output.max(dim=-1)
        topk_values, topk_indices = torch.topk(max_values, 25, dim=1)


        top_25_vectors = torch.zeros((batch_size, 25, 3)).to(device)
        for i in range(final_embed.shape[1]):
            top_25_vectors[i, :, :] = class_output[i, topk_indices[i]]

        final_output = top_25_vectors.mean(dim=1)

        final_output = F.softmax(final_output, dim=-1)

        return final_output
