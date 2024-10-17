
## baseline

import numpy as np
import os
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, embed_dim, channel):
        super().__init__()
        self.norm = nn.LayerNorm([channel, embed_dim])

    def forward(self, x):
        b, t, n, c = x.shape
        x = x.reshape(b * t, n, c)
        x = self.norm(x)
        x = x.reshape(b, t, n, c)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        # self.fc2 = nn.Linear(output_channels, output_channels)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        return x


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

    def forward(self, x1, x2, m2=None):

        B1, T1, N1, C1 = (
            x1.shape
        )  # if x2,x3 -> batch, 66, 62, 64  elif x1,x2 -> batch, 132, 62, 64
        B2, T2, N2, C2 = (
            x2.shape
        )  # if x2,x3 -> batch, 33, 62, 64  elif x1,x2 -> batch, 66, 62, 64
        x1 = x1.transpose(1, 2).reshape(B1 * N1, T1, C1)
        x2 = x2.transpose(1, 2).reshape(B2 * N2, T2, C2)
        q = self.q(x1).reshape(B1 * N1, T1, self.num_head, -1).transpose(1, 2)
        k = self.k(x2).reshape(B2 * N2, T2, self.num_head, -1).transpose(1, 2)
        v = self.v(x2).reshape(B2 * N2, T2, self.num_head, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2) / np.sqrt(C1)
        if m2 is not None:
            m2 = m2.reshape(B2, 1, 1, 1, T2)
            m2 = torch.tile(m2, (1, N2, 1, 1, 1))
            m2 = m2.reshape(B2 * N2, 1, 1, T2)
            attn = attn.masked_fill(m2 == 0, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B1, N1, T1, C1).transpose(1, 2)
        x = self.u(x)
        x = self.dropout(x)
        return x


class CrossFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mca = MCA(embed_dim)
        self.mlp = MLP(embed_dim)
        self.norm1_1 = LayerNorm(embed_dim, 62)
        self.norm1_2 = LayerNorm(embed_dim, 62)
        self.norm = LayerNorm(embed_dim, 62)

    def forward(self, x1, x2, m2):
        # Attention
        h1 = self.norm1_1(x1)
        h2 = self.norm1_2(x2)
        h = self.mca(h1, h2, m2)
        x = x1 + h

        # MLP
        h = self.norm(x)
        h = self.mlp(h)
        x = x + h
        return x


class selfatt(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.multi_att = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.mlp = MLP(embed_dim)
        self.norm1_1 = LayerNorm(embed_dim, 62)
        self.norm = LayerNorm(embed_dim, 62)

    def forward(self, x, m):
        # Attention
        # h = self.norm1_1(x)
        h, _ = self.multi_att(x, x, x, attn_mask=m)
        x = x + h

        # MLP
        # h = self.norm(x)
        h = self.mlp(x)
        x = x + h
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
        self.pool1 = nn.AvgPool1d(2, 2)
        self.pool2 = nn.AvgPool1d(2, 2)
        self.pool3 = nn.AvgPool1d(2, 2)
        self.hidden_dim = output_channels

    def forward(self, x, pos):
        b, t, n = x.shape[:3]
        x = x.permute(0, 2, 3, 1).reshape(b * n, -1, t)  # batch*62, 5, 265
        pos = pos.permute(0, 2, 3, 1).reshape(b * n, -1, t)  # batch*62, 64, 265

        x1 = self.conv1(x)  ## output : (batch, channel, �곗씠�� 湲몄씠)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x1 = x1.reshape(b, n, self.hidden_dim, -1).permute(
            0, 3, 1, 2
        )  # batch, 132, 62, 64
        x2 = x2.reshape(b, n, self.hidden_dim, -1).permute(
            0, 3, 1, 2
        )  # batch, 66, 62, 64
        x3 = x3.reshape(b, n, self.hidden_dim, -1).permute(
            0, 3, 1, 2
        )  # batch, 33, 62, 64

        p1 = self.pool1(pos)
        p2 = self.pool2(p1)
        p3 = self.pool3(p2)

        p1 = p1.reshape(b, n, self.hidden_dim, -1).permute(
            0, 3, 1, 2
        )  # batch, 132, 62, 64
        p2 = p2.reshape(b, n, self.hidden_dim, -1).permute(
            0, 3, 1, 2
        )  # batch, 66, 62, 64
        p3 = p3.reshape(b, n, self.hidden_dim, -1).permute(
            0, 3, 1, 2
        )  # batch, 33, 62, 64

        return x1 + p1, x2 + p2, x3 + p3


class Embed_CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(Embed_CrossAttention, self).__init__()
        self.cross_fusion21 = CrossFusion(embed_dim)
        self.cross_fusion32 = CrossFusion(embed_dim)
        self.relu = nn.ReLU()
        self.norm = LayerNorm(embed_dim, 62)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x1, x2, x3):
        x2 = self.cross_fusion32(x2, x3, None)
        x1 = self.cross_fusion21(x1, x2, None)

        x = self.relu(self.norm(x1))
        x = x.mean([-2])
        x = self.dropout(x)

        return x  # batch,132,64


class Classifier_per_time(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(Classifier_per_time, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=1
        )
        self.linear1 = nn.Linear(in_features=embed_dim, out_features=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        output1 = x.permute(0, 2, 1)

        output2 = self.conv1(output1)
        output2 = self.relu(output2)
        output2 = output2.transpose(1, 2)

        output3 = self.linear1(output2)  # batch,132,3
        return output3


# 5, 64, 62, 3, 8, 25
class Baseline_model(nn.Module):
    def __init__(self, in_channel, out_channel, input_dim, final_dim, num_heads, topk):
        super(Baseline_model, self).__init__()
        self.encoder = Conv1DEncoder(in_channel, out_channel)
        self.eca = Embed_CrossAttention(out_channel)
        self.self_attention = selfatt(embed_dim=out_channel)
        self.classifier = Classifier_per_time(
            embed_dim=out_channel, output_dim=final_dim
        )
        self.relu = nn.ReLU()
        self.topk = topk
        self.classes = 3
        self.softmax = nn.Softmax(dim=-1)
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(132 * 3, 64) ## ablation test �섎떎 �곸슜�� �뚮뒗 132媛� �꾨땲�� 33�� �섏뼱�� ��.
        self.linear2 = nn.Linear(64, 3)
        self._initialize_weights() ## 媛�以묒튂 珥덇린��

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        

    def forward(self, x, pos, m):
        # x.shape = 62,265,5
        embed1, embed2, embed3 = self.encoder(x, pos)
        final_embed = self.eca(embed1, embed2, embed3)  # batch,132,64

        sf_attention_output = self.self_attention(final_embed, None)
        


        class_output = self.classifier(sf_attention_output)  # batch,132,3

        # y = self.flat(class_output)
        
        # y = self.relu(self.linear1(y))
       
        # y = self.linear2(y)

        y = []
        B, T = class_output.shape[:2]  # 8, 132, 3 => 8, 132
        ## print(f"class_output.shape : {class_output.shape}")

        topk = self.topk
        for c in range(self.classes):
            s = class_output[:, :, c]
            i = torch.topk(s, topk, dim=-1)[1]
            yc = [torch.mean(class_output[b, i[b, :], c], dim=0) for b in range(B)]
            yc = torch.stack(yc, dim=0)
            y.append(yc)
        y = torch.stack(y, dim=-1)

        # print(f"y.shape : {y.shape}")
        #y_soft = self.softmax(y)

        return y