import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# 265 == best length

data_list = []
label_list = []

for i in range(15):
    dataname = f'./SEED_data/subject_{i}data.npy'
    labelname = f'./SEED_data/subject_{i}label.npy'
    data = np.load(dataname)
    label = np.load(labelname)
    data_list.append(data)
    label_list.append(label)

concatenated_data = np.concatenate(data_list, axis=0)
concatenated_label = np.concatenate(label_list, axis=0)

sample_feature = torch.tensor(concatenated_data, dtype=torch.float32)
sample_feature = sample_feature.permute(0, 2, 1, 3)
sample_feature = sample_feature.reshape(sample_feature.shape[0], sample_feature.shape[1], -1)
sample_feature = sample_feature.transpose(1, 2)

labels_one_hot = torch.zeros(concatenated_label.shape[0], 3)
index = torch.tensor(concatenated_label + 1, dtype=torch.int64)
labels_one_hot.scatter_(1, index, 1.0)

print(sample_feature.shape)

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

        att_output2 = att_output2.transpose(0, 1)
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
        output = F.softmax(output3, dim=-1)

        return output

# 5, 64, 62, 3, 8, 25
class Baseline_model(nn.Module):
    def __init__(self, in_channel, out_channel, input_dim, final_dim, num_heads, topk):
        super(Baseline_model, self).__init__()
        self.encoder = Conv1DEncoder(input_dim * in_channel, input_dim * out_channel)
        self.eca = Embed_CrossAttention(input_dim * out_channel, num_heads, input_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=out_channel, num_heads=num_heads)
        self.classifier = Classifier_per_time(embed_dim=out_channel, output_dim=final_dim)

    def forward(self, x):
        positional_embedding = torch.arange(1, x.shape[2] + 1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        embed1, embed2, embed3 = self.encoder(x, positional_embedding)
        final_embed = self.eca(embed3, embed2, embed1)

        final_embed = final_embed.transpose(0, 1)
        sf_attention_output, _ = self.self_attention(final_embed, final_embed, final_embed)

        class_output = self.classifier(sf_attention_output)

        max_values, _ = class_output.max(dim=-1)
        topk_values, topk_indices = torch.topk(max_values, 25, dim=1)
        top_25_vectors = class_output[:, topk_indices[0]]
        final_output = top_25_vectors.mean(dim=1)

        return final_output

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


batch_size = 32
learning_rate = 0.001
num_epochs = 50 

dataset = TensorDataset(sample_feature, labels_one_hot)
train_size = int(0.9 * len(dataset)) 
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Baseline_model(in_channel=5, 
                       out_channel=64, 
                       input_dim=62, 
                       final_dim=3, 
                       num_heads=8, 
                       topk=25)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 함수
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # 매 10 배치마다 출력
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

# 평가 함수
def evaluate(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# 학습 및 테스트 루프
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    
    # 5 에포크마다 테스트
    if (epoch + 1) % 2 == 0:
        print(f"\nTesting at epoch {epoch + 1}:")
        evaluate(model, test_loader, criterion)
        print("-" * 50)