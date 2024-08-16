import numpy as np
import torch


def load_data():
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
    # sample_feature = sample_feature.permute(0, 2, 1, 3)
    # sample_feature = sample_feature.reshape(sample_feature.shape[0] * sample_feature.shape[1], sample_feature.shape[2], -1)
    # sample_feature = sample_feature.transpose(1, 2)

    labels_one_hot = torch.zeros(concatenated_label.shape[0], 3)
    index = torch.tensor(concatenated_label + 1, dtype=torch.int64)
    labels_one_hot.scatter_(1, index, 1.0)

    return sample_feature, labels_one_hot

