import numpy as np
import torch


def load_data(test_no, masking_mode, masking_ratio, random_seed):
    np.random.seed(random_seed)

    train_data_list = []
    train_label_list = []
    test_data_list = []
    test_label_list = []

    for i in range(15):
        dataname = f'./SEED_data/subject_{i}data.npy'
        labelname = f'./SEED_data/subject_{i}label.npy'
        data = np.load(dataname)
        label = np.load(labelname)

        if i == test_no:
            test_data_list.append(data)
            test_label_list.append(label)
        else:
            if masking_mode:
                mask1 = np.random.rand(62) < masking_ratio
                mask2 = np.random.rand(265) < masking_ratio
                masked_data = np.copy(data)
                masked_data[:, mask1, :, :] = 0
                masked_data[:, :, mask2, :] = 0
                
                train_data_list.append(masked_data)
            else:
                train_data_list.append(data)
            
            train_label_list.append(label)

    concatenated_data = np.concatenate(train_data_list, axis=0)
    concatenated_label = np.concatenate(train_label_list, axis=0)

    sample_feature = torch.tensor(concatenated_data, dtype=torch.float32)

    labels_one_hot = torch.zeros(concatenated_label.shape[0], 3)
    index = torch.tensor(concatenated_label + 1, dtype=torch.int64)
    labels_one_hot.scatter_(1, index, 1.0)

    test_concatenated_data = np.concatenate(test_data_list, axis=0)
    test_concatenated_label = np.concatenate(test_label_list, axis=0)

    test_sample_feature = torch.tensor(test_concatenated_data, dtype=torch.float32)

    test_labels_one_hot = torch.zeros(test_concatenated_label.shape[0], 3)
    test_index = torch.tensor(test_concatenated_label + 1, dtype=torch.int64)
    test_labels_one_hot.scatter_(1, test_index, 1.0)

    return sample_feature, labels_one_hot, test_sample_feature, test_labels_one_hot