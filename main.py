
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Baseline_model
from dataloader import SeedDataset
import numpy as np
from sklearn.metrics import accuracy_score
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.manifold import TSNE # 추가
import matplotlib.pyplot as plt
import os
import time  ## 시간 측정을 위한 패키지


## Seed_data가 들어있는 폴더를 지정 ##

###################################################################################################################################
#workingDirectory = "/content/drive/MyDrive/DeepDaiv" ## /content/drive/MyDrive/DeepDaiv/Seed_data 가 Seed_data 경로라서 이렇게 지정해줌.
###################################################################################################################################

# 추가
pre_softmax_output = None

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = dim

    def forward(self, predictions, labels, eval = False):
        predictions = predictions.log_softmax(dim=self.dim)
        
        with torch.no_grad():
            indicator = 1.0 - labels
            smooth_labels = torch.zeros_like(labels)
            smooth_labels.fill_(self.smoothing / (self.classes - 1))
            smooth_labels = labels * self.confidence + indicator * smooth_labels#lables->indicator

        return torch.mean(torch.sum(-smooth_labels.cuda(0) * predictions, dim=self.dim))
    

# 추가
def hook_fn(module, input, output):
    global pre_softmax_output
    pre_softmax_output = input[0]


def fix_random_seeds(seed=0):
    """ Fix random seeds. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, pos, labels, masks) in enumerate(train_loader):
        inputs, pos, labels, masks = inputs.to(device), pos.to(device), labels.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, pos, masks)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    return running_loss / (i + 1)

def evaluate(model, test_loader, device):
    model.eval()
    predictions = []
    labels = []
    features = [] # 추가

    # 추가
    #global pre_softmax_output
    #model.softmax.register_forward_hook(hook_fn)


    with torch.no_grad():
        for inputs, pos, label,mask in test_loader:
            inputs, pos, label, mask = inputs.to(device), pos.to(device), label.to(device), mask.to(device)
            outputs = model(inputs, pos, mask)

            # 추가
            #features.append(pre_softmax_output.detach().cpu().numpy())

            prediction = np.squeeze(outputs.detach().to("cpu").numpy())
            predictions.append(prediction)
            label = label.detach().to("cpu").numpy()
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    predictions = np.argmax(predictions, axis = -1)
    labels = np.argmax(labels, axis = -1)
    accuracy = accuracy_score(labels, predictions)

    # 추가
    #features = np.vstack(features)
    labels = np.hstack(labels)

    # 추가
    return accuracy, features, labels

if __name__ == "__main__":

    fix_random_seeds(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #top = [1]
    #top = [132,64,32,16]
    top = [4,8]

    for k in top:
        logs = []
        print(k)
        start_time = time.time()

        for test_no in range(15): ## 15
            batch_size = 8
            learning_rate = 1e-3
            num_epochs = 300

            model = Baseline_model(in_channel=5,
                                out_channel=64,
                                input_dim=62,
                                final_dim=3,
                                num_heads=8,
                                topk=k).to(device)

            #if torch.cuda.device_count() > 1:
            #    print(f"Using {torch.cuda.device_count()} GPUs")
            #    model = nn.DataParallel(model)

            # model.load_state_dict(torch.load('./model/epoch100.pt', map_location=device))

            ############ He Initialization ############

            #criterion = criterion = nn.CrossEntropyLoss().to(device)
            criterion = LabelSmoothingLoss(3, smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.05)

            train_dataset = SeedDataset(f"SEED_data/train_{test_no}de.pt")
            test_dataset = SeedDataset(f"SEED_data/test_{test_no}de.pt" ,training=False)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1)
            scheduler = CosineAnnealingLR(optimizer, len(train_loader) * num_epochs, 1e-6)
            best_accuracy = 0.0
            best_features = None ## 추가
            best_labels = None ## 추가
            for epoch in range(num_epochs):
                #if (epoch+1) % 1 == 0:
                train_dataset = SeedDataset(f"SEED_data/train_{test_no}de.pt")
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                loss = train(model, train_loader, criterion, optimizer, scheduler, device)
                # 추가
                train_accuracy, _, _ = evaluate(model, train_loader, device)
                # 추가
                test_accuracy, test_features, test_labels = evaluate(model, test_loader, device)
                lr = scheduler.get_last_lr()[0]
                if test_accuracy >= best_accuracy:
                    best_accuracy = test_accuracy
                    best_features = test_features ## 추가
                    best_labels = test_labels ## 추가
                    torch.save(model.state_dict(), f'rere/best/test{test_no}_{k}best2.pt')


                print(f"Subject: {test_no}, " +
                    f"Epoch: {epoch + 1:03d}, " +
                    f"Train Acc: {train_accuracy:.4f}, " +
                    f"Train loss: {loss:.4f}, " +
                    f"Test Acc: {test_accuracy:.4f}, " +
                    f"Best Acc: {best_accuracy:.4f}, " 
                    )

                if test_accuracy == 1.0:
                    break

            logs.append([best_accuracy, test_accuracy])


            # 추가
            test_accuracy = best_accuracy ## 추가
            test_features = best_features ## 추가
            test_labels = best_labels ## 추가



            with open(f"rere/best/log-{k}best2.txt", "wt") as f:
                for log in logs:
                    f.writelines(f"{log[0]:.4f}, {log[1]:.4f}\n")

        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"Elapsed time: {elapsed_time:.2f} seconds")