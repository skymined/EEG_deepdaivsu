%%writefile main.py

## baseline, seeEndTime의 main

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append("/content")
from model import Baseline_model
from dataloader import SeedDataset
import numpy as np
from sklearn.metrics import accuracy_score
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import time  ## 시간 측정을 위한 패키지


def fix_random_seeds(seed=0):
    """Fix random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, pos, labels) in enumerate(train_loader):
        inputs, pos, labels = inputs.to(device), pos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, pos)

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
    with torch.no_grad():
        for inputs, pos, label in test_loader:
            inputs, pos, label = inputs.to(device), pos.to(device), label.to(device)
            outputs = model(inputs, pos)
            prediction = np.squeeze(outputs.detach().to("cpu").numpy())
            predictions.append(prediction)
            label = label.detach().to("cpu").numpy()
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    predictions = np.argmax(predictions, axis=-1)
    labels = np.argmax(labels, axis=-1)
    accuracy = accuracy_score(labels, predictions)

    return accuracy


if __name__ == "__main__":

    seed_data_test_dir = "/content/drive/MyDrive/DeepDaiv/Seed_data/test_"
    seed_data_train_dir = "/content/drive/MyDrive/DeepDaiv/Seed_data/train_"

    start_time = time.time()  ## 시간 체크

    fix_random_seeds(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    logs = []
    for test_no in range(15):
        batch_size = 32
        learning_rate = 1e-3
        num_epochs = 300

        model = Baseline_model(
            in_channel=5, out_channel=64, input_dim=62, final_dim=3, num_heads=8, topk=5
        ).to(device)
        # model.load_state_dict(torch.load('./model/epoch100.pt', map_location=device))

        ############ He Initialization ############
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                # He normal initialization (for ReLU activation)
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

            elif isinstance(layer, nn.Conv1d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.05
        )

        train_dataset = SeedDataset(f"{seed_data_train_dir}{test_no}de.pt")
        test_dataset = SeedDataset(f"{seed_data_test_dir}{test_no}de.pt", training=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)
        scheduler = CosineAnnealingLR(optimizer, len(train_loader) * num_epochs, 1e-6)

        best_accuracy = 0.0
        for epoch in range(num_epochs):
            if (epoch + 1) % 5 == 0:
                train_dataset = SeedDataset(f"{seed_data_train_dir}{test_no}de.pt")
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )

            loss = train(model, train_loader, criterion, optimizer, scheduler, device, epoch)
            train_accuracy = evaluate(model, train_loader, device)
            test_accuracy = evaluate(model, test_loader, device)
            lr = scheduler.get_last_lr()[0]
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), f"/content/drive/MyDrive/DeepDaiv/model/test{test_no}_best.pt")
            print(
                f"Subject: {test_no}, "
                + f"Epoch: {epoch + 1:03d}, "
                + f"Train Acc: {train_accuracy:.4f}, "
                + f"Test Acc: {test_accuracy:.4f}, "
                + f"Best Acc: {best_accuracy:.4f}, "
            )

            if test_accuracy == 1.0:
                break

        logs.append([best_accuracy, test_accuracy])

        with open(f"/content/drive/MyDrive/DeepDaiv/best/log-best.txt", "wt") as f:
            for log in logs:
                f.writelines(f"{log[0]:.4f}, {log[1]:.4f}\n")

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")
