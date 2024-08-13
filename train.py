import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import Baseline_model
from dataloader import load_data

def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  

        optimizer.zero_grad()
        outputs = model(inputs, device)
        loss = criterion(outputs, labels)
        old_state=model.state_dict()
        loss.backward()
        optimizer.step()
        new_state=model.state_dict()
        running_loss += loss.item()
        if i % 10 == 9:  
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
            
            running_loss = 0.0

def evaluate(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs, device)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    features, labels = load_data()

    batch_size = 32
    learning_rate = 0.001
    num_epochs = 1000

    dataset = TensorDataset(features, labels)
    train_size = int(0.9 * len(dataset)) 
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    model = Baseline_model(in_channel=5,
                           out_channel=64,
                           input_dim=62,
                           final_dim=3,
                           num_heads=8,
                           topk=25).to(device)  
    
    # model.load_state_dict(torch.load('./model/epoch100.pt', map_location=device))

    criterion = nn.MSELoss().to(device)  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, epoch, device)

        if (epoch + 1) % 20 == 0:
            print(f"\nTesting at epoch {epoch + 1}:")
            evaluate(model, test_loader, criterion, device)
            print("-" * 50)
            torch.save(model.state_dict(), f'./model/epoch{80 + epoch + 1}.pt')
