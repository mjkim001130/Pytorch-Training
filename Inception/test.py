import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # tqdm import

from Inception import GoogleNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True)
    DATA_MEANS = (test_dataset.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (test_dataset.data / 255.0).std(axis=(0, 1, 2))

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEANS, DATA_STD)
    ])

    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    model = GoogleNet(num_classes=100, aux=False).to(device)
    checkpoint_path = './checkpoints/best_model.pth'
    
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # remove auxiliary classifier parameters from the state_dict
        filtered_state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("aux1") or k.startswith("aux2"))}
        model.load_state_dict(filtered_state_dict, strict=False)
        print("Checkpoint loaded from", checkpoint_path)
    else:
        print("Checkpoint path error")
        return
    
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(test_loader, desc="Testing", unit="batch")
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            current_loss = test_loss / total
            current_acc = 100. * correct / total
            pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")
    
    test_loss /= len(test_dataset)
    overall_acc = 100. * correct / total
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {overall_acc:.2f}%")

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cm, annot=False, fmt="d", cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    writer = SummaryWriter(log_dir='./runs/GoogleNet_CIFAR100_test')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt="d", cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    writer.add_figure("Confusion Matrix", fig, global_step=0)
    writer.add_scalar("Test Loss", test_loss, 0)
    writer.add_scalar("Test Accuracy", overall_acc, 0)
    writer.close()

if __name__ == "__main__":
    main()
