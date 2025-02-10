# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# google_net.py 파일에 정의한 GoogleNet 모델 import
from Inception import GoogleNet

def main():

    num_epochs = 50
    batch_size = 128
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CIFAR100 

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True)
    DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2))
    DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2))

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),                # 224x224 upsampling
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEANS, DATA_STD)
    ])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = GoogleNet(num_classes=100, aux=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # TensorBoard 
    writer = SummaryWriter(log_dir='./runs/GoogleNet_CIFAR100')
    
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # GoogleNet's forward() returns main classifier output, and auxiliary classifier outputs
            outputs, aux1, aux2 = model(inputs)
            loss_main = criterion(outputs, targets)
            # If aux classifier exists, add auxiliary classifier losses to main classifier loss
            if aux1 is not None and aux2 is not None:
                loss_aux1 = criterion(aux1, targets)
                loss_aux2 = criterion(aux2, targets)
                loss = loss_main + 0.3 * (loss_aux1 + loss_aux2)
            else:
                loss = loss_main

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix(loss=running_loss/(batch_idx+1), acc=100.*correct/total)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            torch.save(model.state_dict(), './checkpoints/best_model.pth')
    
    writer.close()

if __name__ == "__main__":
    main()
