"""
This is the code for training a bird classification model using ResNet18.
The dataset used is the CUB-200-2011 dataset.
The model is trained using mixed precision training to speed up the training process.
The model is saved as a PyTorch model file (.pth) for later use.
To run this code, you need to have PyTorch installed.
author: 源学社
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import torch.cuda.amp as amp

if __name__ == '__main__':
    # 检查GPU可用性并打印相关信息
    print(torch.cuda.is_available())  # 检查CUDA是否可用
    print(torch.cuda.device_count())  # 检查可用的GPU数量
    print(torch.cuda.current_device())  # 检查当前使用的GPU设备ID
    print(torch.cuda.get_device_name(0))  # 检查第一个GPU的名称

    dataset_path = 'static/img/'
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 构建数据加载器
    train_dataset = ImageFolder(root=train_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = ImageFolder(root=val_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 选择一个模型
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 200)  # 修改全连接层以适应200个类别

    # 将模型移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f'Using device: {device}')

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 混合精度训练
    scaler = amp.GradScaler()

    # 训练模型
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}] Train'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}] Val'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the validation images: {100 * correct / total:.2f}%')

    # 保存模型
    torch.save(model.state_dict(), 'static/pth/cub200_resnet18.pth')
