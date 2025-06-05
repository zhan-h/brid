"""
图片模型:测试
author:源学社
"""
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练数据集以获取类别名称列表
train_path = 'static/img/train'
train_dataset = ImageFolder(root=train_path, transform=transform)
class_names = train_dataset.classes

# 加载模型
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 200)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('static/pth/cub200_resnet18.pth'))
model.to(device)
model.eval()

# 加载图像
image_path = 'static/img/val/027.Shiny_Cowbird/Shiny_Cowbird_0001_796860.jpg'
image = Image.open(image_path).convert('RGB')
image = transform(image)
image = image.unsqueeze(0)

# 将图像移动到设备
image = image.to(device)

# 预测
with torch.no_grad():
    outputs = model(image)
    probabilities = F.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs.data, 1)

# 获取预测的类别名称和概率
predicted_class_index = predicted.item()
predicted_class_name = class_names[predicted_class_index]
predicted_probability = probabilities[0, predicted_class_index].item()
with open('static/json/name.json', 'r', encoding='utf-8') as f:
    name_dict = json.load(f)
predicted_class_name = name_dict[predicted_class_name]
print(f'Predicted bird name: {predicted_class_name}')
print(f'Predicted probability: {predicted_probability:.4f}')
