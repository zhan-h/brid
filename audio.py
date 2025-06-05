import os
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


# 自定义Dataset类
class BirdSoundDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.audio_files = []
        self.labels = []
        self.label_encoder = {}

        # 遍历子文件夹
        for label_dir in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_dir)
            if os.path.isdir(label_path):
                for audio_file in os.listdir(label_path):
                    if audio_file.endswith('.mp3'):
                        audio_path = os.path.join(label_path, audio_file)
                        self.audio_files.append(audio_path)
                        self.labels.append(label_dir)

        # 创建标签编码器
        self.label_encoder = {label: idx for idx, label in enumerate(set(self.labels))}
        self.labels = [self.label_encoder[label] for label in self.labels]

        # 调试信息
        print(f"Found {len(self.audio_files)} audio files in {data_dir}")
        print(f"Unique labels: {self.label_encoder}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        features = extract_features(audio_path)
        if self.transform:
            features = self.transform(features)
        return features, label


# 数据预处理
def extract_features(audio_file, n_mfcc=13):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return torch.tensor(mfccs.T, dtype=torch.float32)  # 转置以符合RNN的输入要求


# 自定义Collate函数以处理不同长度的音频特征
def custom_collate(batch):
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    # 填充数据以匹配最长的序列
    data = pad_sequence(data, batch_first=True)
    return data, torch.tensor(targets, dtype=torch.long)


# 构建模型
class BirdSoundClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BirdSoundClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集路径
train_data_dir = 'static/dataset/train'
test_data_dir = 'static/dataset/test'

# 创建Dataset对象
train_dataset = BirdSoundDataset(train_data_dir)
test_dataset = BirdSoundDataset(test_data_dir)

# 检查数据集是否为空
if len(train_dataset) == 0:
    raise ValueError(f"No audio files found in the train directory: {train_data_dir}")
if len(test_dataset) == 0:
    raise ValueError(f"No audio files found in the test directory: {test_data_dir}")

# 分割数据集
train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42)
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

# 创建DataLoader对象，使用自定义的collate函数
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, collate_fn=custom_collate)
val_loader = DataLoader(train_dataset, batch_size=32, sampler=val_sampler, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)

# 定义模型参数
input_size = 13  # MFCC特征的维度
hidden_size = 128
num_layers = 2
num_classes = len(train_dataset.label_encoder)

# 初始化模型
model = BirdSoundClassifier(input_size, hidden_size, num_layers, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 打印模型结构
print(model)

# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 验证模型
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'static/pth/bird_sound_classification_model.pth')
