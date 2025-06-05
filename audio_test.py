import os
import librosa
import torch
from torch import nn
from torch.utils.data import Dataset
import json


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


def extract_features(audio_file, n_mfcc=13):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return torch.tensor(mfccs.T, dtype=torch.float32)  # 转置以符合RNN的输入要求


train_dataset = BirdSoundDataset('static/dataset/train', transform=extract_features)
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型参数，与训练时相同
input_size = 13  # MFCC特征的维度
hidden_size = 128
num_layers = 2
num_classes = len(set(os.listdir('static/dataset/train')))  # 假设训练集目录结构与测试集相同

# 初始化模型并加载权重
model = BirdSoundClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load('static/pth/bird_sound_classification_model.pth'))
model.eval()  # 设置模型为验证模式

# 定义标签解码器
label_decoder = {idx: label for label, idx in train_dataset.label_encoder.items()}

# 测试音频文件路径
test_audio_path = 'static/mp3/XC1003942 - 普通朱雀 - Carpodacus erythrinus.mp3'

# 提取测试音频的特征
test_features = extract_features(test_audio_path, n_mfcc=input_size)
test_features = torch.unsqueeze(test_features, 0)  # 增加一个批次维度

# 进行预测
with torch.no_grad():
    test_features = test_features.to(device)
    output = model(test_features)
    prob = torch.nn.functional.softmax(output, dim=1)[0]  # 计算 softmax 概率
    _, predicted_idx = torch.max(prob, 0)
    predicted_label = label_decoder[predicted_idx.item()]

# print(f"The predicted label for the audio file is: {predicted_label}")
with open('static/json/mp3.json', 'r', encoding='utf-8') as f:
    mp3_json = json.load(f)
    print(f"The name of the bird is: {mp3_json[predicted_label]}")
print(f"Confidence: {prob[predicted_idx.item()]:.2%}")

# 输出所有类别的概率
print("Probabilities for each class:")
for idx, label in label_decoder.items():
    print(f"{label}: {prob[idx].item():.2%}")
