import hashlib
import random
import shutil
import django
import os
import librosa
from torch.utils.data import Dataset
import json
from django.http import JsonResponse
from django.core.cache import cache
from django.core.mail import send_mail
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from django.conf import settings
from .deepseek import deepseek_generate
from .models import User, DeepseekUser
from django.shortcuts import render, redirect

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageTester:
    def __init__(self):
        self.model_path = os.path.join(settings.STATIC_ROOT, 'pth/cub200_resnet18.pth')
        self.name_json = os.path.join(settings.STATIC_ROOT, 'json/name.json')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        self.classes = self._load_classes()
        self.name_mapping = self._load_name_mapping()

    def _load_classes(self):
        # 加载训练数据集以获取类别名称列表
        train_path = os.path.join(settings.STATIC_ROOT, 'img/train')
        train_dataset = ImageFolder(root=train_path, transform=transform)
        return train_dataset.classes

    def _load_name_mapping(self):
        with open(self.name_json, 'r', encoding='utf-8') as f:
            return json.load(f)

    def predict(self, image_path):
        # 加载图像并进行预处理
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)

        # 加载模型
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.classes))
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        # 预测
        with torch.no_grad():
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)[0]

        # 获取预测的类别名称和概率
        predicted_class_index = torch.argmax(probabilities).item()
        predicted_class_name = self.name_mapping.get(self.classes[predicted_class_index], '未知类别')
        predicted_probability = probabilities[predicted_class_index].item()

        # 获取前两个概率最高的类别和概率
        sorted_probabilities = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
        top_two_probabilities = sorted_probabilities[:2]
        top_two_probabilities_dict = {
            self.name_mapping.get(self.classes[idx], '未知类别'): f"{prob:.2%}"
            for idx, prob in top_two_probabilities
        }

        return {
            'predicted_class': predicted_class_name,
            'confidence': f"{predicted_probability:.2%}",
            'probabilities': {
                self.name_mapping.get(self.classes[i], '未知类别'): f"{probabilities[i].item():.2%}"
                for i in range(len(probabilities))
            },
            'top_two_probabilities': top_two_probabilities_dict
        }


def image_recognition(request):
    if not request.session.get('username'):
        return redirect('login')

    error = None
    result = None
    image_url = None

    if request.method == 'POST':
        if 'image_file' not in request.FILES:
            error = '未接收到图像文件'
        else:
            image_file = request.FILES['image_file']
            if not image_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                error = '仅支持JPG和PNG格式图像'
            else:
                upload_dir = os.path.join(settings.MEDIA_ROOT, 'images')
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, image_file.name)
                try:
                    with open(file_path, 'wb+') as f:
                        for chunk in image_file.chunks():
                            f.write(chunk)
                    if os.path.exists(file_path):
                        image_url = os.path.join(settings.MEDIA_URL, 'images', image_file.name)
                        tester = ImageTester()
                        result = tester.predict(file_path)
                    else:
                        error = '保存图像文件失败'
                except Exception as e:
                    error = f'处理失败: {e}'

    return render(request, 'image.html', {
        'error': error,
        'result': result,
        'image_url': image_url
    })


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


class BirdSoundClassifierTester:
    def __init__(self):
        self.model_path = os.path.join(settings.STATIC_ROOT, 'pth/bird_sound_classification_model.pth')
        self.name_json = os.path.join(settings.STATIC_ROOT, 'json/mp3.json')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 13  # MFCC特征的维度
        self.hidden_size = 128
        self.num_layers = 2
        self.num_classes = len(set(os.listdir('static/dataset/train')))  # 假设训练集目录结构与测试集相同
        self.model = self._load_model()
        self.label_decoder = self._load_label_decoder()

    def _load_model(self):
        model = BirdSoundClassifier(self.input_size, self.hidden_size, self.num_layers, self.num_classes).to(
            self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device).eval()
        return model

    def _load_label_decoder(self):
        train_dataset = BirdSoundDataset('static/dataset/train', transform=extract_features)
        label_decoder = {idx: label for label, idx in train_dataset.label_encoder.items()}
        return label_decoder

    def predict(self, file_path):
        features = extract_features(file_path, n_mfcc=self.input_size)
        features = torch.unsqueeze(features, 0)  # 增加一个批次维度
        features = features.to(self.device)

        with torch.no_grad():
            output = self.model(features)
            prob = torch.nn.functional.softmax(output, dim=1)[0]  # 计算 softmax 概率
            _, predicted_idx = torch.max(prob, 0)
            predicted_label = self.label_decoder[predicted_idx.item()]

        with open(self.name_json, 'r', encoding='utf-8') as f:
            mp3_json = json.load(f)
            predicted_bird_name = mp3_json.get(predicted_label, '未知类别')
        print(f"Predicted label: {predicted_label}, {predicted_bird_name}")
        print(f"Confidence: {prob[predicted_idx.item()]:.2%}")
        print(f"Probabilities: {prob}")

        return {
            "predicted_class": predicted_bird_name,
            "confidence": f"{prob[predicted_idx.item()]:.2%}",
            "probabilities": {
                mp3_json.get(self.label_decoder[idx], '未知类别'): f"{prob[idx].item():.2%}"
                for idx in range(len(prob))
            },
        }


def generate_and_cache_verification_code(email, timeout=60):
    code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    cache.set(email, code, timeout)
    return code


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "deepseek.settings")
django.setup()


def send_verification_code(email, code):
    send_mail(
        '验证码',
        f'您的验证码为：{code}',
        settings.EMAIL_HOST_USER,
        [email],
        fail_silently=False,
    )


def request_verification_code(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        if not email:
            return JsonResponse({'success': False, 'message': '请输入有效的电子邮箱地址'})
        code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        cache.set(email, code, timeout=300)
        send_verification_code(email, code)
        return JsonResponse({'success': True, 'message': '验证码已发送至您的邮箱，请查收'})

    return JsonResponse({'success': False, 'message': '请求方法错误'})


def login(request):
    if request.session.get('username'):
        print(request.session.get('username'))
        return render(request, 'index.html', {'success': '您已登录'})
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        password = hashlib.sha256(password.encode('utf-8')).hexdigest()
        user = User.objects.filter(username=username, password=password).first()
        if user:
            request.session['user_id'] = user.id
            request.session['username'] = user.username
            return render(request, 'index.html', {'success': '登录成功'})
        else:
            return render(request, 'login.html', {'error': '用户名或密码错误'})
    else:
        return render(request, 'login.html')


def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')
        code = request.POST.get('code')
        if User.objects.filter(username=username).exists():
            return render(request, 'register.html', {'error': '用户名已存在', 'username': username, 'email': email})
        if not code:
            verification_code = generate_and_cache_verification_code(email)
            send_verification_code(email, verification_code)
            return render(request, 'register.html', {'message': '验证码已发送到您的邮箱，请查收'})
        cached_code = cache.get(email)
        if code != str(cached_code):
            return render(request, 'register.html', {'error': '验证码错误', 'username': username, 'email': email})
        else:
            password = hashlib.sha256(password.encode('utf-8')).hexdigest()
            user = User.objects.create(username=username, password=password, email=email)
            user.save()
            return render(request, 'login.html')
    return render(request, 'register.html')


def deepseek_api(request):
    if not request.session.get('username'):
        return redirect('login')
    print(request.session.get('username'))
    if request.method == 'POST':
        text = request.POST.get('text')
        deepseek_text = deepseek_generate(text)
        DeepseekUser.objects.create(
            username=request.session.get('username'),
            text=text,
            deepseek_text=deepseek_text
        )
        return redirect('deepseek_api')

    deepseek_users = DeepseekUser.objects.filter(username=request.session.get('username')).order_by('created_at')
    return render(request, 'deepseek.html', {'deepseek_users': deepseek_users})


def clean_media_folder(request):
    if not request.session.get('username'):
        return redirect('login')
    if request.method == 'GET':
        media_dir = settings.MEDIA_ROOT
        try:
            for filename in os.listdir(media_dir):
                file_path = os.path.join(media_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            return render(request, 'index.html', {'success': '清理成功'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'清理失败: {e}'})
    else:
        return JsonResponse({'status': 'error', 'message': '仅支持GET请求'})


def logout(request):
    if request.session.get('username'):
        del request.session['user_id']
        del request.session['username']
    return redirect('login')


def audio_recognition(request):
    if not request.session.get('username'):
        return redirect('login')
    error = None
    result = None
    audio_url = None

    if request.method == 'POST':
        if 'audio_file' not in request.FILES:
            error = '未接收到音频文件'
        else:
            audio_file = request.FILES['audio_file']
            if not audio_file.name.lower().endswith('.mp3'):
                error = '仅支持MP3格式音频'
            else:
                upload_dir = os.path.join(settings.MEDIA_ROOT, 'audio')
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, audio_file.name)
                try:
                    with open(file_path, 'wb+') as f:
                        for chunk in audio_file.chunks():
                            f.write(chunk)
                    if os.path.exists(file_path):
                        audio_url = os.path.join(settings.MEDIA_URL, 'audio', audio_file.name)
                        tester = BirdSoundClassifierTester()
                        result = tester.predict(file_path)
                    else:
                        error = '保存音频文件失败'
                except Exception as e:
                    error = f'处理失败: {e}'

    return render(request, 'audio.html', {
        'error': error,
        'result': result,
        'audio_url': audio_url
    })
