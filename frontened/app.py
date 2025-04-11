from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import dlib
from collections import deque
from datetime import datetime, timedelta

# 设置模板路径
app = Flask(__name__, template_folder="/home/swj/eye_fatigue_detection/frontened/templates")

# 定义判别器网络结构，与训练时一致
class Discriminator(nn.Module):
    def __init__(self, num_classes=4):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# 定义 FatigueRNN 模型
class FatigueRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=4):
        super(FatigueRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# 设置设备，并加载预训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = r"D:\github\eye_fatigue_detection\models\best_discriminator.pth"
discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load(model_path, map_location=device))
discriminator.eval()

rnn_model_path = r"D:\github\eye_fatigue_detection\models\fatigue_rnn.pth"
fatigue_rnn = FatigueRNN().to(device)
checkpoint = torch.load(rnn_model_path, map_location=device)
fatigue_rnn.load_state_dict(checkpoint["model_state_dict"])
fatigue_rnn.eval()

# 初始化dlib人脸检测器
detector = dlib.get_frontal_face_detector()

# 定义图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 类别映射字典
class_names = {0: "awake", 1: "mild_fatigue", 2: "moderate_fatigue", 3: "severe_fatigue"}

# 统计数据存储
fatigue_stats = {
    "awake": deque(maxlen=3600),  # 保存1小时的数据
    "mild_fatigue": deque(maxlen=3600),
    "moderate_fatigue": deque(maxlen=3600),
    "severe_fatigue": deque(maxlen=3600),
}
SEVERE_FATIGUE_THRESHOLD = 50  # 严重疲劳警告阈值

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

# 生成视频帧
def gen_frames():
    frame_count = 0
    prob_accumulator = None
    current_label = "N/A"
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        current_time = datetime.now()
        
        # 人脸检测
        faces = detector(frame)
        
        # 如果检测到人脸
        if len(faces) > 0:
            face = faces[0]  # 使用第一个检测到的人脸
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            
            # 在原始帧上画出红色框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 提取人脸区域
            face_img = frame[y1:y2, x1:x2]
            if face_img.size != 0:  # 确保提取到了有效的人脸图像
                # 转RGB并处理人脸图像
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                try:
                    input_tensor = transform(rgb_face)
                    input_tensor = input_tensor.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        # 模型1：判别器预测
                        output_d = discriminator(input_tensor)
                        prob_d = torch.softmax(output_d, dim=1)
                        
                        # 模型2：RNN预测
                        pooled = F.adaptive_avg_pool2d(input_tensor, (1, 1))
                        rnn_input = pooled.view(1, 1, 3)
                        output_rnn = fatigue_rnn(rnn_input)
                        prob_rnn = torch.softmax(output_rnn, dim=1)
                        
                        # 综合两个模型的概率
                        prob_final = (prob_d + prob_rnn) / 2
                    
                    # 累积概率
                    if prob_accumulator is None:
                        prob_accumulator = prob_final
                    else:
                        prob_accumulator += prob_final
                    frame_count += 1
                    
                    # 每累计10帧，更新预测
                    if frame_count >= 10:
                        avg_prob = prob_accumulator / frame_count
                        final_class = torch.argmax(avg_prob, dim=1).item()
                        current_label = class_names[final_class]
                        
                        # 更新统计数据
                        fatigue_stats[current_label].append(current_time)
                        
                        frame_count = 0
                        prob_accumulator = None
                except Exception as e:
                    print(f"Error processing face: {e}")
        
        # 清理超过1小时的数据
        for level in fatigue_stats:
            while (fatigue_stats[level] and 
                   (current_time - fatigue_stats[level][0]).total_seconds() > 3600):
                fatigue_stats[level].popleft()
        
        # 在图像上显示分类结果
        cv2.putText(frame, f'Class: {current_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 编码为JPEG格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    current_time = datetime.now()
    ten_min_ago = current_time - timedelta(minutes=10)
    
    # 计算最近10分钟的统计
    stats = {
        level: sum(1 for t in times if (current_time - t).total_seconds() <= 600)
        for level, times in fatigue_stats.items()
    }
    
    # 检查是否需要发出警告
    warning = (sum(1 for t in fatigue_stats['severe_fatigue'] 
              if (current_time - t).total_seconds() <= 3600) 
              >= SEVERE_FATIGUE_THRESHOLD)
    
    return jsonify({
        'stats': stats,
        'warning': warning
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)