from flask import Flask, render_template, Response
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

# 设置模板路径
app = Flask(__name__, template_folder=r"D:\github\eye_fatigue_detection\frontened\templates")

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

# 定义 FatigueRNN 模型，与RNN.py中保持一致
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

# 定义图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 类别映射字典
class_names = {0: "awake", 1: "mild_fatigue", 2: "moderate_fatigue", 3: "severe_fatigue"}

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
        
        # 将 BGR 转为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 模型1：判别器预测，并计算 softmax 概率
            output_d = discriminator(input_tensor)
            prob_d = torch.softmax(output_d, dim=1)
            
            # 模型2：对图像进行全局平均池化，转换输入格式后调用 FatigueRNN
            pooled = F.adaptive_avg_pool2d(input_tensor, (1, 1))
            rnn_input = pooled.view(1, 1, 3)  # (batch, seq_len, input_size)
            output_rnn = fatigue_rnn(rnn_input)
            prob_rnn = torch.softmax(output_rnn, dim=1)
            
            # 综合两个模型的概率（简单平均）
            prob_final = (prob_d + prob_rnn) / 2
        
        # 累积概率
        if prob_accumulator is None:
            prob_accumulator = prob_final
        else:
            prob_accumulator += prob_final
        frame_count += 1
        
        # 每累计10帧，更新一次平均概率预测
        if frame_count >= 10:
            avg_prob = prob_accumulator / frame_count
            final_class = torch.argmax(avg_prob, dim=1).item()
            current_label = class_names[final_class]
            frame_count = 0
            prob_accumulator = None
        
        # 在图像上叠加分类信息
        cv2.putText(frame, f'Class: {current_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 编码为 JPEG 格式
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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
