from flask import Flask, render_template, Response
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import os

app = Flask(__name__)

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

# 设置设备，并加载预训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"D:\github\eye_fatigue_detection\models\best_discriminator.pth"
discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load(model_path, map_location=device))
discriminator.eval()

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
    while True:
        success, frame = cap.read()
        if not success:
            break
        # 将 BGR 转为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = discriminator(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        predicted_name = class_names[predicted_class]
        
        # 在图像上叠加类别信息
        cv2.putText(frame, f'Class: {predicted_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 编码为 JPEG 格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # 使用 multipart/x-mixed-replace 返回视频流
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)