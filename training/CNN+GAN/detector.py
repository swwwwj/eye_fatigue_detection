import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import os

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的判别器模型
model_path = r"D:\github\eye_fatigue_detection\models\best_discriminator.pth"
discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load(model_path, map_location=device))
discriminator.eval()

# 预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

class_names = {0: "awake", 1: "mild_fatigue", 2: "moderate_fatigue", 3: "severe_fatigue"}
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧")
        break

    # 转换图像为 RGB 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb_frame)
    input_tensor = input_tensor.unsqueeze(0).to(device)  

    with torch.no_grad():
        output = discriminator(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # 在画面上显示类别结果
    predicted_name = class_names[predicted_class]
    cv2.putText(frame, f'Class: {predicted_name}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Classification", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()