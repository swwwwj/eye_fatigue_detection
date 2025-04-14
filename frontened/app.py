from flask import Flask, render_template, Response, jsonify, send_file
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
import pandas as pd
from pathlib  import Path
from data_analysis import FatigueDataAnalyzer

# 数据存放路径
data_dir = Path("E:/data/eye_data")
data_dir.mkdir(parents=True, exist_ok=True)

# 模板路径
app = Flask(__name__, template_folder="D:/github/eye_fatigue_detection/frontened/templates")

# 定义discriminator判别器结构
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

def save_detection_data(timestamp, label, probabilities):
    """保存最近一小时的检测数据到CSV文件"""
    filename = data_dir / f"fatigue_data_{timestamp.strftime('%Y%m%d')}.csv"
    current_time = timestamp
    
    # 准备新数据
    new_data = {
        'timestamp': [timestamp],
        'label': [label],
        'prob_awake': [probabilities[0]],
        'prob_mild': [probabilities[1]],
        'prob_moderate': [probabilities[2]],
        'prob_severe': [probabilities[3]]
    }
    
    try:
        if filename.exists():
            # 读取现有数据并转换时间戳
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 只保留最近一小时的数据
            one_hour_ago = current_time - timedelta(hours=1)
            df = df[df['timestamp'] >= one_hour_ago]
            
            # 添加新数据
            new_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_df], ignore_index=True)
            
        else:
            # 创建新文件
            df = pd.DataFrame(new_data)
        
        # 保存数据
        df.to_csv(filename, index=False)
        
    except Exception as e:
        print(f"保存数据时出错: {e}")

# 使用GPU或CPU，调用预训练模型
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

# dlib人脸检测器
detector = dlib.get_frontal_face_detector()

# 图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 类别映射
class_names = {0: "awake", 1: "mild_fatigue", 2: "moderate_fatigue", 3: "severe_fatigue"}

# 统计数据存储
fatigue_stats = {
    "awake": deque(maxlen=3600),  # 保存1小时的数据
    "mild_fatigue": deque(maxlen=3600),
    "moderate_fatigue": deque(maxlen=3600),
    "severe_fatigue": deque(maxlen=3600),
}
SEVERE_FATIGUE_THRESHOLD = 50  # 严重疲劳警告阈值


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
        

        if len(faces) > 0:
            face = faces[0]  # 使用第一个检测到的人脸
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            
            # 框出人脸
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 提取人脸区域
            face_img = frame[y1:y2, x1:x2]
            if face_img.size != 0:  
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
                        save_detection_data(current_time, current_label, avg_prob.cpu().numpy()[0])
                        
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

analyzer = FatigueDataAnalyzer("sk-AqV2qUki0lPDuRUDGdrb9cKz7GPn9Tx1n7UQ2fu6gyLEBSSF")

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
    
    # 获取当前疲劳等级（0-3）
    current_level = 0  # 默认为清醒
    max_count = 0
    for i, level in enumerate(['awake', 'mild_fatigue', 'moderate_fatigue', 'severe_fatigue']):
        if stats[level] > max_count:
            max_count = stats[level]
            current_level = i
    
    # 检查是否需要发出警告
    warning = (sum(1 for t in fatigue_stats['severe_fatigue'] 
              if (current_time - t).total_seconds() <= 3600) 
              >= SEVERE_FATIGUE_THRESHOLD)
    
    return jsonify({
        'stats': stats,
        'warning': warning,
        'current_level': current_level  # 新增此字段
    })

# 弹窗后清空后台数据
@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    global fatigue_stats
    fatigue_stats = {
        "awake": deque(maxlen=3600),
        "mild_fatigue": deque(maxlen=3600),
        "moderate_fatigue": deque(maxlen=3600),
        "severe_fatigue": deque(maxlen=3600),
    }
    return jsonify({'status': 'success'})

@app.route('/history/<date>')
def get_history(date):
    """获取指定日期的历史数据"""
    try:
        datetime.strptime(date, '%Y%m%d')
        filename = data_dir / f"fatigue_data_{date}.csv"
        
        if filename.exists():
            df = pd.read_csv(filename)
            return jsonify({
                'status': 'success',
                'data': df.to_dict('records')
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '该日期没有数据'
            })
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': '日期格式无效'
        })
    
@app.route('/analyze/<date>')
def analyze_data(date):
    """分析指定日期的疲劳数据并生成报告"""
    result = analyzer.generate_analysis_report(date)
    return jsonify(result)

@app.route('/download_report/<date>/<format>')
def download_report(date, format):
    """下载报告文件"""
    try:
        if format == 'txt':
            file_path = data_dir / f"fatigue_report_{date}.txt"
            mimetype = 'text/plain'
        else:
            file_path = data_dir / f"fatigue_report_{date}.html"
            mimetype = 'text/html'
            
        if file_path.exists():
            return send_file(file_path, mimetype=mimetype, as_attachment=True)
        else:
            return jsonify({
                'status': 'error',
                'message': '文件不存在'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)