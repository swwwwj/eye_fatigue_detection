from flask import Flask, render_template, Response, jsonify, send_file, request
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
from pathlib import Path
from data_analysis import FatigueDataAnalyzer
import argparse
import base64

latest_detection_time = None
current_fatigue_level = None

# 解析命令行参数
parser = argparse.ArgumentParser(description="Eye Fatigue Detection System")
parser.add_argument("--model_dir", type=str, required=True, help="模型存储路径，例如 D:/github/eye_fatigue_detection/models")
parser.add_argument("--data_dir", type=str, required=True, help="数据存储路径，例如 E:/data/eye_data")
parser.add_argument("--template_dir", type=str, required=True, help="HTML 模板路径，例如 D:/github/eye_fatigue_detection/frontened/templates")
parser.add_argument("--port", type=int, default=5000, help="运行端口，默认5000")
args = parser.parse_args()

# 数据存放路径
data_dir = Path(args.data_dir)
data_dir.mkdir(parents=True, exist_ok=True)

# 模板路径
app = Flask(__name__, template_folder=args.template_dir)

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
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 只保留最近一小时的数据
            one_hour_ago = current_time - timedelta(hours=1)
            df = df[df['timestamp'] >= one_hour_ago]
            
            new_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_df], ignore_index=True)
            
        else:
            df = pd.DataFrame(new_data)
        
        # 保存数据
        df.to_csv(filename, index=False)
        
    except Exception as e:
        print(f"保存数据时出错: {e}")

# 使用GPU或CPU，调用预训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = os.path.join(args.model_dir, "best_discriminator.pth")
discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load(model_path, map_location=device))
discriminator.eval()

rnn_model_path = os.path.join(args.model_dir, "fatigue_rnn.pth")
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
SEVERE_FATIGUE_THRESHOLD = 2  # 严重疲劳警告阈值
MODERATE_FATIGUE_THRESHOLD = 10  # 中度疲劳警告阈值

analyzer = FatigueDataAnalyzer("sk-AqV2qUki0lPDuRUDGdrb9cKz7GPn9Tx1n7UQ2fu6gyLEBSSF")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stats')
def get_stats():
    current_time = datetime.now()

    stats = {
        level: sum(1 for t in times if (current_time - t).total_seconds() <= 600)
        for level, times in fatigue_stats.items()
    }

    global current_fatigue_level, latest_detection_time

    severe_warning = (sum(1 for t in fatigue_stats['severe_fatigue'] 
                     if (current_time - t).total_seconds() <= 3600) 
                     >= SEVERE_FATIGUE_THRESHOLD)
    
    moderate_warning = (sum(1 for t in fatigue_stats['moderate_fatigue']
                       if (current_time - t).total_seconds() <= 3600)
                       >= MODERATE_FATIGUE_THRESHOLD)
    
    warning = severe_warning or moderate_warning 
    warning_type = None
    if severe_warning:
        warning_type = 'severe'
    elif moderate_warning:
        warning_type = 'moderate'
    
    current_level = current_fatigue_level if current_fatigue_level is not None else 0
    detection_time = latest_detection_time if latest_detection_time is not None else current_time.strftime('%H:%M:%S')
    
    return jsonify({
        'stats': stats,
        'warning': warning,
        'warning_type': warning_type,
        'current_level': current_fatigue_level if current_fatigue_level is not None else 0,
        'detection_time': latest_detection_time
    })

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

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame received'})
            
        file = request.files['frame']
        nparr = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        current_time = datetime.now()
        current_label = "N/A"
        face_rect = None  # 添加人脸坐标变量
        
        # 人脸检测和疲劳分析
        faces = detector(frame)
        if len(faces) > 0:
            face = faces[0]
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            face_rect = [int(x1), int(y1), int(x2), int(y2)]  # 保存人脸坐标
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size != 0:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                input_tensor = transform(rgb_face)
                input_tensor = input_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output_d = discriminator(input_tensor)
                    prob_d = torch.softmax(output_d, dim=1)
                    
                    pooled = F.adaptive_avg_pool2d(input_tensor, (1, 1))
                    rnn_input = pooled.view(1, 1, 3)
                    output_rnn = fatigue_rnn(rnn_input)
                    prob_rnn = torch.softmax(output_rnn, dim=1)
                    
                    prob_final = (prob_d + prob_rnn) / 2
                    final_class = torch.argmax(prob_final, dim=1).item()
                    current_label = class_names[final_class]
                    
                    # 更新统计数据
                    global current_fatigue_level, latest_detection_time
                    latest_detection_time = current_time.strftime('%H:%M:%S')
                    current_fatigue_level = final_class
                    fatigue_stats[current_label].append(current_time)
                    save_detection_data(current_time, current_label, prob_final.cpu().numpy()[0])
        
        return jsonify({
            'status': 'success',
            'label': current_label,
            'time': latest_detection_time if latest_detection_time else current_time.strftime('%H:%M:%S'),
            'face_rect': face_rect  # 返回人脸坐标
        })
        
    except Exception as e:
        print(f"处理帧时出错: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print(f"\n服务已启动：")
    print(f"* 本地访问: http://localhost:{args.port}")
    print(f"* 远程访问: http://<服务器IP>:{args.port}\n")
    
    app.run(
        host="0.0.0.0",
        port=args.port,
        debug=False  # 生产环境关闭调试
    )