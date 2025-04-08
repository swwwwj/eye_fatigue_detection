import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from RNN import FatigueRNN, FatigueDataset

data_root = "/home/swj/eye_fatigue_detection/data/processed_features"
model_path = "/home/swj/eye_fatigue_detection/models/fatigue_rnn.pth"
scaler_path = "/home/swj/eye_fatigue_detection/models/scaler.pkl"

def load_test_data():
    features_list = []
    labels = []
    class_names = ['awake', 'mild_fatigue', 'moderate_fatigue', 'severe_fatigue']
    
    print(f"Looking for data in: {data_root}")
    
    if not os.path.exists(data_root):
        raise ValueError(f"Data directory not found: {data_root}")
    
    files_found = 0
    for file in os.listdir(data_root):
        if file.endswith('_features.npy'):
            feature_path = os.path.join(data_root, file)
            # 从文件名中提取疲劳等级
            if 'awake' in file:
                label = 0
            elif 'mild_fatigue' in file:
                label = 1
            elif 'moderate_fatigue' in file:
                label = 2
            elif 'severe_fatigue' in file:
                label = 3
            else:
                print(f"Skipping file with unknown fatigue level: {file}")
                continue
                
            try:
                features = np.load(feature_path)
                features_list.append(features)
                labels.append(label)
                files_found += 1
            except Exception as e:
                print(f"Error loading file {feature_path}: {e}")
    
    print(f"Found {files_found} feature files")
    
    if not features_list:
        raise ValueError("No feature files found! Please check the data directory.")
    
    features_array = np.array(features_list)
    labels_array = np.array(labels)
    
    print(f"Total samples loaded: {len(features_array)}")
    print(f"Features shape: {features_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    
    # 加载StandardScaler并转换特征
    scaler = joblib.load(scaler_path)
    features_scaled = scaler.transform(features_array)
    
    return features_scaled, labels_array, class_names

def validate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model = FatigueRNN().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model with accuracy: {checkpoint['accuracy']:.2f}%")
    
    model.eval()
    
    # 加载测试数据
    X_test, y_test, class_names = load_test_data()
    test_dataset = FatigueDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 进行预测
    all_predictions = []
    all_labels = []
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.unsqueeze(1).to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 打印测试结果
    accuracy = 100 * correct / total
    print(f"\nTest Results:")
    print(f"Average Loss: {test_loss/len(test_loader):.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == '__main__':
    validate_model()