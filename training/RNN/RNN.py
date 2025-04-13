import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

data_root = "/home/swj/eye_fatigue_detection/data/processed_features"
model_save_path = "/home/swj/eye_fatigue_detection/models/fatigue_rnn.pth"
scaler_save_path = "/home/swj/eye_fatigue_detection/models/scaler.pkl"

class FatigueDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

def load_data():
    features_list = []
    labels = []
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
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f"Saved scaler to: {scaler_save_path}")
    
    return features_scaled, labels_array

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    features, labels = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    
    train_dataset = FatigueDataset(X_train, y_train)
    val_dataset = FatigueDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = FatigueRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 500
    best_accuracy = 0
    patience = 20
    no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features = features.unsqueeze(1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.unsqueeze(1).to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
            }, model_save_path)
            print(f'Saved model with accuracy: {best_accuracy:.2f}%')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping after {patience} epochs without improvement')
                break

if __name__ == '__main__':
    train_model()