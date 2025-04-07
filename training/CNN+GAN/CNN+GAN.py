import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 1. 数据准备
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_data = ImageFolder(root="E:/data/face/CNN", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        
# 2. 定义 GAN 的 生成器 和 判别器
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64*64*3),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 3, 64, 64)

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
            nn.Linear(256, num_classes)  # 4 类分类
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# 3. 初始化网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.CrossEntropyLoss()

# 4. GAN 训练（生成数据）
best_loss = float('inf')
patience = 20
stale_epochs = 0

for epoch in range(500):
    epoch_loss = 0.0
    num_batches = 0
    for real_images, labels in train_loader:
        real_images, labels = real_images.to(device), labels.to(device)

        # 生成假图像
        z = torch.randn(real_images.size(0), 100).to(device)
        fake_images = generator(z)

        # 训练判别器
        optimizer_d.zero_grad()
        real_outputs = discriminator(real_images)
        fake_outputs = discriminator(fake_images.detach())

        real_loss = criterion(real_outputs, labels)
        fake_loss = criterion(fake_outputs, torch.randint(0, 4, (real_images.size(0),)).to(device))
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        # 训练生成器
        optimizer_g.zero_grad()
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, labels)  # 让 GAN 生成分类正确的图像
        g_loss.backward()
        optimizer_g.step()

        epoch_loss += (d_loss.item() + g_loss.item())
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    print(f"Epoch [{epoch+1}/500], Avg Loss: {avg_loss:.4f}, D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}")

    # 判断是否更新最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_generator_state = generator.state_dict()
        best_discriminator_state = discriminator.state_dict()
        stale_epochs = 0
        print("Best model updated.")
    else:
        stale_epochs += 1

    # 如果连续 patience 个 epoch loss 没有下降，则提前停止
    if stale_epochs >= patience:
        print(f"Loss has not decreased for {patience} epochs, stopping training.")
        break

# 5.保存表现最佳的模型
save_dir = r"D:\github\eye_fatigue_detection\models"
os.makedirs(save_dir, exist_ok=True)

torch.save(best_generator_state, os.path.join(save_dir, "best_generator.pth"))
torch.save(best_discriminator_state, os.path.join(save_dir, "best_discriminator.pth"))
print("最佳模型已保存到", save_dir)