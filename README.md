🧠 面部疲劳识别系统
📌 项目简介（Project Overview）
本项目是一个基于深度学习的面部疲劳识别系统，通过实时视频分析用户的眼部状态，从多个维度综合判断疲劳等级。项目结合了图像处理与时序建模技术，具有一定的实用性与研究价值，适用于学习监控、驾驶辅助等场景。

✨ 功能特点（Features）
实时视频采集与人脸检测

CNN 提取面部静态特征

RNN 分析眨眼频率与瞳孔运动

综合判断疲劳等级（共四级）

网页端展示用户疲劳状态

🚀 安装与运行（Installation & Usage）
克隆项目
bash
复制
编辑
git clone https://github.com/yourusername/fatigue-detector.git
cd project_root
安装依赖
bash
复制
编辑
pip install -r requirements.txt
启动前端
bash
复制
编辑
cd frontend
npm install
npm run dev
启动后端
bash
复制
编辑
cd ../backend
python app.py
📷 数据采集与标注（Data Collection & Annotation）
项目采用自建数据集，通过摄像头采集用户面部视频。采集过程中由观察者依据疲劳表现对每段数据进行等级标注。标注依据以下三个维度：

眨眼频率

眼睛闭合比例（PERCLOS）

瞳孔运动轨迹

💤 疲劳等级标注标准
等级 0：清醒（Alert）
眨眼频率正常，节奏自然

眼睛始终睁开，闭眼时间极短

目光灵活，注视点频繁移动

等级 1：轻度疲劳（Slight Fatigue）
偶尔眨眼频率偏快或偏慢

偶发短暂闭眼，眼神略显迟钝

目光活动稍有减少，但仍在变动

等级 2：中度疲劳（Moderate Fatigue）
眨眼节奏紊乱，闭眼时间明显延长

出现眼皮下垂，眨眼频繁或稀少

目光呆滞，注视点变化范围减小

等级 3：重度疲劳（Severe Fatigue）
出现打盹或长时间闭眼倾向

眨眼极少或剧烈频繁

几乎没有目光移动，眼神涣散或停滞

标注建议：每段视频建议以 5～10 秒为单位，观察视频并根据上述描述标注疲劳等级（0~3）。

🧠 模型设计（Model Architecture）
CNN 模块：用于提取眼部图像的静态特征，如眼睛开合状态

RNN 模块：建模时间序列数据，如眨眼频率与瞳孔轨迹

输出模块：融合多个指标，最终输出疲劳等级（0~3）

模型基于 PyTorch 实现，并支持 GPU 加速训练与推理。

📁 项目结构（Project Structure）
bash
复制
编辑
project_root/
├── frontend/           # 前端页面（React/Vite）
├── backend/            # 后端服务（Flask/FastAPI）
├── models/             # 模型训练脚本与模型文件
├── data/               # 数据集与标注文件
├── utils/              # 工具函数，如 PERCLOS 计算
├── demo/               # 示例视频与截图
└── README.md           # 项目说明文档
📸 示例与演示（Demo & Screenshots）
📹 示例视频路径：demo/demo_video.mp4
🖼️ 项目运行界面截图如下：
（此处可插入截图或演示 GIF）

📚 致谢与引用（Acknowledgements / Reference）
OpenFace - 面部关键点检测工具

PyTorch - 深度学习框架

其他引用文献与开源资源可在此处列出

🔧 TODO / 未来工作（Future Work）
 支持夜间或弱光环境识别优化

 支持多人同时疲劳检测

 支持移动端部署（PWA、小程序）

 添加更多行为特征，如打哈欠检测

