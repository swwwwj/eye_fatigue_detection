🧠 基于CNN-RNN与Gan增强的青年面部疲劳识别系统

📌 项目简介
本项目是一个基于CNN-RNN实现的面部疲劳识别系统，创新性结合静态图像与动态视频分析，实现对青年用户婆老状态的精准和实时检测。系统基于自主采集制作的数据集进行训练，考虑到样本较少，引入Gan进行数据增强，以提升模型的泛化能力。通过实时视频捕捉用户静态面部状况和动态眼部信息，系统可通过模型捕捉疲劳信息并实时反馈。项目紧贴现代年轻人长时间使用电子设备的实际问题，旨在以智能科技手段引导健康用眼与作息习惯的建立，为数字生活注入关怀与守护。

📌 项目安装
1.克隆项目
git clone https://github.com/yourusername/fatigue-detector.git

2.进入项目目录
cd project_root

3.安装项目依赖
pip install -r quirements.txt

4.运行脚本
cd D:\github\eye_fatigue_detection\frontened && python app.py --model_dir "" --data_dir "" --template_dir ""

📌 数据集制作
项目采用自建数据集，使用opencv库通过摄像头采集用户面部照片，使用dlib对照片进行裁剪，抓取有效面部特征。采集过程中依据被采集着的疲劳表现进行等级标注，主要依据以下几个维度：
眨眼频率，眼睛闭合比例（PERCLOS），瞳孔运动轨迹，其他相关面部特征。

详细分类标准：
🔹 等级 0：清醒（Alert）
眨眼行为：
频率正常，节奏自然
闭眼时间极短，基本不影响视觉专注

目光特征：
眼睛始终睁开，目光灵活
注视点频繁变化，跟踪物体能力良好

面部特征：
面部表情自然、活跃
无明显下垂、僵硬等疲劳表情

🔸 等级 1：轻度疲劳（Slight Fatigue）
眨眼行为：
偶尔眨眼频率略高或略低
偶发性短暂闭眼现象

目光特征：
眼神略显迟钝
注视点移动略减少

面部特征：
表情开始趋于平静，轻微松弛
眼睑偶有轻微下垂现象

🟠 等级 2：中度疲劳（Moderate Fatigue）
眨眼行为：
频率明显异常（过快或过慢）
闭眼时间延长，有“困意”迹象

目光特征：
呆滞感增强，注视点变化范围收窄

面部特征：
眼皮下垂明显，肌肉放松
面部表情反应迟缓

🔴 等级 3：重度疲劳（Severe Fatigue）
眨眼行为：
长时间闭眼或出现打盹现象
眨眼极少，或出现剧烈频繁抖动

目光特征：
几乎无目光移动，眼神涣散或停滞

面部特征：
面部表情呆滞、缺乏活力
眼睑持续下垂，肌肉极度松弛

🧠 模型设计
CNN模块：用于提取面部静态特征

RNN模块：建模时间序列数据，如眨眼频率与瞳孔轨迹

detector模块：融合多个指标，最终输出疲劳等级（0~3）

型基于 PyTorch 实现，并支持 GPU 加速训练与推理。

✨ 反馈模块

1.疲劳检测报告
系统接入LLM，对用户的面部疲劳数据进行连续追踪与语义分析。用户可随时一键生成个性化疲劳检测报告，全面展示疲劳波动趋势、异常行为分析及科学建议，帮助用户更好地认知和管理自身状态，养成健康用眼习惯。

2.数据展示
系统内置可视化模块，通过柱状图、折线图和饼图，实时展现用户在不同时间段的疲劳等级分布、波动趋势、疲劳占比等关键信息。

项目结构
EYE FATIGUE DETECTION/
│
├── frontend/
│   ├── templates/
│   │   └── index.html
│   ├── app.py
│   └── data_analysis.py
│
├── models/
│   ├── best_discriminator.pth
│   ├── best_generator.pth
│   ├── fatigue_rnn.pth
│   ├── scaler.pkl
│   └── shape_predictor_68_face_landmarks.dat
│
├── training/
│   ├── CNN+GAN/
│   │   ├── CNN+GAN.py
│   │   └── detector.py
│   ├── RNN/
│   │   ├── detector.py
│   │   ├── extract_features_from_npy.py
│   │   └── RNN.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── run.sh

🧠 我们不仅识别疲劳，更助你洞察疲劳、管理疲劳、摆脱疲劳。