# Emotional Concealment (语义——韵律分析平台)

基于 **ECAPA-TDNN** + **SpeechMapper** 的多模态情感计算与认知解析系统。本系统旨在通过将语音的“语义特征”与“韵律特征”进行对齐分析，量化两者之间的偏差度，从而有效检测目标是否存在隐瞒真实情感或认知失调的状况。

## 🌟 核心特性
- **双通道分析**：结合 768 维高阶语义特征（SpeechMapper）与 192 维底层韵律特征（ECAPA-TDNN）。
- **跨模态对齐**：使用统一对齐架构，在原生的 192 维流形空间内计算余弦极角偏差，实现高效比对。
- **现代化可视化界面**：提供直观的 Web 界面，上传音频或实时录音即可生成偏差热力图和风险评估报告。

---

## 📁 目录结构

```text
├── demo/           # 前端 Web 界面
│   ├── index.html       # Web 页面主入口
│   ├── script.js        # 前端交互与绘图逻辑
│   ├── styles.css       # 页面样式
│   └── assets/          # 静态资源
├── backend/        # Speech2Vec 后端（基于 PyTorch / FastAPI 服务）
│   ├── app.py               # FastAPI 服务的主程序
│   ├── pipeline.py          # 特征提取及推断统一管线
│   ├── mel_spectrogram.py   # 梅尔频谱图提取 (80 通道)
│   ├── prosody_encoder.py   # 韵律编码器 (ECAPA-TDNN)
│   ├── semantic_encoder.py  # 语义编码器 (SpeechMapper)
│   ├── train.py             # 韵律模型自监督训练脚本
│   ├── train_semantic.py    # 语义模型跨模态蒸馏训练脚本
│   ├── download_dataset.py  # 数据集下载脚本 (LibriSpeech)
│   └── checkpoints/         # 训练或下载完毕的模型权重应放置于此
└── test/           # 示例测试音频文件文件夹
```

---

## 🚀 从零开始搭建指南 (How to run from scratch)

### 1. 系统要求与环境依赖
本平台后端使用 Python 开发，建议使用 Anaconda 或 Miniconda 管理环境。需确保本地机器具备可用的显卡环境（推荐 NVIDIA GPU + CUDA 架构），若无 GPU 亦可在 CPU 上进行推理。

#### 创建并激活 Conda 环境
```bash
conda create -n d2l python=3.10
conda activate d2l
```

#### 安装系统依赖
```bash
# 安装基础库
pip install torch torchaudio torchvision numpy soundfile

# 安装后端 Web 服务所需依赖库
pip install fastapi uvicorn python-multipart
```
> **注意：** 更多深度环境依赖细则及参数对照，参见 `backend/README.md`。

---

### 2. 模型下载与准备
如果是首次运行，需要准备相关的模型和数据：

1. **下载预训练韵律模型** (ECAPA-TDNN):
   本系统使用的韵律编码器预训练权重来源于 HuggingFace 的 SpeechBrain 仓库，你需要在运行前确保网络能访问 HF 自动下载，或者提前下载并放置在对应目录（比如 `backend/pretrained_models/`）。
2. **准备语义蒸馏数据集** (仅训练需要):
   ```bash
   cd backend
   python download_dataset.py  # 自动下载 LibriSpeech dev-clean 子集
   ```
3. **训练语义编码器** (需要配对数据跨模态拉偏蒸馏):
   如果你没有现成的 `semantic_best.pt` 且需要重头训练：
   ```bash
   python train_semantic.py --data_dir dataset_librispeech --batch_size 16 --epochs 50
   ```
> 运行时的模型权重（如 `encoder_best.pt` 与 `semantic_best.pt`）默认放在 `backend/checkpoints/` 目录下。如果你已有预训练的参数文件，请直接放置到该文件夹中。

---

### 3. 本地启动服务步骤

要在你的机器本地将前后端同时跑起来，按照以下两步启动：

#### Step 1: 启动 FastAPI 后端服务
打开**第一个控制台终端**：
```bash
conda activate d2l
cd backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
启动成功后，控制台会显示 API 服务运行正在监听 `http://127.0.0.1:8000`。
> **Tips:** FastAPI 自带交互式 Swagger 接口文档，可以通过访问 `http://127.0.0.1:8000/docs` 测试 API `/analyze` 端口。

#### Step 2: 启动前端 Web 界面
前端为纯静态的 HTML/CSS/JS 项目结构。你可以直接双击 `demo/index.html` 在 Chrome 或 Edge 浏览器打开页面；也可以使用简易 HTTP 服务器将其托管：
打开**第二个控制台终端**：
```bash
cd demo
# 任意起一个纯静态服务器（以 Python 为例）
python -m http.server 3000
```
随后，在浏览器中访问 `http://localhost:3000` 即可看到分析主界面。

---

### 4. 系统使用操作
1. **输入语音：** 在界面中央的输入控制面板中，点击麦克风进行实时语音收录，或者点击上传按钮/拖拽音频文件到虚线框内。
2. **深度分析：** 点击 “开始深度分析”，系统将自动发起 Fetch 请求给后端的 FastAPI 服务。
3. **解读结果：** 约数秒执行时间（包含梅尔截断、TDNN提取、特征运算及投影等），系统将返回详细的报告面板。
   - 绿色面板证明“自然流畅”
   - 红色警报面板证明存在高频的语义-韵律背离，表明有较高的掩饰情绪可能（高风险）。

## 📖 参考与进阶用法

对于纯命令行 (CLI) 形态的用户或者用于批量数据处理的用户，本平台直接在后端内置了 Pipeline 命令：
```bash
cd backend
python pipeline.py --audio ../test/your_audio.wav --json
```

更多具体关于模型结构维度、代码解析以及网络结构的原理详情，请参阅内置详细文档：[backend/README.md](backend/README.md)。
