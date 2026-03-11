from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import time
import numpy as np

# 因为我们在后端目录运行，可以直接导入 pipeline
try:
    from pipeline import Speech2VecPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: failed to import Speech2VecPipeline ({e})")
    PIPELINE_AVAILABLE = False


app = FastAPI(title="Semantic-Prosody Analysis API")

# 配置 CORS，允许前端直接调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境下允许所有域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 常驻内存加载模型管线
# 注意：这里我们优先读取 `checkpoints/semantic_best.pt`
semantic_ckpt = "checkpoints/semantic_best.pt"
if PIPELINE_AVAILABLE and os.path.exists(semantic_ckpt):
    print("Loading Speech2VecPipeline with semantic model...")
    pipeline = Speech2VecPipeline(semantic_checkpoint=semantic_ckpt, device="cuda")
else:
    print("Running in MOCK mode (Pipeline or Checkpoint not found).")
    pipeline = None


import torch
import torch.nn.functional as F
from torch import nn

import math

class CrossModalAligner(nn.Module):
    """跨模态对齐计算器
    
    直接将 192 维的语义特征与 192 维韵律特征对齐，并计算余弦夹角偏差。
    """
    def __init__(self):
        super().__init__()
        
    @torch.no_grad()
    def forward(self, prosody_vec: np.ndarray, semantic_vec: np.ndarray) -> float:
        # 两者现在在架构层面都已经原生输出 192 维
        p_tensor = torch.from_numpy(prosody_vec).float().view(1, -1)
        s_tensor = torch.from_numpy(semantic_vec).float().view(1, -1)
        
        # 直接计算原生 192 维空间的余弦相似度
        cos_sim = F.cosine_similarity(p_tensor, s_tensor, dim=-1).item()
        
        # 3. [差值放大机制] 解决高维正交衰减
        # 问题：在 1024 维空间中，未训练的随机投影会使得任何输入的余弦相似度都极其趋近于 0（即默认正交）。
        # 这导致所有音频算出来的理论偏差都在 1.0 附近徘徊，区分度可能只有 0.01 左右。
        # 方案：利用“温度缩放 (Temperature Scaling) + Sigmoid”将这微小的夹角偏差作指数级放大！
        
        temperature = 40.0  # 敏感度系数：越大，微小的输入差异就会被放大得越极端
        amplified_sim = cos_sim * temperature 
        
        # 将被放大的相似度通过 Sigmoid 压缩成 0 ~ 1.0 的平滑概率分布（一致性）
        consistency = 1.0 / (1.0 + math.exp(-amplified_sim))
        
        # 偏差度 = 1 - 一致性（0.5为中位线，<0.5表现自然，>0.5表现异常）
        deviation = 1.0 - consistency
        
        return deviation


# 初始化对齐器
aligner = CrossModalAligner()


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """接收前端音频，提取特征，计算偏差度"""
    
    # 1. 保存上传文件到临时路径
    start_time = time.time()
    suffix = os.path.splitext(file.filename)[1]
    if not suffix:
        suffix = ".wav"
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        content = await file.read()
        temp_audio.write(content)
        temp_file_path = temp_audio.name

    try:
        # 2. 调用核心管线
        if pipeline:
            print(f"Processing audio: {temp_file_path}")
            result = pipeline.process(temp_file_path)
            
            p_vec = result["prosody_vector"]
            s_vec = result["semantic_vector"]
            
            # 【跨模态偏移量化计算】
            # 使用对齐器计算 1024 维下的夹角余弦偏差
            deviation = aligner(p_vec, s_vec)

            # 额外对齐指标（用于前端可视化）
            p_np = np.asarray(p_vec, dtype=np.float32)
            s_np = np.asarray(s_vec, dtype=np.float32)
            metric_dim = int(min(p_np.shape[0], s_np.shape[0]))
            p_m = p_np[:metric_dim]
            s_m = s_np[:metric_dim]
            denom = float(np.linalg.norm(p_m) * np.linalg.norm(s_m) + 1e-12)
            cos_sim = float(np.dot(p_m, s_m) / denom)
            cos_sim_clamped = max(-1.0, min(1.0, cos_sim))
            angle_deg = float(math.degrees(math.acos(cos_sim_clamped)))

            temperature = 40.0
            amplified_sim = float(cos_sim * temperature)
            consistency = float(1.0 / (1.0 + math.exp(-amplified_sim)))
            
            # 直接应用 0.5 偏差判定阈值
            is_high_risk = deviation > 0.5
            
            response = {
                "status": "success",
                "filename": file.filename,
                "process_time": round(time.time() - start_time, 2),
                "deviation_score": round(deviation, 3), # 0~2 的实数
                "is_high_risk": is_high_risk,
                "alignment": {
                    "cosine_similarity": cos_sim,
                    "angle_degrees": angle_deg,
                    "temperature": temperature,
                    "amplified_similarity": amplified_sim,
                    "consistency": consistency,
                    "metrics_dim": metric_dim,
                },
                "features": {
                    "prosody_dim": len(p_vec),
                    "semantic_dim": len(s_vec),
                    "aligned_dim": 192,
                    "prosody_sample": p_vec[:5].tolist(),  
                    "semantic_sample": s_vec[:5].tolist(),
                    "prosody_vector": p_vec.tolist(),
                    "semantic_vector": s_vec.tolist(),
                }
            }
        else:
            # Mock 模式
            time.sleep(2)
            mock_dev = float(np.random.uniform(0.1, 0.9))
            mock_cos = float(np.random.uniform(-0.2, 0.8))
            mock_cos_clamped = max(-1.0, min(1.0, mock_cos))
            mock_angle = float(math.degrees(math.acos(mock_cos_clamped)))
            mock_temperature = 40.0
            mock_amplified = float(mock_cos * mock_temperature)
            mock_consistency = float(1.0 / (1.0 + math.exp(-mock_amplified)))
            response = {
                "status": "mock_success",
                "filename": file.filename,
                "process_time": round(time.time() - start_time, 2),
                "deviation_score": round(mock_dev, 3),
                "is_high_risk": mock_dev > 0.5,
                "alignment": {
                    "cosine_similarity": mock_cos,
                    "angle_degrees": mock_angle,
                    "temperature": mock_temperature,
                    "amplified_similarity": mock_amplified,
                    "consistency": mock_consistency,
                    "metrics_dim": 192,
                },
                "features": {
                    "prosody_dim": 192,
                    "semantic_dim": 768,
                    "aligned_dim": 192,
                    "prosody_sample": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "semantic_sample": [0.01, -0.02, 0.05, 0.0, 0.1],
                    "prosody_vector": [float(np.random.uniform(-1, 1)) for _ in range(192)],
                    "semantic_vector": [float(np.random.uniform(-1, 1)) for _ in range(192)],
                }
            }
            
        return response
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return {"status": "error", "message": str(e)}
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    # 让服务器监听在 8000 端口
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
