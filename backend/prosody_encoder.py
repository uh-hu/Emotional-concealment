from __future__ import annotations

"""
韵律编码器模块 — 基于 ECAPA-TDNN (纯 PyTorch 实现)
Prosody Encoder — ECAPA-TDNN (Pure PyTorch Implementation)

架构参照论文:
  "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
   in Time Delay Neural Network" (Desplanques et al., 2020)

核心组件:
  1. SE-Res2Block: Squeeze-Excitation + Res2Net 结构的 1D 卷积块
  2. Attentive Statistical Pooling: 注意力加权的统计池化层
  3. 全连接层: 输出 192 维韵律嵌入向量

输入: 梅尔频谱图 [batch, n_mels=80, T]
输出: 韵律向量 [batch, 192]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 基础组件
# ═══════════════════════════════════════════════════════════════


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) 块。

    通过全局信息自适应地重新校准通道特征。
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        scale = self.se(x).unsqueeze(-1)  # [B, C, 1]
        return x * scale


class Res2NetBlock(nn.Module):
    """Res2Net 风格的多尺度 1D 卷积块。

    将通道分为 scale 组，每组依次卷积并累加，
    捕获多种感受野的时域特征。
    """

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: int = 1, scale: int = 8):
        super().__init__()
        assert channels % scale == 0, f"channels({channels}) must be divisible by scale({scale})"
        self.scale = scale
        self.width = channels // scale

        self.convs = nn.ModuleList([
            nn.Conv1d(
                self.width, self.width,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2,
            )
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.width)
            for _ in range(scale - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        chunks = torch.chunk(x, self.scale, dim=1)
        outputs = [chunks[0]]

        for i in range(1, self.scale):
            if i == 1:
                inp = chunks[i]
            else:
                inp = chunks[i] + outputs[-1]
            out = F.relu(self.bns[i - 1](self.convs[i - 1](inp)), inplace=True)
            outputs.append(out)

        return torch.cat(outputs, dim=1)


class SERes2Block(nn.Module):
    """SE-Res2Block: ECAPA-TDNN 的核心构建块。

    结构: Conv1D → Res2Net → Conv1D → SE → 残差连接
    """

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: int = 1, scale: int = 8, reduction: int = 8):
        super().__init__()

        self.block = nn.Sequential(
            # 第一个 1x1 卷积
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),

            # Res2Net 多尺度卷积
            Res2NetBlock(channels, kernel_size, dilation, scale),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),

            # 第二个 1x1 卷积
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),

            # Squeeze-Excitation
            SEBlock(channels, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AttentiveStatisticsPooling(nn.Module):
    """注意力统计池化层。

    对时间维度进行注意力加权统计池化，
    输出加权均值和加权标准差的拼接。
    """

    def __init__(self, channels: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(attention_dim),
            nn.Conv1d(attention_dim, channels, kernel_size=1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        alpha = self.attention(x)  # [B, C, T]

        # 加权平均
        mean = (alpha * x).sum(dim=-1)  # [B, C]

        # 加权标准差
        var = (alpha * (x ** 2)).sum(dim=-1) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-9))  # [B, C]

        # 拼接
        return torch.cat([mean, std], dim=1)  # [B, 2C]


# ═══════════════════════════════════════════════════════════════
# ECAPA-TDNN 主网络
# ═══════════════════════════════════════════════════════════════


class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN: 韵律/说话人嵌入提取网络。

    Architecture:
        Input [B, 80, T]
          → Conv1D stem (80 → C)
          → 3 × SERes2Block (dilations: 2, 3, 4)
          → Multi-layer Feature Aggregation (MFA)
          → Conv1D bottleneck
          → Attentive Statistics Pooling
          → FC → BN → 192-dim embedding

    Parameters
    ----------
    in_channels : int
        输入特征维度（梅尔通道数），默认 80。
    channels : int
        中间层通道数，默认 1024。
    embedding_dim : int
        输出嵌入维度，默认 192。
    """

    def __init__(
        self,
        in_channels: int = 80,
        channels: int = 1024,
        embedding_dim: int = 192,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.embedding_dim = embedding_dim

        # ─── Stem ───
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # ─── SE-Res2Blocks (膨胀卷积: 2, 3, 4) ───
        self.block1 = SERes2Block(channels, kernel_size=3, dilation=2)
        self.block2 = SERes2Block(channels, kernel_size=3, dilation=3)
        self.block3 = SERes2Block(channels, kernel_size=3, dilation=4)

        # ─── Multi-layer Feature Aggregation ───
        # 拼接 stem 输出 + 3 个 block 输出，然后用 1x1 卷积压缩
        self.mfa = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # ─── Attentive Statistics Pooling ───
        self.asp = AttentiveStatisticsPooling(channels, attention_dim=128)

        # ─── Final embedding ───
        self.fc = nn.Linear(channels * 2, embedding_dim)  # *2 因为 ASP 拼接了 mean+std
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            梅尔频谱图，形状 [batch, n_mels=80, T]。

        Returns
        -------
        embedding : torch.Tensor
            韵律嵌入向量，形状 [batch, 192]。
        """
        # Stem
        x0 = self.stem(x)  # [B, C, T]

        # 3 个 SE-Res2Block
        x1 = self.block1(x0)  # [B, C, T]
        x2 = self.block2(x1)  # [B, C, T]
        x3 = self.block3(x2)  # [B, C, T]

        # Multi-layer Feature Aggregation
        # 论文中是拼接所有中间层 (跳过 stem 输出 x0，只用 block 输出)
        x_cat = torch.cat([x1, x2, x3], dim=1)  # [B, 3C, T]
        x_mfa = self.mfa(x_cat)  # [B, C, T]

        # Attentive Statistics Pooling
        x_pool = self.asp(x_mfa)  # [B, 2C]

        # Final embedding
        emb = self.fc(x_pool)  # [B, 192]
        emb = self.bn(emb)

        return emb


# ═══════════════════════════════════════════════════════════════
# 高级封装
# ═══════════════════════════════════════════════════════════════


class ProsodyEncoder:
    """韵律编码器 — 封装 ECAPA-TDNN 推理接口。

    Parameters
    ----------
    checkpoint_path : str, optional
        训练好的模型权重路径（.pt 文件）。
        如果为 None，则使用随机初始化的模型（未训练）。
    device : str, optional
        计算设备。None 则自动选择。
    embedding_dim : int
        输出嵌入维度，默认 192。
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = None,
        embedding_dim: int = 192,
    ):
        self.embedding_dim = embedding_dim

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 构建模型
        self.model = ECAPA_TDNN(
            in_channels=80,
            channels=1024,
            embedding_dim=embedding_dim,
        ).to(self.device)

        # 加载权重
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # 兼容包含 "model." 前缀的 checkpoint
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            print(f"[ProsodyEncoder] 已加载权重: {checkpoint_path}")
        else:
            print("[ProsodyEncoder] 注意: 使用随机初始化模型（未训练）")

        self.model.eval()
        print(f"[ProsodyEncoder] 设备: {self.device} | 嵌入维度: {self.embedding_dim}")

        # 打印模型参数量
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[ProsodyEncoder] 模型参数量: {n_params:,} ({n_params / 1e6:.1f}M)")

    @torch.no_grad()
    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """从梅尔频谱图中提取韵律嵌入向量。

        Parameters
        ----------
        mel : torch.Tensor
            梅尔频谱图，形状 [batch, 80, T] 或 [80, T]。

        Returns
        -------
        embedding : torch.Tensor
            韵律嵌入向量，形状 [batch, 192]。
        """
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        mel = mel.to(self.device)
        embedding = self.model(mel)
        return embedding

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


# ─────────────────────────── CLI 入口 ───────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, ".")

    parser = argparse.ArgumentParser(description="ECAPA-TDNN Prosody Encoder")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint (.pt)")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    from mel_spectrogram import MelSpectrogramExtractor

    # 提取梅尔频谱图
    extractor = MelSpectrogramExtractor()
    result = extractor.from_file(args.audio)
    mel = result["mel"]

    # 编码
    encoder = ProsodyEncoder(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    embedding = encoder.encode(mel)

    print("=" * 50)
    print("Prosody Encoder Output")
    print("=" * 50)
    print(f"  Input mel:       {list(mel.shape)}")
    print(f"  Embedding:       {list(embedding.shape)}")
    print(f"  First 10 dims:   {embedding[0, :10].cpu().tolist()}")
    print(f"  L2 norm:         {embedding.norm(dim=-1).item():.4f}")
    print("=" * 50)
