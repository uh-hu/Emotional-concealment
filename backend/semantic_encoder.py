from __future__ import annotations

"""
语义编码器模块 — 基于 SpeechMapper (ICASSP 2026)
Semantic Encoder — SpeechMapper Architecture

论文参考:
  "SpeechMapper: Speech-to-text Embedding Projector for LLMs"
  (Mohapatra, Boito & Calapodescu, ICASSP 2026)

核心组件:
  1. PositionalEncoding: 正弦位置编码
  2. SpeechMapperBlock: Conv1D 下采样 + Transformer 编码器层
  3. AttentivePooling: 注意力池化（变长序列 → 固定向量）
  4. SpeechMapper: 完整网络

输入: 梅尔频谱图 [batch, n_mels=80, T]
输出: 语义向量 [batch, semantic_dim]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 基础组件
# ═══════════════════════════════════════════════════════════════


class PositionalEncoding(nn.Module):
    """正弦位置编码。

    为 Transformer 输入序列注入位置信息。

    Parameters
    ----------
    d_model : int
        模型维度。
    max_len : int
        支持的最大序列长度。
    dropout : float
        Dropout 比率。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            形状 [B, T, d_model]。
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AttentivePooling(nn.Module):
    """注意力池化层。

    将变长序列压缩为固定维度的向量表示。

    Parameters
    ----------
    d_model : int
        输入特征维度。
    attention_dim : int
        注意力中间维度。
    """

    def __init__(self, d_model: int, attention_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            形状 [B, T, d_model]。
        mask : torch.Tensor, optional
            形状 [B, T]，True 表示有效位置。

        Returns
        -------
        pooled : torch.Tensor
            形状 [B, d_model]。
        """
        # 注意力权重
        attn_scores = self.attention(x).squeeze(-1)  # [B, T]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T]

        # 加权求和
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [B, d_model]
        return pooled


# ═══════════════════════════════════════════════════════════════
# SpeechMapper 核心结构
# ═══════════════════════════════════════════════════════════════


class SpeechMapperBlock(nn.Module):
    """SpeechMapper Block — 论文核心构建块。

    结构:
        Conv1D (kernel=6, stride=2) → LayerNorm → GELU
        → Transformer Encoder × n_layers
        → 序列下采样 2×

    Parameters
    ----------
    d_model : int
        Transformer 模型维度，默认 1024。
    n_transformer_layers : int
        Transformer 编码器层数，默认 6。
    nhead : int
        多头注意力头数，默认 8。
    dim_feedforward : int
        Transformer FFN 中间维度，默认 2048。
    conv_kernel_size : int
        下采样卷积核大小，默认 6。
    conv_stride : int
        下采样卷积步幅，默认 2。
    dropout : float
        Dropout 比率，默认 0.1。
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_transformer_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        conv_kernel_size: int = 6,
        conv_stride: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ─── 卷积下采样层 ───
        # Conv1D with kernel=6, stride=2 实现序列长度下采样 2×
        padding = (conv_kernel_size - 1) // 2
        self.conv_downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size,
                      stride=conv_stride, padding=padding),
            nn.GELU(),
        )
        self.conv_norm = nn.LayerNorm(d_model)

        # ─── 位置编码 ───
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # ─── Transformer 编码器层 ───
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # 使用 [B, T, C] 格式
            norm_first=True,    # Pre-LN (更稳定)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            形状 [B, T, d_model]。

        Returns
        -------
        out : torch.Tensor
            形状 [B, T', d_model]，其中 T' ≈ T // 2。
        """
        # Conv1D 下采样: [B, T, C] → [B, C, T] → Conv → [B, C, T'] → [B, T', C]
        x_conv = x.transpose(1, 2)              # [B, C, T]
        x_conv = self.conv_downsample(x_conv)    # [B, C, T']
        x_conv = x_conv.transpose(1, 2)          # [B, T', C]
        x_conv = self.conv_norm(x_conv)

        # 位置编码
        x_pos = self.pos_encoder(x_conv)

        # Transformer 编码
        out = self.transformer_encoder(x_pos)    # [B, T', C]

        return out


class SpeechMapper(nn.Module):
    """SpeechMapper: 语音到语义向量的投射网络。

    Architecture:
        Input [B, 80, T]
          → Input Projection (80 → d_model)
          → SpeechMapperBlock × n_blocks
          → Attentive Pooling (变长 → 固定)
          → FFN Projector (d_model → hidden → semantic_dim)

    论文核心参数:
        - d_model = 1024
        - n_transformer_layers = 6 (per block)
        - conv kernel = 6, stride = 2
        - FFN: 1024 → 2048 → output_dim

    Parameters
    ----------
    in_channels : int
        输入特征维度（梅尔通道数），默认 80。
    d_model : int
        模型内部维度，默认 1024。
    n_blocks : int
        SpeechMapperBlock 重复次数，默认 1。
    n_transformer_layers : int
        每个 Block 内 Transformer 层数，默认 6。
    nhead : int
        注意力头数，默认 8。
    dim_feedforward : int
        Transformer FFN 维度，默认 2048。
    semantic_dim : int
        最终输出语义向量维度，默认 768。
    ffn_hidden : int
        投射头 FFN 中间维度，默认 2048。
    dropout : float
        Dropout 比率，默认 0.1。
    """

    def __init__(
        self,
        in_channels: int = 80,
        d_model: int = 1024,
        n_blocks: int = 1,
        n_transformer_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        semantic_dim: int = 768,
        ffn_hidden: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.d_model = d_model
        self.semantic_dim = semantic_dim

        # ─── 输入投影: 梅尔通道 → d_model ───
        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(d_model)

        # ─── SpeechMapper Blocks ───
        self.blocks = nn.ModuleList([
            SpeechMapperBlock(
                d_model=d_model,
                n_transformer_layers=n_transformer_layers,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ])

        # ─── Attentive Pooling ───
        self.pooling = AttentivePooling(d_model, attention_dim=256)

        # ─── FFN Projector: d_model → hidden → semantic_dim ───
        # 对应论文中 1024 → 2048 → output_dim
        self.projector = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.LayerNorm(ffn_hidden),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, semantic_dim),
            nn.LayerNorm(semantic_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            梅尔频谱图，形状 [batch, n_mels=80, T]。

        Returns
        -------
        semantic_vector : torch.Tensor
            语义嵌入向量，形状 [batch, semantic_dim]。
        """
        # Input projection: [B, 80, T] → [B, d_model, T]
        x = self.input_projection(x)
        # → [B, T, d_model]
        x = x.transpose(1, 2)
        x = self.input_norm(x)

        # SpeechMapper Blocks
        for block in self.blocks:
            x = block(x)  # [B, T', d_model]，每个 block 下采样 2×

        # Attentive Pooling: [B, T', d_model] → [B, d_model]
        x = self.pooling(x)

        # FFN Projector: [B, d_model] → [B, semantic_dim]
        semantic_vector = self.projector(x)

        return semantic_vector


# ═══════════════════════════════════════════════════════════════
# 高级封装
# ═══════════════════════════════════════════════════════════════


class SemanticEncoder:
    """语义编码器 — 封装 SpeechMapper 推理接口。

    支持两种加载模式:
        1. 自训练的 .pt checkpoint
        2. 随机初始化（仅测试用）

    Parameters
    ----------
    checkpoint_path : str, optional
        训练好的模型权重路径（.pt 文件）。
    device : str, optional
        计算设备。None 则自动选择。
    semantic_dim : int
        输出语义向量维度，默认 768。
    d_model : int
        模型内部维度，默认 1024。
    n_blocks : int
        SpeechMapperBlock 数量，默认 1。
    n_transformer_layers : int
        每个 Block 内 Transformer 层数，默认 6。
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = None,
        semantic_dim: int = 768,
        d_model: int = 1024,
        n_blocks: int = 1,
        n_transformer_layers: int = 6,
    ):
        self.semantic_dim = semantic_dim

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 构建模型
        self.model = SpeechMapper(
            in_channels=80,
            d_model=d_model,
            n_blocks=n_blocks,
            n_transformer_layers=n_transformer_layers,
            semantic_dim=semantic_dim,
        ).to(self.device)

        # ─── 加载权重 ───
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        else:
            self.model.eval()
            print("[SemanticEncoder] 注意: 使用随机初始化模型（未训练）")

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[SemanticEncoder] SpeechMapper | {n_params:,} params "
              f"({n_params / 1e6:.1f}M) | dim={semantic_dim} | 设备: {self.device}")

    def _load_checkpoint(self, checkpoint_path: str):
        """加载训练好的 .pt checkpoint。"""
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        # 兼容 DDP 保存的 "module." 前缀
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("model.", "")
            cleaned[k] = v

        self.model.load_state_dict(cleaned)
        self.model.eval()
        print(f"[SemanticEncoder] 已加载权重: {checkpoint_path}")

    @torch.no_grad()
    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """从梅尔频谱图中提取语义嵌入向量。

        Parameters
        ----------
        mel : torch.Tensor
            梅尔频谱图，形状 [batch, 80, T] 或 [80, T]。

        Returns
        -------
        embedding : torch.Tensor
            语义嵌入向量，形状 [batch, semantic_dim]。
        """
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        mel = mel.to(self.device)
        embedding = self.model(mel)

        return embedding

    def get_embedding_dim(self) -> int:
        return self.semantic_dim


# ─────────────────────────── CLI 入口 ───────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, ".")

    parser = argparse.ArgumentParser(description="SpeechMapper Semantic Encoder")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint (.pt)")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--semantic_dim", type=int, default=768,
                        help="Semantic vector dimension")
    args = parser.parse_args()

    from mel_spectrogram import MelSpectrogramExtractor

    # 提取梅尔频谱图
    extractor = MelSpectrogramExtractor()
    result = extractor.from_file(args.audio)
    mel = result["mel"]

    # 编码
    encoder = SemanticEncoder(
        checkpoint_path=args.checkpoint,
        device=args.device,
        semantic_dim=args.semantic_dim,
    )
    embedding = encoder.encode(mel)

    print("=" * 50)
    print("SpeechMapper Semantic Encoder Output")
    print("=" * 50)
    print(f"  Input mel:       {list(mel.shape)}")
    print(f"  Embedding:       {list(embedding.shape)}")
    print(f"  First 10 dims:   {embedding[0, :10].cpu().tolist()}")
    print(f"  L2 norm:         {embedding.norm(dim=-1).item():.4f}")
    print("=" * 50)
