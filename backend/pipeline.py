from __future__ import annotations

"""
Prosody2Vec 统一管线
Prosody2Vec Unified Pipeline

将声学特征预处理（梅尔频谱图）和韵律编码器（ECAPA-TDNN）组合为
一站式管线：音频文件 → 192 维韵律向量。
"""

import torch
import numpy as np
import json
from pathlib import Path

from mel_spectrogram import MelSpectrogramExtractor
from prosody_encoder import ProsodyEncoder


class Prosody2VecPipeline:
    """Prosody2Vec 完整管线。

    音频文件 → 梅尔频谱图 → ECAPA-TDNN → 192 维韵律向量

    Parameters
    ----------
    checkpoint_path : str, optional
        训练好的编码器权重路径（.pt 文件）。
        如果为 None，使用随机初始化模型（仅测试用）。
    device : str, optional
        计算设备。None 则自动选择。
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("[Pipeline] Initializing mel-spectrogram extractor...")
        self.mel_extractor = MelSpectrogramExtractor()

        print("[Pipeline] Initializing prosody encoder...")
        self.encoder = ProsodyEncoder(
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

        print(f"[Pipeline] Ready | Device: {self.device}")

    def process(self, audio_path: str | Path) -> dict:
        """处理单个音频文件，提取韵律向量。

        Parameters
        ----------
        audio_path : str or Path
            音频文件路径。

        Returns
        -------
        result : dict
            - "prosody_vector": np.ndarray [192]
            - "mel_spectrogram": np.ndarray [80, T]
            - "metadata": dict
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Step 1: 提取梅尔频谱图
        mel_result = self.mel_extractor.from_file(audio_path)
        mel = mel_result["mel"]  # [1, 80, T]

        # Step 2: 编码
        embedding = self.encoder.encode(mel)  # [1, 192]

        # 转 numpy
        prosody_vector = embedding.squeeze(0).cpu().numpy()  # [192]
        mel_np = mel.squeeze(0).cpu().numpy()                # [80, T]

        return {
            "prosody_vector": prosody_vector,
            "mel_spectrogram": mel_np,
            "metadata": {
                "audio_path": str(audio_path.resolve()),
                "duration_sec": mel_result["duration_sec"],
                "sample_rate": mel_result["sample_rate"],
                "num_frames": mel_result["num_frames"],
                "embedding_dim": prosody_vector.shape[0],
                "device": self.device,
            },
        }

    def process_batch(self, audio_paths: list) -> list:
        """批量处理多个音频文件。"""
        results = []
        for i, path in enumerate(audio_paths):
            print(f"  [{i+1}/{len(audio_paths)}] Processing: {Path(path).name}")
            result = self.process(path)
            results.append(result)
        return results


# ─────────────────────────── CLI ───────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prosody2Vec Pipeline: Audio -> 192-dim Prosody Vector"
    )
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--checkpoint", default=None, help="Encoder checkpoint (.pt)")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--output", default=None, help="Save vector to .npy file")
    parser.add_argument("--json", action="store_true", help="Output metadata as JSON")

    args = parser.parse_args()

    pipeline = Prosody2VecPipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    result = pipeline.process(args.audio)

    vec = result["prosody_vector"]
    meta = result["metadata"]

    print()
    print("=" * 60)
    print("  Prosody2Vec Result")
    print("=" * 60)
    print(f"  Audio:          {Path(args.audio).name}")
    print(f"  Duration:       {meta['duration_sec']}s")
    print(f"  Mel frames:     {meta['num_frames']}")
    print(f"  Embedding dim:  {meta['embedding_dim']}")
    print(f"  Device:         {meta['device']}")
    print(f"  L2 norm:        {np.linalg.norm(vec):.4f}")
    print(f"  First 5 dims:   {vec[:5].tolist()}")
    print("=" * 60)

    if args.output:
        np.save(args.output, vec)
        print(f"\n  Vector saved to: {args.output}")

    if args.json:
        json_output = {"prosody_vector": vec.tolist(), "metadata": meta}
        print("\n" + json.dumps(json_output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
