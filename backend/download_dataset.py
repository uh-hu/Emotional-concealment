import os
import tarfile
import urllib.request
from pathlib import Path

"""
下载并处理 LibriSpeech dev-clean 数据集
专为 SpeechMapper 语义模型训练准备

本脚本将：
1. 从 OpenSLR 下载 LibriSpeech dev-clean.tar.gz (约 337MB)
2. 解压文件
3. 解析所有的 .txt 文本转录文件
4. 生成 metadata.csv 供 train_semantic.py 使用
"""

URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
DOWNLOAD_DIR = Path("dataset_librispeech")
TAR_FILE = DOWNLOAD_DIR / "dev-clean.tar.gz"
EXTRACT_DIR = DOWNLOAD_DIR / "LibriSpeech" / "dev-clean"
METADATA_FILE = DOWNLOAD_DIR / "metadata.csv"

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        print(f"\r下载进度: {min(100.0, percent):.1f}% ({downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)", end="")
    else:
        print(f"\r已下载: {downloaded/(1024*1024):.1f}MB", end="")

def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 下载
    if not TAR_FILE.exists():
        print(f"开始下载 LibriSpeech dev-clean (约 337MB)...")
        print(f"URL: {URL}")
        try:
            urllib.request.urlretrieve(URL, TAR_FILE, reporthook=download_progress)
            print("\n下载完成！")
        except Exception as e:
            print(f"\n下载失败: {e}")
            print("请尝试手动下载 https://www.openslr.org/resources/12/dev-clean.tar.gz 并放入 backend/dataset_librispeech 目录下")
            return
    else:
        print("压缩包已存在，跳过下载。")

    # 2. 解压
    if not EXTRACT_DIR.exists():
        print("开始解压...")
        with tarfile.open(TAR_FILE, "r:gz") as tar:
            tar.extractall(path=DOWNLOAD_DIR)
        print("解压完成！")
    else:
        print("已解压，跳过解压。")

    # 3. 处理转录生成 metadata.csv
    print(f"正在扫描并生成映射文件 {METADATA_FILE}...")
    transcripts = []
    
    # LibriSpeech 结构：READER/CHAPTER/transcripts.txt
    for txt_file in EXTRACT_DIR.rglob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    audio_id = parts[0]
                    text = parts[1]
                    # 寻找对应的 FLAC 文件
                    audio_file = txt_file.parent / f"{audio_id}.flac"
                    if audio_file.exists():
                        # 使用相对路径
                        rel_path = audio_file.relative_to(DOWNLOAD_DIR)
                        transcripts.append(f"{rel_path}|{text}")

    if not transcripts:
        print("未找到任何配对的音频和文本！")
        return

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        f.write("audio_path|text\n")
        f.write("\n".join(transcripts))
    
    print(f"\n全部完成！共处理了 {len(transcripts)} 条数据。")
    print(f"你现在可以运行: python train_semantic.py --data_dir {DOWNLOAD_DIR}")

if __name__ == "__main__":
    main()
