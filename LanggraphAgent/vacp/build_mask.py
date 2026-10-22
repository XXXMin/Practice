"""使用你已有的 VACP 代码构建词汇表掩码"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model.factory import get_local_model
from vacp.vocab_mask import build_vocabulary_mask
from vacp.config import config

print("=" * 60)
print("构建 VACP 词汇表掩码（使用你复现的代码）")
print("=" * 60)

#加载模型
print("\n【1】加载本地模型...")
model, tokenizer = get_local_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 模型加载成功，设备: {device}")

#准备验证文本（从你的知识库中提取）
print("\n【2】准备验证文本...")
validation_texts = [
    "扫地机器人吸力有多大",
    "小户型适合哪种扫地机器人",
    "扫地机器人怎么维修",
    "电池续航时间多长",
    "噪音大吗",
    "如何连接手机APP",
]
print(f"验证文本数量: {len(validation_texts)}")

#直接调用你已有的 build_vocabulary_mask
print("\n【3】构建词汇表掩码...")
vocab_mask = build_vocabulary_mask(
    model=model,
    tokenizer=tokenizer,
    validation_texts=validation_texts,
    prob_threshold=config.prob_threshold,  # 使用你的配置
    batch_size=config.batch_size
)

#保存掩码
print("\n【4】保存掩码...")
save_path = Path(__file__).parent / "vocab_mask.pt"
torch.save(vocab_mask, save_path)

print(f"\n✅ 掩码已保存到: {save_path}")
print(f"   有效词汇数: {vocab_mask.sum().item()} / {len(vocab_mask)}")
print(f"   保留比例: {100 * vocab_mask.sum().item() / len(vocab_mask):.1f}%")