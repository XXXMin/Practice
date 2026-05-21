import os
# 设置环境变量，告诉下载工具使用镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
import random
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import config
from data.squad_loder import load_squad_data, split_data
from vacp import (
    build_vocabulary_mask, validate_mask_coverage,VACPScorer,
    calibrate_threshold, evaluate_coverage, tune_temperature
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=" * 60)
    print("VACP Reproduction")
    print("=" * 60)

    set_seed(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. 加载模型
    print("\n1. Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map="auto" if device == 'cuda' else None
    )
    model.to(device)
    model.eval()

    # 2. 加载数据
    print("\n2. Loading data...")
    all_data = load_squad_data(tokenizer, num_samples=5000)
    calib_data, val_data, test_data = split_data(
        all_data, config.calib_ratio, config.test_ratio, config.val_size
    )

    # 3. 构建词汇表掩码
    print("\n3. Building vocabulary mask...")
    val_texts = [d[0] for d in val_data]
    vocab_mask = build_vocabulary_mask(
        model, tokenizer, val_texts, config.prob_threshold, config.batch_size
    )
    vocab_mask = vocab_mask.to(device)

    # 添加 p=1 验证
    val_targets = [str(d[1]) for d in val_data]  # 提取真实 token
    validate_mask_coverage(vocab_mask, tokenizer, val_texts, val_targets)

    #调试
    print(f"\n[DEBUG] Vocab mask info:")
    print(f"  Mask shape: {vocab_mask.shape}")
    print(f"  Mask sum (有效token数): {vocab_mask.sum().item()}")
    print(f"  Mask device: {vocab_mask.device}")

    # 4. 温度调优
    print("\n4. Temperature tuning...")
    # best_tau=0.1
    best_tau, _ = tune_temperature(
        model, tokenizer, calib_data[:1000], test_data[:500],
        vocab_mask, config.alpha, config.temperature_candidates,
        config.batch_size, device,
        model_vocab_size=model.config.vocab_size
    )

    # 5. VACP evaluation
    print("\n5. VACP evaluation...")

    # 使用温度调优找到的最佳温度
    print(f"[INFO] Using temperature from tuning: τ = {best_tau}")

    scorer = VACPScorer(vocab_mask, temperature=best_tau, model_vocab_size=model.config.vocab_size)
    q_hat = calibrate_threshold(model, tokenizer, calib_data, scorer, config.alpha, config.batch_size, device)
    coverage, mean_size, sizes = evaluate_coverage(model, tokenizer, test_data, scorer, q_hat, config.batch_size,
                                                   device)

    # 保存结果
    os.makedirs("results", exist_ok=True)
    with open("results/vacp_results.json", "w") as f:
        json.dump({
            "coverage": coverage,
            "mean_set_size": mean_size,
            "median_set_size": np.median(sizes),
            "threshold": q_hat,
            "temperature": best_tau,
            "vocab_size": vocab_mask.sum().item()
        }, f, indent=2)

    print("\nResults saved to results/vacp_results.json")


if __name__ == "__main__":
    main()