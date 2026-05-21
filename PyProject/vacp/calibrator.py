# vacp/calibrator.py
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from .scorer import VACPScorer


def calibrate_threshold(
        model, tokenizer, calib_data: List[Tuple[str, int]],
        scorer: VACPScorer, alpha: float = 0.1,
        batch_size: int = 8, device: str = 'cuda'
) -> float:
    """校准阈值"""
    model.eval()
    scores = []

    with torch.no_grad():
        for i in tqdm(range(0, len(calib_data), batch_size), desc="Calibrating"):
            batch = calib_data[i:i + batch_size]
            prompts = [b[0] for b in batch]
            targets = [b[1] for b in batch]

            inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]

            for j, tid in enumerate(targets):
                scores.append(scorer.compute_score(logits[j], tid))

    n = len(scores)
    print(f"\n[DEBUG] Calibration scores stats:")
    print(f"  min score: {min(scores):.6f}")
    print(f"  max score: {max(scores):.6f}")
    print(f"  mean score: {np.mean(scores):.6f}")
    print(f"  unique scores: {len(set(scores))}")

    q_hat = np.quantile(scores, (1 - alpha) * (1 + 1 / n))
    print(f"  quantile level: {(1 - alpha) * (1 + 1 / n):.6f}")
    print(f"  q_hat: {q_hat:.6f}")

    return q_hat


def evaluate_coverage(
        model, tokenizer, test_data: List[Tuple[str, int]],
        scorer: VACPScorer, threshold: float,
        batch_size: int = 8, device: str = 'cuda'
) -> Tuple[float, float, List[int]]:
    """评估覆盖率和集合大小"""
    model.eval()
    covered = 0
    set_sizes = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
            batch = test_data[i:i + batch_size]
            prompts = [b[0] for b in batch]
            targets = [b[1] for b in batch]

            inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]

            for j, tid in enumerate(targets):
                pred_set = scorer.get_prediction_set(logits[j], threshold)
                set_sizes.append(len(pred_set))
                if tid in pred_set:
                    covered += 1

    coverage = covered / len(test_data)
    mean_size = np.mean(set_sizes)
    print(f"Coverage: {coverage * 100:.1f}%, Mean |C|: {mean_size:.1f}")
    return coverage, mean_size, set_sizes