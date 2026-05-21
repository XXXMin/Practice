# vacp/temperature_tuning.py
import torch
import numpy as np
from typing import List, Tuple, Dict
from .scorer import VACPScorer
from .calibrator import calibrate_threshold, evaluate_coverage


def tune_temperature(
        model, tokenizer,
        calib_data: List[Tuple[str, int]],
        val_data: List[Tuple[str, int]],
        vocab_mask: torch.Tensor,
        alpha: float = 0.1,
        temperature_candidates: List[float] = None,
        batch_size: int = 8,
        device: str = 'cuda',
        model_vocab_size: int = None
) -> Tuple[float, Dict]:
    """温度调优"""
    if temperature_candidates is None:
        # temperature_candidates = [0.05, 0.1, 0.2, 0.5, 1.0]
        temperature_candidates = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    results = {}

    for tau in temperature_candidates:
        scorer = VACPScorer(vocab_mask, temperature=tau, model_vocab_size=model_vocab_size)
        q_hat = calibrate_threshold(model, tokenizer, calib_data, scorer, alpha, batch_size, device)
        coverage, mean_size, _ = evaluate_coverage(model, tokenizer, val_data, scorer, q_hat, batch_size, device)
        results[tau] = {'threshold': q_hat, 'coverage': coverage, 'mean_set_size': mean_size}
        print(f"τ={tau}: coverage={coverage * 100:.1f}%, mean|C|={mean_size:.1f}")

    # 选择覆盖率 ≥ 90% 中，集合最小的温度
    valid_taus = [tau for tau, res in results.items() if res['coverage'] >= (1 - alpha)]
    if valid_taus:
        best_tau = min(valid_taus, key=lambda tau: results[tau]['mean_set_size'])
    else:
        # 如果没有达标的，选最接近的
        best_tau = min(results.keys(), key=lambda tau: abs(results[tau]['coverage'] - (1 - alpha)))
    print(f"\nBest temperature: τ={best_tau}")
    return best_tau, results