# vacp/scorer.py
import torch
import numpy as np


class VACPScorer:
    """论文Appendix A的VACPScorer实现"""

    def __init__(self, vocab_mask: torch.Tensor, temperature: float = 0.1, model_vocab_size: int = None):
        """
        Args:
            vocab_mask: 布尔张量，True表示保留该token
            temperature: 温度参数
            model_vocab_size: 模型的实际词汇表大小（用于扩展掩码）
        """
        #调试
        print(f"[DEBUG] VACPScorer __init__: temperature = {temperature}")

        self.temperature = temperature
        self.device = vocab_mask.device if isinstance(vocab_mask, torch.Tensor) else 'cpu'
        self.model_vocab_size = model_vocab_size

        # 如果掩码大小小于模型词汇表大小，需要扩展
        if model_vocab_size is not None and vocab_mask.size(0) < model_vocab_size:
            # 扩展掩码到模型词汇表大小，新增的部分默认False（不包含）
            expanded_mask = torch.zeros(model_vocab_size, dtype=torch.bool, device=self.device)
            expanded_mask[:vocab_mask.size(0)] = vocab_mask
            self.vocab_mask = expanded_mask
            print(f"  Extended vocab mask from {vocab_mask.size(0)} to {model_vocab_size}")
        else:
            self.vocab_mask = vocab_mask.to(self.device)

        self._debug_printed = False
        self._temp_printed = False

    def compute_score(self, logits: torch.Tensor, target_id: int) -> float:
        #调试
        if not self._temp_printed:
            print(f"[DEBUG] compute_score: self.temperature = {self.temperature}")
            self._temp_printed = True

        # 确保logits和掩码大小一致
        if logits.size(-1) != self.vocab_mask.size(0):
            if logits.size(-1) > self.vocab_mask.size(0):
                logits = logits[:, :self.vocab_mask.size(0)] if logits.dim() > 1 else logits[:self.vocab_mask.size(0)]
            else:
                new_mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=self.device)
                new_mask[:self.vocab_mask.size(0)] = self.vocab_mask
                self.vocab_mask = new_mask

        masked_logits = logits.clone()
        masked_logits[~self.vocab_mask] = float('-inf')

        # 温度缩放 + 裁剪防止溢出
        logits_for_softmax = masked_logits / self.temperature
        logits_for_softmax = torch.clamp(logits_for_softmax, min=-50, max=50)  # 裁剪
        probs = torch.softmax(logits_for_softmax, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        #调试
        # 在 compute_score 中，找到 rank 之前添加
        if not self._debug_printed:
            # 检查 target_id 是否在掩码中
            is_masked = self.vocab_mask[target_id] if target_id < len(self.vocab_mask) else False
            print(f"  target_id {target_id} in vocab_mask? {is_masked}")

            # 检查原始logits中target位置的logit值
            raw_logit = logits[target_id].item() if target_id < len(logits) else float('-inf')
            print(f"  raw logit for target: {raw_logit:.2f}")

            masked_logit = masked_logits[target_id].item() if target_id < len(masked_logits) else float('-inf')
            print(f"  masked logit for target: {masked_logit:.2f}")

        # 找到target的排名
        rank_mask = (sorted_indices == target_id)
        rank = rank_mask.nonzero()

        if len(rank) == 0:
            if not self._debug_printed:
                print(f"[DEBUG] target_id {target_id} not in masked vocabulary!")
                self._debug_printed = True
            return 1.0

        rank = rank[0, 0].item()
        cumsum = torch.cumsum(sorted_probs, dim=0)

        U = np.random.uniform(0, 1)

        if rank == 0:
            score = U * sorted_probs[0].item()
        else:
            score = cumsum[rank - 1].item() + U * sorted_probs[rank].item()

        # 调试打印（仅第一次）
        if not self._debug_printed:
            print(f"\n[DEBUG] First sample details:")
            print(f"  target_id: {target_id}")
            print(f"  rank: {rank}")
            print(f"  sorted_probs[0]: {sorted_probs[0].item():.6f}")
            print(f"  sorted_probs[rank]: {sorted_probs[rank].item():.6f}")
            print(f"  cumsum[rank-1]: {cumsum[rank - 1].item() if rank > 0 else 'N/A'}")
            print(f"  U: {U:.6f}")
            print(f"  score: {score:.6f}")
            print(f"  sorted_probs sum: {sorted_probs.sum().item():.6f}")
            print(
                f"  masked_logits min/mean/max: {masked_logits.min().item():.2f} / {masked_logits.mean().item():.2f} / {masked_logits.max().item():.2f}")
            self._debug_printed = True

        # 确保得分在 [0, 1] 范围内
        score = min(max(score, 0.0), 1.0)

        return score

    def get_prediction_set(self, logits: torch.Tensor, threshold: float) -> list:
        """根据阈值生成预测集"""
        #调试
        if not hasattr(self, '_debug_printed_set'):
            self._debug_printed_set = False

        # 确保logits和掩码大小一致
        if logits.size(-1) != self.vocab_mask.size(0):
            if logits.size(-1) > self.vocab_mask.size(0):
                logits = logits[:, :self.vocab_mask.size(0)] if logits.dim() > 1 else logits[:self.vocab_mask.size(0)]
            else:
                new_mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=self.device)
                new_mask[:self.vocab_mask.size(0)] = self.vocab_mask
                self.vocab_mask = new_mask

        masked_logits = logits.clone()
        masked_logits[~self.vocab_mask] = float('-inf')

        logits_for_softmax = masked_logits / self.temperature
        logits_for_softmax = torch.clamp(logits_for_softmax, min=-50, max=50)
        probs = torch.softmax(logits_for_softmax, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        cumsum = torch.cumsum(sorted_probs, dim=0)

        # 修复：找到第一个累积概率超过阈值的位置
        # 注意：threshold 是 APS 得分的阈值，不是概率阈值
        # 我们需要包含所有累积概率 <= threshold 的 token
        exceed_mask = cumsum > threshold
        if not exceed_mask.any():
            # 所有累积概率都 <= threshold，返回所有 token
            # 但这种情况不应该发生，如果发生说明 threshold 太大了
            set_size = len(sorted_probs)
        else:
            # 找到第一个超过阈值的位置，该位置之前的 token 都需要包含
            set_size = exceed_mask.nonzero()[0, 0].item() + 1

        # 确保至少返回1个 token
        set_size = max(1, set_size)

        return sorted_indices[:set_size].tolist()