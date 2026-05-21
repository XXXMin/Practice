# vacp/vocab_mask.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def build_vocabulary_mask(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        validation_texts: list,
        prob_threshold: float = 1e-5,
        batch_size: int = 8
) -> torch.Tensor:
    """构建有效词汇表掩码V*"""
    # 关键：使用模型的实际词汇表大小，而不是tokenizer的
    vocab_size = model.config.vocab_size
    tokenizer_vocab_size = len(tokenizer)

    print(f"Model vocab size: {vocab_size}")
    print(f"Tokenizer vocab size: {tokenizer_vocab_size}")

    if vocab_size != tokenizer_vocab_size:
        print(f"Warning: vocab size mismatch! Using model vocab size: {vocab_size}")

    mask = torch.ones(vocab_size, dtype=torch.bool)

    # Step 1: 结构过滤（只处理tokenizer中存在的token）
    special_tokens = ['<pad>', '<eos>', '<bos>', '<unk>', '<s>', '</s>']
    for token in special_tokens:
        if token in tokenizer.vocab:
            token_id = tokenizer.vocab[token]
            if token_id < vocab_size:
                mask[token_id] = False
                print(f"  Removed special token: {token} (id={token_id})")

    # 移除保留token
    removed_count = 0
    for token, idx in list(tokenizer.vocab.items()):
        if idx >= vocab_size:
            continue
        if 'unused' in token.lower() or 'reserved' in token.lower():
            if mask[idx]:
                mask[idx] = False
                removed_count += 1
    if removed_count > 0:
        print(f"  Removed {removed_count} reserved/unused tokens")

    # Step 2: 经验过滤
    print(f"  Computing empirical probabilities on {len(validation_texts)} samples...")

    device = next(model.parameters()).device
    max_probs = torch.zeros(vocab_size)
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(validation_texts), batch_size), desc="Building V*"):
            batch = validation_texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]

            # 确保logits大小与mask一致
            if logits.size(-1) > vocab_size:
                logits = logits[:, :vocab_size]
            elif logits.size(-1) < vocab_size:
                pad = torch.zeros(logits.size(0), vocab_size - logits.size(-1), device=device)
                logits = torch.cat([logits, pad], dim=-1)

            probs = torch.softmax(logits, dim=-1)

            batch_max, _ = probs.max(dim=0)
            max_probs = torch.maximum(max_probs, batch_max.cpu())

    # 移除从未超过阈值的token
    low_prob_mask = max_probs > prob_threshold
    mask = mask & low_prob_mask

    removed_low_prob = (~low_prob_mask).sum().item()
    if removed_low_prob > 0:
        print(f"  Removed {removed_low_prob} tokens with max prob <= {prob_threshold}")

    final_size = mask.sum().item()
    print(f"\nFinal effective vocabulary size: {final_size} / {vocab_size} "
          f"({100 * final_size / vocab_size:.1f}% retained)")

    return mask


def validate_mask_coverage(
        mask: torch.Tensor,
        tokenizer: AutoTokenizer,
        validation_texts: list,
        ground_truth_tokens: list
) -> float:
    """验证ground-truth token是否在V*中"""
    vocab_size = mask.size(0)
    covered = 0
    total_valid = 0

    for text, gt in zip(validation_texts, ground_truth_tokens):
        if isinstance(gt, int):
            token_id = gt
        else:
            try:
                token_ids = tokenizer.encode(gt, add_special_tokens=False)
                if not token_ids:
                    continue
                token_id = token_ids[0]
            except:
                continue

        if token_id >= vocab_size:
            continue

        total_valid += 1
        if mask[token_id]:
            covered += 1

    if total_valid == 0:
        print("Warning: No valid ground-truth tokens found for coverage validation")
        return 0.0

    ratio = covered / total_valid
    print(f"Ground-truth coverage in V*: {100 * ratio:.1f}% ({covered}/{total_valid})")
    return ratio