# data/squad_loader.py
import random
from typing import List, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer


def load_squad_data(
        tokenizer: AutoTokenizer,
        num_samples: int = 5000,
        seed: int = 42
) -> List[Tuple[str, int]]:
    """
    加载SQuAD数据集，格式化为 (prompt, target_token_id)
    论文格式: "Context: {context} Question: {question} Answer:"
    目标: 答案的第一个token
    """
    random.seed(seed)

    print("Loading SQuAD dataset...")
    squad = load_dataset("squad", split="train")

    indices = random.sample(range(len(squad)), min(num_samples, len(squad)))

    data = []
    skipped = 0
    for idx in indices:
        example = squad[idx]
        context = example['context']
        question = example['question']
        answer_text = example['answers']['text'][0]

        prompt = f"Context: {context} Question: {question} Answer:"

        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        if answer_tokens:
            target_id = answer_tokens[0]
            data.append((prompt, target_id))
        else:
            skipped += 1

    print(f"Loaded {len(data)} samples (skipped {skipped} empty answers)")
    return data


def split_data(
        data: List,
        calib_ratio: float = 0.6,
        test_ratio: float = 0.4,
        val_size: int = 1000,
        seed: int = 42
) -> Tuple[List, List, List]:
    """划分数据集: 校准集、验证集(构建V*)、测试集"""
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    calib_size = int(n * calib_ratio)

    # 验证集用于构建V*
    remaining = data[calib_size:]
    val_data = random.sample(remaining, min(val_size, len(remaining)))

    # 测试集
    test_data = [d for d in remaining if d not in val_data]
    calib_data = data[:calib_size]

    print(f"\nData split:")
    print(f"  Calibration: {len(calib_data)} samples")
    print(f"  Validation (for V*): {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    return calib_data, val_data, test_data