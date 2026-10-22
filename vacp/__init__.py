# vacp/__init__.py
from .vocab_mask import build_vocabulary_mask, validate_mask_coverage
from .scorer import VACPScorer
from .calibrator import calibrate_threshold, evaluate_coverage
from .temperature_tuning import tune_temperature  # 只导入 tune_temperature

__all__ = [
    'build_vocabulary_mask',
    'validate_mask_coverage',
    'VACPScorer',
    'calibrate_threshold',
    'evaluate_coverage',
    'tune_temperature'
]