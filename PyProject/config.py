# config.py
from dataclasses import dataclass


@dataclass
class VACPConfig:
    # Model
    model_name: str = "Qwen/Qwen2-1.5B-Instruct"

    # Data splits (论文: 60%校准, 40%测试)
    calib_ratio: float = 0.6
    test_ratio: float = 0.4
    val_size: int = 1000  # 用于构建V*的验证集

    # Conformal prediction
    alpha: float = 0.1  # 错误率，目标覆盖率90%

    # Temperature
    temperature: float = 0.1  # 论文最优值
    temperature_candidates: list = None  # 用于调优

    # Vocabulary filtering
    prob_threshold: float = 1e-5  # 原来是 1e-3，降低阈值会保留更多 token

    # Generation
    max_new_tokens: int = 1  # 只预测下一个token
    batch_size: int = 8

    # Random seed
    seed: int = 42

    def __post_init__(self):
        if self.temperature_candidates is None:
            # self.temperature_candidates = [0.05, 0.1, 0.2, 0.5, 1.0]
            self.temperature_candidates = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


config = VACPConfig()