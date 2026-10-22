import torch
import sys
from pathlib import Path

def get_best_device():
    """自动选择最佳设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

#添加项目根路径到路径，以便导入vacp模块
sys.path.insert(0,str(Path(__file__).parent.parent))

from vacp.scorer import VACPScorer

class VACPAdapter:
    '''
    VACP适配器：将论文复现的那个VACPScorer封装成Agent可用的接口
    '''
    def __init__(
            self,
            model=None,
            tokenizer=None,
            vocab_mask=None,
            temperature:float=1.0,
            device=get_best_device(),
    ):
        self.model=model
        self.tokenizer=tokenizer
        self.device=device
        self.temperature=temperature
        #判断是否有完整的VACP功能
        if vocab_mask is not None and model is not None:
            self.scorer=VACPScorer(
                vocab_mask=vocab_mask,
                temperature=temperature,
                model_vocab_size=model.config.vocab_size if model else None
            )
            self.has_full_scorer=True
            print(f"VACP完整模式已启用（temperature={temperature},device={device}）")
        else:
            self.has_full_scorer=False
            print(f"VACP降级为启发式模式（temperature={temperature}）")

    def compute_confidence_from_logits(self, logits, target_token_id=None):
        if not self.has_full_scorer:
            return self._heuristic_confidence(None)

        if isinstance(logits, list):
            logits = torch.tensor(logits)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        try:
            # 1. 应用词汇表掩码
            masked_logits = logits[0].clone()
            masked_logits[~self.scorer.vocab_mask] = float('-inf')

            # 2. 温度缩放
            scaled_logits = torch.clamp(masked_logits / self.temperature, min=-50, max=50)

            # 3. 计算概率分布
            probs = torch.softmax(scaled_logits, dim=-1)

            # 4. 取 Top-10 概率之和
            top_k = 10
            top_probs, _ = torch.topk(probs, top_k)
            top_k_mass = top_probs.sum().item()

            # 5. 归一化到 0-1
            confidence = min(1.0, max(0.0, top_k_mass))

        except Exception as e:
            print(f"VACP计算失败：{e}")
            confidence = 0.5

        return {
            "confidence": round(confidence, 3),
            "prediction_set_size": None,
            "method": "top_k_mass"
        }

    def compute_confidence_from_response(
            self,
            response_text:str,
            context:str=None
    )->dict:
        '''基于回复文本计算置信度（降级方案）'''
        return self._heuristic_confidence(response_text)

    def _heuristic_confidence(self,response_text:str)->dict:
        '''
        启发式置信度计算（降级方案）（启发式规则）
        启发式指的是一种基于经验和直觉的、非严格的、旨在快速找到“足够好”解决方案的方法
        '''
        if not response_text:
            confidence=0.3
        elif any(word in response_text for word in ["可能","也许","不太确定","好像"]):
            confidence=0.3
        elif len(response_text)<20:
            confidence=0.5
        else:
            confidence=0.7
        return {
            "confidence":confidence,
            "prediction_set_size":None,
            "method":"heuristic"
        }

    def evaluate_response(self,response_text:str,logits:torch.Tensor=None)->dict:
        '''评估回复质量（主接口）:有logits就用精确计算，否则就用启发式'''
        if logits is not None and self.has_full_scorer:
            return self.compute_confidence_from_logits(logits)
        else:
            return self.compute_confidence_from_response(response_text)

    def needs_human_review(
            self,
            response_text:str=None,
            logits:list=None,
            confidence_threshold:float=0.7
    )->tuple:
        '''判断是否需要人工确认(VACP or 启发式)'''
        #优先使用VACP（如果有logits且完整模式可用）
        if logits is not None and self.has_full_scorer:
            # 如果是 Tensor，直接使用；如果是 list，转换为 Tensor
            if isinstance(logits,list):
                logits_tensor=torch.tensor(logits)
            else:
                logits_tensor=logits
            result=self.compute_confidence_from_logits(logits_tensor)
            confidence=result["confidence"]
            method="VACP"
        elif response_text is not None:
            result=self._heuristic_confidence(response_text)#使用启发式
            confidence=result["confidence"]
            method="heuristic"
        else:
            return True,0.3,"无有效输入，需要人工确认"
        needs_review=confidence<confidence_threshold
        #优化reason信息
        if needs_review:
            if method == "VACP":
                reason=f"VACP置信度({confidence:.2f})低于阈值({confidence_threshold})"
            else:
                reason =f"启发式置信度({confidence:.2f})低于阈值({confidence_threshold})"
        else:
            reason=""
        return needs_review, confidence,reason

#全局实例
vacp_adapter=None

def init_vacp_adapter(
        model=None,
        tokenizer=None,
        vocab_mask=None,
        temperature:float=1.0,
        device=get_best_device()
):
    #初始化全局VACP适配器(创建一次全局实例)
    global vacp_adapter
    vacp_adapter=VACPAdapter(
        model=model,
        tokenizer=tokenizer,
        vocab_mask=vocab_mask,
        temperature=temperature,
        device=device
    )
    return vacp_adapter

def get_vacp_adapter():
    #获取全局VACP适配器（在其他地方获取这个实例）
    return vacp_adapter