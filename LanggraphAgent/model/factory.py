#模型工厂代码（提供模型）
from abc import ABC, abstractmethod
from typing import Optional,Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.config_handler import rag_conf
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
import torch

import os
from langchain_openai import ChatOpenAI,OpenAIEmbeddings


class BaseModelFactory(ABC):#ABC:python内置抽象类
    @abstractmethod
    def generator(self)->Optional[Embeddings|BaseChatModel]:#生成想要的对象
        pass#抽象体只定义函数名，不写内容

class ChatModelFactory(BaseModelFactory):
    def generator(self):
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7,
        )

class EmbeddingsFactory(BaseModelFactory):
    def generator(self):
        return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],  # 读 rag.yml 里的新模型名
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            chunk_size=10,  # 🆕 关键：让 LangChain 每次只发 10 条给 API
            check_embedding_ctx_length=False  # 防止因token长度检查报错
        )

class LocalQwenFactory(BaseModelFactory):
    '''本地qwen模型工厂'''
    def __init__(self,model_name:str=None,device:str=None):
        #如果没有传入model_name，从缓存查找
        if model_name is None:
            from pathlib import Path
            cache_base=Path.home()/".cache"/ "huggingface" / "hub"
            model_cache=cache_base/"models--Qwen--Qwen2-1.5B-Instruct"
            snapshots_dir = model_cache / "snapshots"
            if snapshots_dir.exists():
                snapshot_dirs=list(snapshots_dir.iterdir())#iterdir()：返回目录下所有文件和子目录的迭代器
                if snapshot_dirs:
                    model_name=str(snapshot_dirs[0])
                    print(f"自动找到缓存模型：{model_name}")
                else:
                    raise FileNotFoundError(f"snapshots 目录为空: {snapshots_dir}")
            else:
                raise FileNotFoundError(f"缓存目录不存在: {snapshots_dir}")
        #保存实例变量
        self.model_name=model_name#地址
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")

    def generator(self):
        '''返回本地模型和tokenizer'''
        import os
        # 强制离线模式
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(f"正在加载本地模型：{self.model_name}")
        print(f"设备：{self.device}")
        tokenizer=AutoTokenizer.from_pretrained(
            self.model_name,#路径或仓库名
            trust_remote_code=True,#信任远程代码（qwen需要）
            local_files_only=True#只使用本地文件
        )
        #设置填充令牌
        if tokenizer.pad_token is None:
            tokenizer.pad_token=tokenizer.eos_token
        model=AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device=="cuda" else torch.float32,
            device_map="auto" if self.device=="cuda" else None,
            trust_remote_code=True,
            local_files_only=True
        )
        if self.device!="cuda":
            model.to(self.device)
        model.eval()#禁用Dropout、BatchNorm 等训练专用层，返回模型和分词器的元组
        return model,tokenizer

chat_model=ChatModelFactory().generator()
embed_model=EmbeddingsFactory().generator()
#本地模型
_local_model=None
_local_tokenizer=None

def get_local_model(device:str=None):
    '''获取本地模型（懒加载）'''
    global _local_model,_local_tokenizer
    if _local_model is None:#第一次调用
        factory=LocalQwenFactory(device=device)
        _local_model,_local_tokenizer=factory.generator()#加载模型
    return _local_model,_local_tokenizer#返回缓存实例