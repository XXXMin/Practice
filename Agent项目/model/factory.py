#模型工厂代码（提供模型）
from abc import ABC, abstractmethod
from typing import Optional
from utils.config_handler import rag_conf
from langchain_community.chat_models import ChatTongyi
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings


class BaseModelFactory(ABC):#ABC:python内置抽象类
    @abstractmethod
    def generator(self)->Optional[Embeddings|BaseChatModel]:#生成想要的对象
        pass#抽象体只定义函数名，不写内容

class ChatModelFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings|BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])

class EmbeddingsFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings|BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])

chat_model=ChatModelFactory().generator()
embed_model=EmbeddingsFactory().generator()