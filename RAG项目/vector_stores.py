#获得一个可以加入链条的检索器，同时在过程中加载了向量库
#核心：返回一个向量检索器，便于后续构造链条
from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    def __init__(self,embedding):
        '''
        :param embedding:表示嵌入模型的传入
        '''
        self.embedding = embedding
        self.vector_store=Chroma(
            collection_name=config.collection_name,#数据库表名
            embedding_function=self.embedding,#默认model是text-embedding-v1
            persist_directory=config.persist_directory,#Chroma数据库的本地存储路径(文件夹)
        )

    def get_retriever(self):
        '''
        返回向量检索器
        检索器作用：方便加入链
        '''
        #as_retriever：将向量数据库(VectorStore)转换为检索器(Retriever)的方法
        #检索器(Retriever) 是一个专门负责从文档集合中查找相关信息的组件,它是 RAG(检索增强生成)架构中的核心部分
        return self.vector_store.as_retriever(search_kwargs={'k':config.similarity_threshold})#数字表示每次检索需要返回几个匹配结果


if __name__=="__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_core.documents import Document

    retriever=VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()#Embedding(嵌入) 是将文本、图像等非结构化数据转换为固定长度的数值向量的技术,让计算机能够"理解"和"计算"数据的语义相似度。

    res:list[Document]=retriever.invoke("我的体重180斤，尺码推荐")
    print(res)