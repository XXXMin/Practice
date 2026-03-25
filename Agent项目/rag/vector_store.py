from langchain_core.documents import Document
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader,txt_loader,listdir_with_allowed_type,get_file_md5_hex
from utils.logger_handler import logger

class VectorStoreService(object):
    def __init__(self):
        '''
        :param embedding:表示嵌入模型的传入
        '''
        self.vector_store=Chroma(
            collection_name=chroma_conf["collection_name"],#数据库表名
            embedding_function=embed_model,#默认model是text-embedding-v1
            persist_directory=chroma_conf["persist_directory"],#Chroma数据库的本地存储路径(文件夹)
        )#向量存储实例
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        '''
        返回向量检索器
        检索器作用：方便加入链
        '''
        #as_retriever：将向量数据库(VectorStore)转换为检索器(Retriever)的方法
        #检索器(Retriever) 是一个专门负责从文档集合中查找相关信息的组件,它是 RAG(检索增强生成)架构中的核心部分
        return self.vector_store.as_retriever(search_kwargs={'k':chroma_conf["k"]})#数字表示每次检索需要返回几个匹配结果

    def load_document(self):
        '''
        从数据文件夹内读取数据文件，转为向量存入
        要计算文件的md5去重（已加载文件不作操作）
        :return:None
        '''

        def check_md5_hex(md5_for_check:str):
            '''
            检查传入的md5字符串是否已经被处理过了
            本地留个md5.txt文件，到时候读文件对比
            return False(md5未处理过)/True(md5已处理过，已有记录)
            '''
            if not os.path.exists(get_abs_path(chroma_conf['md5_hex_store'])):  # 用于检查文件或目录是否存在,存在→True;不存在→False
                # 文件不存在会进来,md5肯定没处理过
                open(get_abs_path(chroma_conf['md5_hex_store']), "w", encoding="utf-8").close()
                return False

            for line in open(get_abs_path(chroma_conf['md5_hex_store']), "r", encoding="utf-8").readlines():
                line = line.strip()  # 处理字符串前后的空格和回车（等于前后的空格和回车都去掉，只保留核心内容，比较更精准）
                if line == md5_for_check:
                    return True  # 文件里读到了一模一样的md5，已处理过
            return False  # for循环跑完了还没找到

        def save_md5_hex(md5_str: str):
            '''
            将传入的md5字符串记录到文件内并保存(追加"a")
            '''
            with open(get_abs_path(chroma_conf['md5_hex_store']), "a", encoding="utf-8") as f:
                f.write(md5_str + '\n')

        def get_file_documents(read_path:str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []#表示啥也没读到

        allowed_files_path:list[str]=listdir_with_allowed_type(get_abs_path(chroma_conf["data_path"]),tuple(chroma_conf["allow_knowledge_file_type"]))

        for path in allowed_files_path:
            md5_hex=get_file_md5_hex(path)#获取文件md5值
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在于知识库中，跳过")#正常的日志信息
                continue

            try:
                documents:list[Document]=get_file_documents(path)
                if not documents:
                    #啥也没读到
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document:list[Document]=self.spliter.split_documents(documents)

                if not split_document:
                    #分片后没有有效文本内容，跳过
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                #将内容存入向量库中
                self.vector_store.add_documents(split_document)

                #记录已经处理好的md5值，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")

            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)#exc_info=True:会记录详细的报错堆栈，False仅记录报错本身
                continue

if __name__ == '__main__':
    vs=VectorStoreService()
    vs.load_document()
    retriever=vs.get_retriever()
    res=retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)