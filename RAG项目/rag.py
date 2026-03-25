'''
在线检索相关（核心就是RagService这个类）
最为核心的代码
内置四个私有成员变量，以及一个私有成员方法
1.向量服务：获得检索器
2.提示词模板
3.聊天模型
4.链（上下组装在一起都为它服务self.chain）
成员方法：获得链
'''
from jinja2.runtime import new_context
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from file_history_store import get_history

def print_prompt(prompt:PromptValue):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagService(object):
    def __init__(self ):
        self.vector_service=VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name),
        )#向量服务:用来做检索的
        self.prompt_template=ChatPromptTemplate.from_messages(
            [
                ("system","以我提供的已知参考资料为主，简洁并专业的回答用户的问题。参考资料:{context}。"),
                ("system", "并且我提供用户对话的历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user","请回答用户提问：{input}"),
            ]
        )#提示词模板
        self.chat_model=ChatTongyi(model=config.chat_model_name)#模型

        self.chain=self.__get_chain()#链

    def __get_chain(self):
        '''
        获取最终的执行链
        '''
        retriever=self.vector_service.get_retriever()

        def format_document(docs:list[Document]):
            if not docs:
                #没有
                return "无相关参考资料"

            #有参考资料的话：
            formatted_str=""
            for doc in docs:
                formatted_str+=f"文档片段：{doc.page_content}\n文档源数据：{doc.metadata}\n\n"

            return formatted_str

        #测试debug
        # def tem(value):
        #     print("="*20)
        #     print(value)
        #     return value

        def format_for_retriever(dict):
            return dict["input"]

        def format_for_template(value):
            new_value={}
            new_value["input"]=value["input"]["input"]
            new_value["context"]=value["context"]
            new_value["history"]=value["input"]["history"]
            return new_value

        chain=(
            {
                "input":RunnablePassthrough(),#保留原始输入
                "context":RunnableLambda(format_for_retriever)|retriever|format_document
            } | RunnableLambda(format_for_template)|self.prompt_template |print_prompt|self.chat_model|StrOutputParser()
        )

        conversation_chain=RunnableWithMessageHistory(
            chain,#原有链
            get_history,
            input_messages_key="input",
            history_messages_key="history",#历史消息的一个占位
        )#增强链的invoke要的是字典而非字符串

        return conversation_chain#有历史增强功能

if __name__=="__main__":
    #session_id配置(字典)
    session_config={
        "configurable":{
            "session_id":"user_001",
        },
    }

    print(RagService().chain.invoke({"input":"春天穿什么颜色的衣服"},session_config))