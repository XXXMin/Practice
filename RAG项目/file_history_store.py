#将用户聊天记录以JSON格式存下来（前面的长期会话记忆，类似）
import json
import os
from typing import Sequence
import config_data as config
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

def get_history(session_id):
    return FileChatMessageHistory(session_id,config.history_path)

class FileChatMessageHistory(BaseChatMessageHistory):#替换掉InMemoryChatMessageHistory实现文件存储会话记忆
    def __init__(self,session_id,storage_path):#创建类的实例时自动调用
        self.session_id=session_id#会话id
        self.storage_path=storage_path#不同会话id的存储文件，所在的文件夹路径
        #完整的文件路径
        self.file_path=os.path.join(self.storage_path,self.session_id)
        '''
        os.path.join():无论在什么系统，都能得到正确的路径
        path = os.path.join("folder", "subfolder", "file.txt")
        Windows: folder\subfolder\file.txt
        Linux/Mac: folder/subfolder/file.txt
        '''
        #确保文件夹存在(如果文件夹不存在则会创建)
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)
        '''
        os.makedirs:一次性创建多级不存在的目录
        os.path.dirname:获取路径的目录部分：返回文件或文件夹所在的父目录路径
        '''

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:#->None是Python中的类型注解，表示这个函数没有返回值(或者说返回None)
        #Sequence序列，类似list，tuple
        all_messages=list(self.messages)#确保已有的消息是列表,messages是个序列、列表
        all_messages.extend(messages)#新消息和已有的融合成一个list
        #将数据同步写入本地文件中
        #类对象写入文件都是一堆二进制，为了方便看，将BaseMessage消息转为字典（借助json模块以json字符串写入文件）
        #message_to_dict（单->单）：单个消息对象（BaseMessage类实例->字典）
        '''
        new_messages=[]
        for message in all_messages:
            new_messages.append(message_to_dict(message))#存放都是字典
        '''
        new_messages=[message_to_dict(message) for message in all_messages]#等价上面写法（列表推导式）
        #将数据写入文件
        '''
        with...as...:自动管理资源（比如文件），用完后自动关闭，不用手动写 close()
        json.dump:将Python的数据（如字典，列表等）转化为JSON格式写入文件
        '''
        with open(self.file_path,'w',encoding='utf-8') as f:
            json.dump(new_messages,f)#把列表中所有消息转为json并写到文件中

    @property#装饰器@property将messages方法变成成员属性用
    def messages(self)->list[BaseMessage]:#返回值是list[BaseMessage]
        #当前文件内：list[字典]
        try:
            with open(self.file_path,'r',encoding='utf-8') as f:
                #json.load：从文件读取 JSON 数据并转换为 Python 对象（反序列化）
                messages_data=json.load(f)#返回值就是：list[字典]
                # messages_from_dict：[字典，字典...]->[消息，消息...]（BaseMessage类实例）
                return messages_from_dict(messages_data)

        except FileNotFoundError:
            return []

    def clear(self)->None:
        with open(self.file_path,'w',encoding='utf-8') as f:
            json.dump([],f)#写空列表相当于清空文件
