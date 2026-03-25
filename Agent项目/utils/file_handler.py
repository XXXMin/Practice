#文件处理的相关内容
import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader


def get_file_md5_hex(filepath:str):#获取文件的md5值
    if not os.path.exists(filepath):#判断文件是否存在
        logger.error(f"[md5计算]文件{filepath}不存在")
        return

    if not os.path.isfile(filepath):#判断文件是否是文件夹
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return

    md5_obj=hashlib.md5()#创建一个 MD5 哈希对象，用于后续逐步计算数据的 MD5 值
    chunk_size=4096#分片大小：4KB，避免文件过大爆内存
    try:
        with open(filepath,"rb") as f:#rb:二进制读取
            while chunk:=f.read(chunk_size):#:=（海象运算法）：同时完成赋值和返回赋值结果两项工作
                md5_obj.update(chunk)#读取chunk_size大小的文件赋值给chunk，当chunk没有了，文件空了，返回False停止
            '''
            等价于：
            chunk=f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk=f.read(chunk_size)
            '''
            md5_hex=md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        '''
        BaseException
        ├── SystemExit
        ├── KeyboardInterrupt
        ├── GeneratorExit
        └── Exception  # ← 所有常规异常的基类
            ├── TypeError
            ├── ValueError
            ├── KeyError
            ├── FileNotFoundError
            └── ... (其他具体异常)
        '''
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None#return 和 return None 是等价的

def listdir_with_allowed_type(path:str,allowed_types:tuple[str]):#获取允许的文件列表（判断哪些文件被允许，写在列表内返回）
    files=[]
    if not os.path.isdir(path):#如果不是文件夹
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types
    for f in os.listdir(path):#python内置，列出整个文件夹的东西
        if f.endswith(allowed_types):#如果是允许的文件类型结尾（.endswith:用于判断字符串是否以指定的后缀结尾）
            files.append(os.path.join(path,f))#文件路径和文件名组合加到列表

    return tuple(files)#转成元组不允许修改

def pdf_loader(filepath:str,passwd=None)->list[Document]:#加载PDF文档
    return PyPDFLoader(filepath,passwd).load()#全量加载

def txt_loader(filepath:str)->list[Document]:#加载TXT文档
    return TextLoader(filepath,encoding="utf-8").load()