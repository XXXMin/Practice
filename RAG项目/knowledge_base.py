'''
知识库
主要提供了一个类（知识库服务），接收网页上传的文件内容，并转为字符串存入向量库中
还有去重等功能：基于字符串的md5值完成，md5:只要字符串一样，md5结果也一样（计算上传文件的md5）
'''
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def check_md5(md5_str:str):
    '''
    检查传入的md5字符串是否已经被处理过了
    本地留个md5.txt文件，到时候读文件对比
    return False(md5未处理过)/True(md5已处理过，已有记录)
    '''
    if not os.path.exists(config.md5_path):#用于检查文件或目录是否存在,存在→True;不存在→False
        #文件不存在会进来,md5肯定没处理过
        open(config.md5_path,"w",encoding="utf-8").close()
        return False
    else:#表明文件已经存在
        '''
        .readlines():一次读取文件的所有行，返回一个列表，其中每个元素是文件中的一行字符串（包含换行符）
        .strip():移除字符串首尾的指定字符（默认为空白字符：空格、换行符\n、制表符\t等）
        '''
        for line in open(config.md5_path,"r",encoding="utf-8").readlines():
            line=line.strip()#处理字符串前后的空格和回车（等于前后的空格和回车都去掉，只保留核心内容，比较更精准）
            if line == md5_str:
                return True#文件里读到了一模一样的md5，已处理过
        return False#for循环跑完了还没找到

def save_md5(md5_str:str):
    '''
    将传入的md5字符串记录到文件内并保存(追加"a")
    '''
    with open(config.md5_path,"a",encoding="utf-8") as f:
        f.write(md5_str+'\n')

def get_string_md5(input_str:str,encoding="utf-8"):
    '''
    将传入的字符串转换为md5字符串
    '''
    #将字符串转换为bytes字节数组（相当于将字符串还原为二进制）
    #.encode()将字符串转换为字节串（bytes）
    str_bytes=input_str.encode(encoding=encoding)
    #创建md5对象
    md5_obj=hashlib.md5()#得到md5对象
    md5_obj.update(str_bytes)#更新内容（传入即将要转换的字节数组）
    return md5_obj.hexdigest()#得到md5的16进制字符串

class KnowledgeBaseService(object):
    #表示，如果文件夹不存在则创建，如果存在则跳过，就是啥也不干
    os.makedirs(config.persist_directory,exist_ok=True)#通过该函数确保文件夹存在（创建多级目录）

    def __init__(self):
        self.chroma=Chroma(
            collection_name=config.collection_name,#数据库表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),#默认model是text-embedding-v1(v4比较新，在阿里云平台上免费额度也比较充足)
            persist_directory=config.persist_directory,#Chroma数据库的本地存储路径(文件夹)
        )#记录向量存储的实例，chroma向量库对象

        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,#表示我们在分割的时候后，分割后的文本段最大长度
            chunk_overlap=config.chunk_overlap,#允许的连续文本段之间的字符重叠数量
            separators=config.separators,#自然段落划分的符号，方便文本分割器分割文本
            length_function=len,#默认使用python自带的len函数做长度统计的依据
        )#我们要用文本分割器的对象

    def upload_by_str(self, data:str,filename):
        '''
        将传入的字符串进行向量化，存入向量数据库中
        '''
        #先拿到传入字符串的md5值
        md5_hex=get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]内容已经存在在知识库中"
        #没进if，是新数据，需要处理了
        if len(data)>config.max_split_char_number:
            #文本分割
            knowledge_chunks = self.spliter.split_text(data)  # 返回过来的是list[str]
        else:
            #不分割
            knowledge_chunks=[data]

        metadata={
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),#加入向量库的时间点
            "operator":"小吴"
        }
        self.chroma.add_texts(
            knowledge_chunks,#传迭代器
            metadatas=[metadata for _ in knowledge_chunks]#数据的源数据,我们可自定义[{},{},...]
            #上面的，列表推导式：每一份数据共用一份源数据
        )

        #把已经处理过的数据的md5记录在册
        save_md5(md5_hex)

        return "[成功]内容已经成功载入向量库"

    #DEBUG版本：
    '''
    def upload_by_str(self, data: str, filename):
    print("=" * 50)
    print("[DEBUG] upload_by_str 被调用")
    
    # 先拿到传入字符串的md5值
    md5_hex = get_string_md5(data)
    print(f"[DEBUG] 计算的 MD5: {md5_hex}")
    
    # 打印实际使用的 md5 文件路径
    abs_md5_path = os.path.abspath(config.md5_path)
    print(f"[DEBUG] MD5 文件路径: {abs_md5_path}")
    print(f"[DEBUG] 文件是否存在: {os.path.exists(abs_md5_path)}")
    
    if os.path.exists(abs_md5_path):
        with open(abs_md5_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"[DEBUG] MD5 文件内容 ({len(lines)} 行):")
            for i, line in enumerate(lines[:10]):  # 只打印前10行
                print(f"  {i+1}: {line.strip()}")
    
    # 检查 MD5
    should_skip = check_md5(md5_hex)
    print(f"[DEBUG] check_md5 返回: {should_skip}")
    
    if should_skip:
        print("[DEBUG] 返回跳过信息")
        return "[跳过]内容已经存在在知识库中"
    
    print("[DEBUG] MD5 检查通过，开始处理新数据")
    
    # 没进if，是新数据，需要处理了
    if len(data) > config.max_split_char_number:
        print(f"[DEBUG] 文本长度 {len(data)} > {config.max_split_char_number}，进行分割")
        knowledge_chunks = self.spliter.split_text(data)
    else:
        print(f"[DEBUG] 文本长度 {len(data)} <= {config.max_split_char_number}，不分割")
        knowledge_chunks = [data]
    
    print(f"[DEBUG] 分割后得到 {len(knowledge_chunks)} 个片段")
    
    metadata = {
        "source": filename,
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operator": "小吴"
    }
    
    print("[DEBUG] 开始写入 Chroma 数据库...")
    try:
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks]
        )
        print("[DEBUG] 写入成功")
    except Exception as e:
        print(f"[DEBUG] 写入失败: {e}")
        return f"[错误] 写入数据库失败: {e}"
    
    # 把已经处理过的数据的md5记录在册
    save_md5(md5_hex)
    print(f"[DEBUG] MD5 {md5_hex} 已保存到文件")
    
    print("[DEBUG] 返回成功信息")
    return "[成功]内容已经成功载入向量库"
    '''

if __name__ == '__main__':
    # r1 = get_string_md5("周杰轮")
    # r2 = get_string_md5("周杰轮")
    # r3 = get_string_md5("周杰轮2")
    #
    # print(r1)
    # print(r2)
    # print(r3)

    # save_md5("71f03bf4dbad9398b6fab2a3dc261a4b")
    # print(check_md5("71f03bf4dbad9398b6fab2a3dc261a4b"))#周杰轮

    service=KnowledgeBaseService()
    r=service.upload_by_str("周杰轮","testfile")
    print(r)
