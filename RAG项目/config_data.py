
md5_path="D:/XIAO/Python_project/RAG项目/md5.text"

#Chroma
collection_name="RAG"
persist_directory="D:/XIAO/Python_project/RAG项目/chroma_db"

#spliter
chunk_size=1000
chunk_overlap=100
separators=["\n\n","\n",".","!","?","。","！","？"," ",""]
max_split_char_number=1000#文本分割的阈值（超过才分割，否则不分割）

#vector_stores
similarity_threshold=2#相似度检索阈值,即 检索返回的匹配文档的数量

#模型
embedding_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

#history
history_path="D:\XIAO\Python_project\RAG项目\chat_history"

#session_config
session_config={
        "configurable":{
            "session_id":"user_001",
        },
    }