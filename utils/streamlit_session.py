'''
Streamlit 会话管理辅助函数:
    提供：
        会话ID生成和管理
        消息历史存储
        Agent实例管理
'''
import streamlit as st
from typing import Optional,Dict,Any
from datetime import datetime
import uuid

def init_session_state():
    '''
    初始化 Streamlit 会话状态：在 app 启动时调用一次
    '''
    #会话ID（用于区分不同用户）
    if "session_id" not in st.session_state:
        st.session_state.session_id=str(uuid.uuid4())[:8]
    #消息历史
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    #Agent实例
    if "agent" not in st.session_state:
        st.session_state.agent=None
    #人工确认状态
    if "awaiting_review" not in st.session_state:
        st.session_state.awaiting_review=False
    if "pending_response" not in st.session_state:
        st.session_state.pending_response=None
    #VACP置信度显示
    if "last_confidence" not in st.session_state:
        st.session_state.last_confidence=None
    #统计信息
    if "total_messages" not in st.session_state:
        st.session_state.total_messages=0
    #RAG知识库状态
    if "knowledge_base_ready" not in st.session_state:
        st.session_state.knowledge_base_ready=False
    if "knowledge_base_doc_count" not in st.session_state:
        st.session_state.knowledge_base_doc_count=0
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files=[]

def get_or_create_agent():
    '''获取或创建Agent实例:确保整个会话使用同一个 Agent'''
    from langgraph_agent import ReactAgentV2
    
    if st.session_state.agent is None:
        with st.spinner("正在初始化智能助手......"):
            st.session_state.agent = ReactAgentV2(
                session_id=st.session_state.session_id,
                auto_init_vacp=True
            )
    return st.session_state.agent

def add_message(role:str,content:str):
    '''添加消息到历史'''
    st.session_state.messages.append({
        "role":role,
        "content":content,
        "timestamp":datetime.now().isoformat()
    })
    st.session_state.total_messages+=1

def clear_messages():
    '''清空历史消息'''
    st.session_state.messages=[]
    st.session_state.total_messages=0
    st.session_state.last_confidence=None

def set_review_state(needs_review:bool,pending_response:Optional[Dict]=None):
    '''设置人工确认状态'''
    st.session_state.awaiting_review=needs_review
    if pending_response:
        st.session_state.pending_response=pending_response

def get_session_stats()->Dict[str, Any]:
    '''获取会话统计信息'''
    return {
        "session_id":st.session_state.session_id,
        "message_count":len(st.session_state.messages),
        "total_messages":st.session_state.total_messages,
        "awaiting_review":st.session_state.awaiting_review,
        "last_confidence":st.session_state.last_confidence,
        "knowledge_base_ready":st.session_state.knowledge_base_ready,
        "foc_count":st.session_state.knowledge_base_doc_count
    }

#RAG相关函数
def update_knowledge_base_status():
    '''更新知识库状态（检查向量库中的文档数量）'''
    try:
        from rag.vector_store import VectorStoreService
        vs=VectorStoreService()
        doc_count=vs.vector_store._collection.count()
        st.session_state.knowledge_base_ready=doc_count>0
        st.session_state.knowledge_base_doc_count=doc_count
        return doc_count
    except Exception as e:
        print(f"获取知识库状态失败: {e}")
        st.session_state.knowledge_base_ready = False
        st.session_state.knowledge_base_doc_count = 0
        return 0

def rebuild_vector_store():
    """
    重建向量库（清空现有数据后，重新加载 data/ 目录中的所有文档）
    执行步骤：
    1. 清空向量库中的所有文档
    2. 清空 md5 记录文件
    3. 重新加载 data/ 目录中的文档
    4. 返回新的文档数量
    Returns:
        int: 加载后的文档数量，失败返回 None
    """
    from rag.vector_store import VectorStoreService
    import os
    from utils.path_tool import get_abs_path
    from utils.config_handler import chroma_conf
    import gc
    try:
        # 强制垃圾回收，释放可能的连接
        gc.collect()
        vs = VectorStoreService()
        #通过 ID 删除所有文档
        try:
            all_ids = vs.vector_store.get()['ids']
            if all_ids:
                vs.vector_store.delete(ids=all_ids)
                print(f"✅ 已删除 {len(all_ids)} 个文档")
            else:
                print("⚠️ 集合已经是空的")
        except Exception as e:
            print(f"删除文档失败: {e}")
            # 备用方案：重建集合
            collection_name = vs.vector_store._collection.name
            vs.vector_store._client.delete_collection(collection_name)
            vs.vector_store._client.create_collection(collection_name)
            print(f"✅ 已重建集合: {collection_name}")
        # 清空 md5 记录文件
        md5_file_path = get_abs_path(chroma_conf['md5_hex_store'])
        if os.path.exists(md5_file_path):
            open(md5_file_path, "w", encoding="utf-8").close()
            print(f"✅ 已清空 md5 记录: {md5_file_path}")
        else:
            # 确保目录存在
            os.makedirs(os.path.dirname(md5_file_path), exist_ok=True)
            open(md5_file_path, "w", encoding="utf-8").close()
        # 重新加载 data/ 目录中的文档
        print("🔄 开始重新加载 data/ 目录中的文档...")
        vs.load_document()
        # 获取并返回新的文档数量
        new_count = vs.vector_store._collection.count()
        print(f"✅ 重建完成！新文档数: {new_count}")
        # 更新 session 状态
        st.session_state.knowledge_base_ready = new_count > 0
        st.session_state.knowledge_base_doc_count = new_count
        return new_count
    except Exception as e:
        print(f"❌ 重建向量库失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_uploaded_file(uploaded_file)->str:
    '''保存上传的文件到data目录'''
    from utils.path_tool import get_abs_path
    import os

    save_path=get_abs_path(f"data/{uploaded_file.name}")
    with open(save_path,"wb") as f:
        f.write(uploaded_file.getbuffer())
    #记录已上传的文件
    if uploaded_file.name not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.append(uploaded_file.name)
    return save_path

def clear_vector_store():
    '''清空向量库（删除所有向量数据，保留空库）'''
    from rag.vector_store import VectorStoreService
    from utils.path_tool import get_abs_path
    from utils.config_handler import chroma_conf
    import os
    try:
        vs=VectorStoreService()
        #删除集合中的所有文档
        try:
            all_ids=vs.vector_store.get()['ids']
            if all_ids:
                vs.vector_store.delete(ids=all_ids)
                print(f"✅ 已删除 {len(all_ids)} 个文档")
            else:
                print("⚠️ 集合已经是空的")
        except Exception as e:
            print(f"删除文档失败: {e}")
            # 备用方案：尝试删除整个集合后重建
            collection_name = vs.vector_store._collection.name
            vs.vector_store._client.delete_collection(collection_name)
            vs.vector_store._client.create_collection(collection_name)
            print(f"✅ 已重建集合: {collection_name}")
        #同时清空 md5 记录文件（记录哪些文件已加载）
        md5_file_path=get_abs_path(chroma_conf['md5_hex_store'])
        if os.path.exists(md5_file_path):
            open(md5_file_path, "w", encoding="utf-8").close()
            print("✅ md5 记录已清空")
        #更新session状态
        st.session_state.knowledge_base_ready = False
        st.session_state.knowledge_base_doc_count = 0
        return 0
    except Exception as e:
        print(f"清空失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def reset_vector_store():
    '''完全重置向量库（删除整个数据库目录，重建空库）'''
    import shutil
    from pathlib import Path
    from utils.path_tool import get_abs_path
    from utils.config_handler import chroma_conf
    import gc
    try:
        #先关闭可能的连接
        import chromadb
        #强制垃圾回收
        gc.collect()
        #获取chroma数据库路径
        chroma_path=get_abs_path(chroma_conf.get("persist_directory","chroma_db"))
        md5_file_path=get_abs_path(chroma_conf.get("md5_hex_store"))
        print(f"数据库路径: {chroma_path}")
        print(f"md5记录路径: {md5_file_path}")
        import time
        time.sleep(0.5)
        #删除数据库目录
        if Path(chroma_path).exists():
            # 尝试重命名后删除（避免占用问题）
            temp_path = chroma_path + "_old_" + str(int(time.time()))
            try:
                Path(chroma_path).rename(temp_path)
                print(f"✅ 已重命名为: {temp_path}")
                # 异步删除重命名的文件夹
                import threading
                def delayed_delete():
                    time.sleep(1)
                    try:
                        shutil.rmtree(temp_path, ignore_errors=True)
                    except:
                        pass

                threading.Thread(target=delayed_delete, daemon=True).start()
            except Exception as e:
                print(f"重命名失败，尝试直接删除: {e}")
                shutil.rmtree(chroma_path, ignore_errors=True)
            print(f"✅ 已处理数据库目录")
        #删除MD5记录文件
        if Path(md5_file_path).exists():
            Path(md5_file_path).unlink()
            print(f"✅ 已删除 md5 记录: {md5_file_path}")
        #重新创建空向量库（会自动创建新目录）
        from rag.vector_store import VectorStoreService
        vs=VectorStoreService()
        print("✅ 已创建空向量库")
        #更新session状态
        st.session_state.knowledge_base_ready = False
        st.session_state.knowledge_base_doc_count = 0
        st.session_state.uploaded_files=[]
        return 0
    except Exception as e:
        print(f"重置失败: {e}")
        import traceback
        traceback.print_exc()
        return None