"""
扫地机器人智能客服——小吴
LangGraph + VACP 版本
功能：
- 多轮对话
- 工具调用（天气、位置等）
- VACP 置信度评估
- 低置信度时人工确认
- SQLite 持久化
- 文件上传+知识库管理
- 架构图展示
"""
import streamlit as st
import time
#导入辅助函数
from utils.streamlit_session import init_session_state,get_or_create_agent,add_message,clear_messages,set_review_state,get_session_stats,update_knowledge_base_status,rebuild_vector_store,save_uploaded_file,clear_vector_store,reset_vector_store

#页面配置
st.set_page_config(
    page_title="扫地机器人智能客服——小吴",
    page_icon="🤖",
    layout="wide"
)
#初始化会话状态
init_session_state()
#标题
st.title("🤖 扫地机器人智能客服——小吴")
st.caption(f"会话 ID: {st.session_state.session_id} | VACP 置信度评估已启用")
#侧边栏
with st.sidebar:
    st.markdown("### 📊 会话信息")
    stats=get_session_stats()
    st.metric("消息数量",stats["message_count"])
    if stats["last_confidence"] is not None:
        confidence=stats["last_confidence"]
        color="🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.5 else "🔴"
        st.metric("最后置信度", f"{color} {confidence:.2f}")
    st.divider()
    st.markdown("### 🔧 功能")
    st.markdown("- ✅ 多轮对话")
    st.markdown("- ✅ 工具调用（天气/位置/时间）")
    st.markdown("- ✅ RAG 知识库检索")
    st.markdown("- ✅ VACP 置信度评估")
    st.markdown("- ✅ 低置信度人工确认")
    st.markdown("- ✅ 会话持久化")
    st.divider()
    #RAG知识库管理
    st.markdown("### 📁 知识库管理")
    #显示当前知识库状态
    doc_count=update_knowledge_base_status()
    if doc_count > 0:
        st.success(f"✅ 知识库已就绪 ({doc_count} 个文档)")
    else:
        st.warning("⚠️ 知识库为空，请上传文档")
    #文件上传
    uploaded_file=st.file_uploader(
        "上传文档到知识库",
        type=["pdf", "txt", "md"],
        help="支持 PDF、TXT、Markdown 格式。上传后会自动添加到知识库。"
    )
    if uploaded_file is not None:
        try:
            #保存文件
            save_path=save_uploaded_file(uploaded_file)
            st.success(f"✅ 已上传: {uploaded_file.name}")
            #自动更新向量库
            with st.spinner("正在更新知识库向量索引..."):
                new_count=rebuild_vector_store()
            st.success(f"✅ 知识库已更新！当前文档数: {new_count}")
            st.rerun()
        except Exception as e:
            st.error(f"❌ 上传失败: {e}")
    #已上传文件列表
    if st.session_state.uploaded_files:
        with st.expander("📄 已上传的文件"):
            for f in st.session_state.uploaded_files:
                st.caption(f"- {f}")
    st.divider()
    #三个管理按钮
    col1,col2,col3=st.columns(3)
    with col1:
        if st.button("🔄 重新加载", use_container_width=True, help="读取 data/ 文件夹中的文档，增量添加"):
            with st.spinner("正在重新加载文档..."):
                new_count=rebuild_vector_store()
            if new_count is not None:
                st.success(f"✅ 重新加载完成！文档数: {new_count}")
            else:
                st.error("❌ 重新加载失败")
            st.rerun()
    with col2:
        if st.button("🗑️ 清空数据", use_container_width=True, help="删除所有向量数据，清空知识库"):
            with st.spinner("正在清空知识库..."):
                result = clear_vector_store()
            if result is not None:
                st.success("✅ 知识库已清空！")
            else:
                st.error("❌ 清空失败")
            st.rerun()
    with col3:
        if st.button("💣 完全重置", use_container_width=True, help="删除整个数据库目录，完全重置"):
            with st.spinner("正在完全重置知识库..."):
                result = reset_vector_store()
            if result is not None:
                st.success("✅ 知识库已完全重置！")
            else:
                st.error("❌ 重置失败")
            st.rerun()
    st.caption("💡 提示：清空/重置后需要重新上传文档")
    #会话控制
    if st.button("🗑️ 清空对话",use_container_width=True):
        clear_messages()
        st.rerun()
    if st.button("🔄 重置 Agent", use_container_width=True):
        agent=get_or_create_agent()
        agent.reset_session()
        clear_messages()
        st.success("Agent已重置！")
        time.sleep(1)
        st.rerun()
    #架构图展示
    st.divider()
    st.markdown("### 系统架构")

    from pathlib import Path
    arch_png=Path(__file__).parent / "assets" / "agent_architecture.png"
    arch_mmd=Path(__file__).parent / "assets" / "agent_architecture.mmd"

    if arch_png.exists():
        with st.expander("查看架构图",expanded=False):
            st.image(str(arch_png),caption="LangGraph Agent 架构",use_column_width=True)
    else:
        st.caption("架构图未生成")
        st.caption("运行 `python scripts/visualize_graph.py` 生成")

    if arch_mmd.exists():
        with st.expander("Mermaid 源码",expanded=False):
            with open(arch_mmd,"r",encoding="utf-8") as f:
                mermaid_code=f.read()
            st.code(mermaid_code,language="mermaid")
    with st.expander("节点与边",expanded=False):#创作可折叠区域
        from scripts.visualize_graph import get_graph_structure
        try:
            structure = get_graph_structure()
            st.markdown(f"**节点({len(structure['nodes'])}个)：**")
            for node in structure['nodes']:
                st.caption(f"- {node}")
            st.markdown(f"**边 ({len(structure['edges'])} 条):**")
            for edge in structure['edges']:
                st.caption(f"- {edge['source']} -> {edge['target']}")
        except Exception as e:
            st.caption(f"获取结构失败: {e}")

#显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        #如果是AI回复且包含置信度，显示小字
        if message["role"]=="assistant" and "confidence" in message:
            confidence=message["confidence"]
            # 🆕 添加 None 检查
            if confidence is not None:
                if confidence<0.7:
                    st.caption(f"⚠️ 置信度: {confidence:.2f} (低于阈值)")
                else:
                    st.caption(f"✅ 置信度: {confidence:.2f}")
            else:
                st.caption("置信度: 未计算")
#人工确认处理
if st.session_state.awaiting_review:
    st.divider()
    st.warning("⚠️ **需要您确认**")
    pending=st.session_state.pending_response
    if pending:
        st.markdown(f"**AI 回复预览**: {pending.get('content', '')[:200]}...")
        st.markdown(f"**置信度**: {pending.get('confidence', 0):.2f}")
        st.markdown(f"**原因**: {pending.get('reason', '未知')}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 接受", use_container_width=True, key="accept_btn"):
            agent=get_or_create_agent()
            agent.respond_to_review(approved=True)
            set_review_state(False,None)
            st.rerun()
    with col2:
        if st.button("❌ 拒绝", use_container_width=True, key="reject_btn"):
            agent=get_or_create_agent()
            agent.respond_to_review(approved=False)
            set_review_state(False,None)
            st.info("已拒绝，请重新提问")
            st.rerun()
#用户输入处理
prompt=st.chat_input("请输入您的问题......")
if prompt and not st.session_state.awaiting_review:
    #显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    add_message("user",prompt)
    #获取agent
    agent=get_or_create_agent()
    #显示AI回复
    with st.chat_message("assistant"):
        response_placeholder=st.empty()
        full_response=""
        try:
            #流式输出
            for chunk in agent.execute_stream(prompt):
                full_response+=chunk
                response_placeholder.markdown(full_response+ "▌")
            response_placeholder.markdown(full_response)
            #获取置信度
            confidence=agent.get_last_confidence()
            st.session_state.last_confidence=confidence
            #保存消息(带置信度)
            msg_with_confidence={
                "role":"assistant",
                "content":full_response,
                "confidence":confidence
            }
            st.session_state.messages.append(msg_with_confidence)
            st.session_state.total_messages+=1
            #更新消息显示置信度
            if confidence is not None:
                if confidence<0.7:
                    st.caption(f"⚠️ 置信度: {confidence:.2f} (低于阈值)")
                else:
                    st.caption(f"✅ 置信度: {confidence:.2f}")
            #检查是否需要人工确认
            if agent.needs_review():
                pending=agent.get_pending_response()
                set_review_state(True,pending)
                st.rerun()
        except Exception as e:
            error_msg=f"❌ 处理失败: {e}"
            response_placeholder.markdown(error_msg)
            add_message("assistant",error_msg)
#脚注
st.divider()
st.caption("💡 提示：当 AI 置信度较低时，系统会请求您的确认。上传 PDF/TXT 文件后，可以问关于文档内容的问题。")