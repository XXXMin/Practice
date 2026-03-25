#streamlit完成一个聊天的UI（提供一个web网页，里面有对话框，提出问题，等待回答）
#st.session_state 本质上就是一个字典
import time
from rag import RagService
import streamlit as st
import config_data as config

#标题
st.title("智能会话（by 小吴）")
st.divider()#分隔符（让分隔符下面是用户的对话框）

if "message" not in st.session_state:
    st.session_state["message"]=[{"role":"assistant","content":"你好，有什么可以帮助你的？"}]

if "rag" not in st.session_state:
    st.session_state["rag"]=RagService()

for messege in st.session_state["message"]:
    st.chat_message(messege["role"]).write(messege["content"])

#在页面最下方提供用户输入栏
prompt=st.chat_input()

if prompt:
    #如果用户输入了,在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    ai_res_list=[]
    with st.spinner("思考中……"):
        # res = st.session_state["rag"].chain.invoke({"input":prompt},config.session_config)
        res_stream= st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)
        #.stream返回的是一个生成器对象（Generator），而生成器是迭代器的一种

        #捕获函数：
        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk#yield:用于创建生成器（generator）的关键字，它让函数可以逐步产生值，而不是一次性返回所有结果,函数暂停，状态保留（可迭代，可多次执行）

        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})
        #"".join(ai_res_list)快速将list变成了str（将list里面所有str连成了一个字符串），且片段之间用空字符串连接
