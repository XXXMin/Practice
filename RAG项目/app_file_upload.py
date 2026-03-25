#streamlit框架开发的WEB网页服务
#streamlit有个机制：当WEB页面元素发生变化，则代码重新执行一遍（可能会导致状态的丢失）, 本质上就是一个字典
'''
用户在WEB网页上传文件
'''
import time

#基于streamlit框架完成WEB网页上传服务
import streamlit as st#可以用最快速度开发一个想要的网页
#streamlit 官方提供了组件.session_state:Streamlit 特有的用于在页面交互过程中持久化存储变量的机制
import knowledge_base
from knowledge_base import KnowledgeBaseService

#添加网页标题
st.title("知识库更新服务")
#添加文件上传服务
uploader_file=st.file_uploader(
    "请上传TXT文件",
    type=["txt"],#文件上传端所支持的文件类型
    accept_multiple_files=False,#是否接受多文件上传，False不接受
)

#session_state本质上就是一个字典
if "service" not in st.session_state:
    st.session_state["service"]=KnowledgeBaseService()

if uploader_file is not None:
    #文件不空，提取文件信息
    file_name=uploader_file.name
    file_type=uploader_file.type
    file_size=uploader_file.size/1024#单位是B，除以1024得到KB

    st.subheader(f"文件名：{file_name}")#字体比title小
    st.write(f"格式：{file_type} | 大小：{file_size:.2f}KB")#网页内显示正常大小的文本

    #获取上传文件的内容，得到的是字节数组（bytes）-》decode(按utf-8变成字符串)
    text=uploader_file.getvalue().decode("utf-8")#得到的是字节数组
    # st.write(text)

    #增加用户体验，让刷新不那么快出现
    with st.spinner("载入知识库中……"):#在spinner内的代码执行过程中，会有一个转圈动画
        time.sleep(1)#睡眠一会，别一下子跳出来
        result=st.session_state["service"].upload_by_str(text,file_name)#返回结果是个字符串，成功or跳过
        st.write(result)