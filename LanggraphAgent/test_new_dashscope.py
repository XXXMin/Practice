# test_new_dashscope.py
import os
from langchain_dashscope import ChatDashScope

# 确保你的 API Key 已设置在环境变量中
# os.environ["DASHSCOPE_API_KEY"] = "你的Key"

try:
    model = ChatDashScope(model="qwen-max")
    response = model.invoke("你好，请回复一个‘测试成功’")
    print(f"✅ 新包调用成功: {response.content}")
except Exception as e:
    print(f"❌ 新包调用失败: {type(e).__name__}: {e}")