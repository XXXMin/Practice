from model.factory import chat_model

print("测试 deepseek-v4.1-flash 调用...")
try:
    response = chat_model.invoke("你好，请回复'测试成功'")
    print(f"✅ 调用成功: {response.content}")
except Exception as e:
    print(f"❌ 调用失败: {type(e).__name__}: {e}")