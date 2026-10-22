from model.factory import chat_model
from agent.tools.agent_tools import get_weather, get_current_month

# 绑定工具
model_with_tools = chat_model.bind_tools([get_weather, get_current_month])

# 测试工具调用
response = model_with_tools.invoke("北京今天天气怎么样？")

print("模型回复:")
print(response.content)
print()
print("工具调用:")
if hasattr(response, "tool_calls") and response.tool_calls:
    for tc in response.tool_calls:
        print(f"  - {tc['name']}: {tc['args']}")
else:
    print("  无工具调用")