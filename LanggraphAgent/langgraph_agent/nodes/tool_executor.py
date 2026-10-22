'''
工具执行节点：执行模型调用的工具
           执行日志
           执行日志
'''
from langgraph.prebuilt import ToolNode
from agent.tools.agent_tools import get_weather,get_user_location,get_user_id,get_current_month,rag_summarize
from langgraph_agent.nodes.logger_middleware import log_tool_call

#定义工具列表，和call_model里保持一致
tools=[
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month
]
#创建原始ToolNode
_original_tool_node=ToolNode(tools)
#创建带日志的工具节点包装器
class LoggingToolNode:
    '''
    带日志的ToolNode包装器
    '''
    def __init__(self,wrapped_node):
        self._wrapped_node=wrapped_node
    def __call__(self,state:dict)->dict:
        '''执行工具并记录日志'''
        #获取最后一次消息中的tool_calls
        last_message=state["messages"][-1]
        #记录执行前日志
        if hasattr(last_message,"tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                #记录日志
                log_tool_call(
                    tool_name=tool_call.get("name","unknown"),
                    tool_input=tool_call.get("args",{}),
                    tool_result=""#执行后再更新
                )
        #执行原始ToolNode
        result=self._wrapped_node.invoke(state)
        #打印工具返回结果（方便调试）
        if "messages" in result and result["messages"]:
            for msg in result["messages"]:
                if hasattr(msg,"content") and msg.content:
                    print(f"工具返回：{msg.content[:100]}")
        return result
#导出带日志的工作节点
tool_node=LoggingToolNode(_original_tool_node)