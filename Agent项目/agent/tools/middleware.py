from langchain.agents import AgentState
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing import Callable#顶级抽象
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts,load_report_prompts

@wrap_tool_call
#工具监控
def monitor_tool(
        request:ToolCallRequest,#请求的数据封装（类似入参）
        handler:Callable[[ToolCallRequest],ToolMessage|Command],#执行的函数本身（类似函数）
)->ToolMessage|Command:#工具执行的监控
    #一些日志打印，知道工具被调用了之类的
    logger.info(f"[tool_monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool_monitor]传入参数：{request.tool_call['args']}")

    try:
        result=handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name']=="fill_context_for_report":
            request.runtime.context["report"] = True  # 字典（上下文对象）report默认false，察觉调用工具就改成true（标记的注入）

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e#抛弃异常，程序被动停止

@before_model
#模型监控
def log_before_model(
        state:AgentState,#带有整个Agent智能体的状态结构
        runtime:Runtime,#记录整个执行过程中的上下文信息
):#模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
    logger.debug(f"[log_before_model] {type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")#啰嗦信息(-1:每次只取最新)(且取消空格和换行符)
    return None

@dynamic_prompt
#动态提示词（每次生成提示词之前调用此函数）
def report_prompt_switch(request:ModelRequest):#动态切换提示词
    is_report=request.runtime.context.get("report",False)#拿字典的值，拿不到返回false
    if is_report:#标识符起作用，是要生成报告，返回报告生成提示词内容
        return load_report_prompts()
    return load_system_prompts()