'''
日志中间件(迁移原来agent的log_before_model和monitor_tool)
'''
from utils.logger_handler import logger
from datetime import datetime

def log_before_model(state):
    '''
    模型调用前的日志
    '''
    messages=state.get('messages',[])
    last_message=messages[-1] if messages else None
    logger.info(f"[模型调用] 时间：{datetime.now()}")
    logger.info(f"[模型调用] 当前消息数：{len(messages)}")
    if last_message and hasattr(last_message, 'content'):
        #打印前100字（不要太长，只是看下）
        preview=last_message.content[:100] if last_message.content else ""
        logger.info(f"[模型调用] 用户输入预览：{preview}")
    logger.info(f"[模型调用] report_mode:{state.get('report_mode',False)}")

def log_tool_call(tool_name:str,tool_input:dict,tool_result:str=""):
    '''
    工具调用的日志
    '''
    logger.info(f"[工具调用] 工具名称：{tool_name}")
    logger.info(f"[工具调用] 输入参数：{tool_input}")
    if tool_result:
        result_preview=tool_result[:200] if len(tool_result)>200 else tool_result
        logger.info(f"[工具调用] 结果预览：{result_preview}")
    return True#返回True表示监控通过