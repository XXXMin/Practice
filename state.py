from typing import List,Annotated,TypedDict,Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class MyAgentState(TypedDict):
    '''
    langgraph的状态定义
    '''
    #对话核心
    messages: Annotated[List[BaseMessage],add_messages]
    #报告生成控制（是否）
    report_mode:bool
    #用户ID
    user_id:Optional[str]
    #VACP相关
    vacp_precision_set:Optional[List[str]]#预测集合
    vacp_confidence:Optional[float]#置信度0-1
    last_logits:Optional[List[float]]#存储最后一次的logits
    #人工确认相关
    needs_human_review:bool#是否需要人工确认
    review_reason:Optional[str]#确认原因
    review_status:Optional[str]#确认状态：pending,approved,rejected
    pending_response:Optional[dict]#待确认的回复内容
    #会话管理
    session_id:Optional[str]#会话标识符
    thread_id:Optional[str]#LangGraph thread_if