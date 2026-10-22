'''
langgraph构建:
Day02:支持工具调用
Day03:多轮工具调用支持
'''
import sqlite3

from langgraph.graph import StateGraph,END
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
from langgraph_agent.state import MyAgentState
from langgraph_agent.nodes.call_model import call_model_node
from langgraph_agent.nodes.tool_executor import tool_node
from langgraph_agent.nodes.vacp import vacp_node
from langgraph_agent.nodes.human_review import human_review_node

def build_simple_graph():
    '''
    先建个简单的graph吧：用户输入->LLM回复->结束
    '''
    builder=StateGraph(MyAgentState)
    #添加节点
    builder.add_node("agent",call_model_node)
    builder.add_node("tools",tool_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent",should_continue)
    builder.add_edge("tools","agent")
    # builder.set_finish_point("agent")#显式声明，非多终点可不用
    #编译
    graph=builder.compile()
    return graph

def should_continue(state:MyAgentState):
    '''
    判断是否需要继续调用工具
    条件边(conditional edge)的核心函数
    '''
    last_message=state["messages"][-1]
    #如果最后一条消息包含tool_calls，就执行工具
    if hasattr(last_message,"tool_calls") and last_message.tool_calls:
        print(f"检测到工具调用，路由到tools节点")
        return "tools"#返回节点名称
    #否则件数
    print(f"未检测到调用工具的需求，进入VACP评估")
    return 'vacp'

def after_vacp(state:MyAgentState):
    '''VACP评估后的路由'''
    if state.get("needs_human_review",False):
        print(f"VACP置信度低，需要人工确认")
        return "human_review"
    print(f"VACP置信度高，结束")
    return END

def after_human_review(state:MyAgentState):
    '''人工确认后的路由'''
    review_status=state.get("review_status","")
    if review_status=="approved":
        print(f"用户已确认，结束")
        return END
    elif review_status=="rejected":
        print(f"用户拒绝，需要重新生成回复")
        # 注意：这里可以重新路由到 agent，但需要处理循环
        # 简化版：直接结束
        return END
    else:
        # pending 状态，等待用户输入
        print(f"等待用户确认......")
        return END

def build_graph_without_persistence():
    '''
    构建完整 Graph:(不带持久化的graph)
    流程：agent → 判断->有工具 → tools → agent（循环）
                  ↓ 无工具 → vacp → 判断->低置信度 → human_review → END
                                    ↓ 高置信度 → END
    '''
    builder=StateGraph(MyAgentState)
    #添加节点
    builder.add_node("agent",call_model_node)
    builder.add_node("tools",tool_node)
    builder.add_node("vacp",vacp_node)
    builder.add_node("human_review",human_review_node)
    #设置入口
    builder.set_entry_point("agent")
    #条件边：从agent出发，根据should_continue的返回值决定下一步
    builder.add_conditional_edges("agent",should_continue)
    #普通边：tools执行完后，回到agent（让模型看到工具结果）
    builder.add_edge("tools","agent")
    #条件边：vacp → human_review 或 END
    builder.add_conditional_edges("vacp",after_vacp)
    #条件边：human_review → END（简化版）
    builder.add_conditional_edges("human_review",after_human_review)
    return builder.compile()

#带递归限制的版本（用于复杂多轮调用）
def build_agent_graph_with_limit(recursion_limit:int=50):
    '''
    构建带递归限制的graph
    recursion_limit:最大迭代次数，防止无限循环
    '''
    graph=build_graph_without_persistence()
    graph.recursion_limit=recursion_limit
    return graph

def build_graph_with_persistence(db_path:str=None):
    '''
    构建带持久化的 Graph:
    Args:
        db_path: SQLite 数据库路径，默认为 langgraph_agent/checkpoints/agent.db
    '''
    if db_path is None:
        db_path=Path(__file__).parent/"checkpoints"/"agent.db"
    #确保目录存在
    Path(db_path).parent.mkdir(parents=True,exist_ok=True)
    #创建SQLite持久化器
    #手动创建AQLite连接
    conn=sqlite3.connect(str(db_path),check_same_thread=False)#check_same_thread=False:允许多线程访问
    saver=SqliteSaver(conn)
    builder=StateGraph(MyAgentState)
    builder.add_node("agent",call_model_node)
    builder.add_node("tools",tool_node)
    builder.add_node("vacp",vacp_node)
    builder.add_node("human_review",human_review_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent",should_continue,{"tools":"tools","vacp":"vacp"})#映射表（路由函数返回值 → 实际节点名）
    builder.add_edge("tools","agent")
    builder.add_conditional_edges("vacp",after_vacp,{"human_review":"human_review","__end__":END})
    builder.add_conditional_edges("human_review",after_human_review,{"agent":"agent","__end__":END})
    #编译时传入checkpointer
    graph=builder.compile(checkpointer=saver)
    print(f"✅ Graph 已编译，持久化存储: {db_path}")
    return graph
#默认导出带持久化的版本
build_agent_graph=build_graph_with_persistence

def get_config(thread_id:str):
    '''
    获取会话配置：
    Args:
        thread_id: 会话唯一标识符
    Returns:
        用于 graph.invoke() 的 config 字典
    '''
    return {"configurable":{"thread_id":thread_id}}