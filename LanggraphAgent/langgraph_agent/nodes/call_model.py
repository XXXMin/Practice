'''
LLM调用节点：对应原来的ReactAgent的核心逻辑
           支持工具绑定
           支持日志(Day03)
           支持RAG工具
'''
from langchain_core.messages import SystemMessage
from model.factory import chat_model,get_local_model
from utils.prompt_loader import load_system_prompts,load_report_prompts
from langgraph_agent.state import MyAgentState
from agent.tools.agent_tools import get_weather,get_user_id,get_user_location,get_current_month,rag_summarize
from langgraph_agent.nodes.logger_middleware import log_before_model
import torch

def call_model_node(state:MyAgentState):
    '''
    调用LLM的节点
    input：当前state(包含历史对话、report_mode等)
    output: 更新后的state(添加LLM的回复)
    Day02:搞得支持工具调用
    '''
    #调用日志中间件（记录调用前状态）
    log_before_model(state)
    #根据report_mode选择系统提示词
    if state.get("report_mode",False):
        system_prompts=load_report_prompts()
        print(f"报告生成模式")
    else:
        system_prompts=load_system_prompts()
        print(f"普通对话模式")
    #工具
    tools=[
        rag_summarize,
        get_weather,
        get_user_location,
        get_user_id,
        get_current_month
    ]
    #使用bind_tools将工具绑定到模型
    model_with_tools=chat_model.bind_tools(tools)
    print(f"已绑定的工具: {[t.name for t in tools]}")
    #调用模型
    print(f"调用模型，当前消息数：{len(state['messages'])}")
    # 构造带 system prompt 的消息列表
    messages_with_system = [SystemMessage(content=system_prompts)] + state["messages"]
    response = model_with_tools.invoke(messages_with_system)

    #打印下模型是否调用了工具
    if hasattr(response,"tool_calls") and response.tool_calls:
        tool_names = [tc['name'] for tc in response.tool_calls]
        print(f"🔧 模型调用工具：{tool_names}")
        if 'rag_summarize' in tool_names:
            print(f"📚 触发了 RAG 检索")
    else:
        print(f"模型没有调用工具，直接回复")
    #尝试获取logits（用于VACP）
    logits=None
    try:
        #获取本地模型
        local_model,local_tokenizer=get_local_model()
        #获取最后一条用户消息
        # 🆕 用主 LLM 的回复作为输入，让本地模型"复述"一遍，捕获生成时的 logits
        ai_response = response.content if hasattr(response, "content") else ""
        if ai_response:
            # 构造对话格式
            chat_messages = []
            for msg in state["messages"]:
                if hasattr(msg, "type"):
                    if msg.type == "human":
                        chat_messages.append({"role": "user", "content": msg.content})
                    elif msg.type == "ai":
                        chat_messages.append({"role": "assistant", "content": msg.content})
            # 用 chat template 构造输入
            text = local_tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = local_tokenizer(text, return_tensors="pt")
            device = next(local_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = local_model(**inputs)
                #取"生成回复第一个 token"位置的 logits
                last_logits = outputs.logits[0, -1, :]
                logits = last_logits.cpu().tolist()#state 中的字段需要 JSON 可序列化，PyTorch 张量不能直接存
            print(f"成功捕获生成 logits，长度：{len(logits)}")
    except Exception as e:
        print(f"无法捕获logits：{e}，VACP将使用启发式方法")
    #返回更新
    return {
        "messages":[response],
        "last_logits":logits#存储logits
    }