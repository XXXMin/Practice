'''
VACP置信度评估节点
'''
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation.vacp_adapter import get_vacp_adapter, VACPAdapter
from langgraph_agent.state import MyAgentState

def vacp_node(state: MyAgentState):
    '''VACP置信度评估节点'''
    print("\n"+"="*50)
    print("进入VACP置信度评估节点")
    print("="*50)
    last_message=state["messages"][-1]
    response_text=last_message.content if hasattr(last_message, "content") else ""
    print(f"AI回复预览：{response_text[:100]}...")
    #获取适配器
    adapter=get_vacp_adapter()
    if adapter is None:
        print("VACP适配器未初始化，创建临时适配器")
        adapter=VACPAdapter()
    #获取state中的logits
    logits=state.get("last_logits",None)
    if logits:
        print(f"检测到logits，长度：{len(logits)}，使用VACP精确计算")
    else:
        print(f"未检测到logits，使用启发式方法")
    #调用适配器（传入logits）
    needs_review,confidence,reason=adapter.needs_human_review(
        response_text=response_text,
        logits=logits,
        confidence_threshold=0.7
    )
    print(f"\n置信度分数：{confidence:.3f}")
    if needs_review:
        print(f"置信度低于阈值，需要人工确认")
        print(f"原因：{reason}")
    else:
        print(f"置信度高于阈值，直接通过")
    return {
        "vacp_confidence": confidence,
        "needs_human_review": needs_review,
        "review_reason": reason or ""
    }