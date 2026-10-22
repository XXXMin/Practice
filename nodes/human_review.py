'''
人工确认节点：低置信度时等待用户确认
'''
from langchain_core.messages import AIMessage
from langgraph_agent.state import MyAgentState

class HumanReview:
    '''
    人工确认节点：
        当VACP置信度低于阈值时，进入此节点
        返回确认消息，等待用户输入yes/no
        支持中断恢复（状态持久化）
    '''
    def __init__(self):
        self.pending_state=None#存储待确认状态
        self.user_response=None#用户响应

    def __call__(self,state:MyAgentState)->dict:#使类的实例可以像函数一样调用
        '''
        进入确认状态，返回询问消息
        Args:
            state:当前状态（包含低置信度的回复）
        Return:
            包含确认消息的更新
        '''
        print("\n" + "=" * 50)
        print("👤 进入人工确认节点")
        print("=" * 50)
        #获取会话信息
        session_id=state.get("session_id","unknown")
        thread_id=state.get("thread_id","unknown")
        print(f"会话: session_id={session_id}, thread_id={thread_id}")
        #获取置信度和原因
        confidence=state.get("vacp_confidence",0.0)
        review_reason=state.get("review_reason","未知原因")
        #获取待确认的AI回复
        last_message=state["messages"][-1]
        response_text=last_message.content if hasattr(last_message,"content") else ""
        #构建确认消息
        confirm_msg = f"""⚠️ **需要您确认**
        **原因**: {review_reason}
        **置信度**: {confidence:.2f}

        **AI 回复预览**:
        {response_text[:300]}{"..." if len(response_text) > 300 else ""}
        -----------------------------------------
        请回复 **yes** 接受，或 **no** 拒绝/重新回答
        """
        #创建确认消息
        review_message=AIMessage(content=confirm_msg.strip())
        #保存待确认的状态，以便后续恢复
        self.pending_state=state
        print(f"📊 置信度: {confidence:.3f}")
        print(f"📝 确认原因: {review_reason}")
        print(f"💬 已发送确认请求")
        return {
            "messages":[review_message],
            "needs_human_review":True,
            "review_status":"pending",#标记等待确认
            "pending_response":{
                "content":response_text,
                "confidence":confidence,
                "reason":review_reason
            }
        }

    def continue_with_response(self,approved:bool)->dict:
        '''
        处理用户的人工确认响应:
        Args:
            approved: True 表示接受，False 表示拒绝
        Returns:
            更新后的状态
        '''
        self.user_response=approved
        if approved:
            print("✅用户已确认，继续返回结果")
            return {
                "needs_human_review":False,
                "review_status":"approved"
            }
        else:
            print("❌用户拒绝，需要重新生成回复")
            return {
                "needs_human_review": False,
                "review_status": "rejected"
            }

    def get_pending_response(self)->dict:
        '''获取待确认的回复内容'''
        return self.pending_state.get("pending_response", {}) if self.pending_state else {}

    def reset(self):
        '''重置节点状态'''
        self.pending_state=None
        self.user_response=None
        print("人工确认节点已重置")

#全局实例
human_review_node=HumanReview()

def get_human_review_node():
    '''获取人工确认节点实例'''
    return human_review_node

def reset_human_review():
    '''重置人工确认节点状态（用于新会话）'''
    global human_review_node
    human_review_node=HumanReview()
    return human_review_node