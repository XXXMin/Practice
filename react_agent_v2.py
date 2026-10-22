'''
ReactAgentV2 - LangGraph 版本的 Agent 封装:提供与原 ReactAgent 相同的接口，便于无缝替换
    支持：
    - 流式输出 (execute_stream)
    - 人工确认回调 (respond_to_review)
    - 多会话隔离 (session_id / thread_id)
    - VACP 置信度评估
'''
from pathlib import Path
import sys

#添加项目根路径到目录
sys.path.insert(0,str(Path(__file__).parent.parent))

from typing import Optional,Dict,Any,Generator
from langchain_core.messages import HumanMessage
from langgraph_agent.graph import build_graph_with_persistence,get_config
from langgraph_agent.nodes.human_review import human_review_node
from evaluation.vacp_adapter import init_vacp_adapter,get_vacp_adapter

class ReactAgentV2:
    '''
    LangGraph 版本的 Agent
    使用方式：
        agent = ReactAgentV2(session_id="user_123")
        for chunk in agent.execute_stream("你好"):
            print(chunk, end="")
        # 如果触发了人工确认
        if agent.needs_review():
            agent.respond_to_review(approved=True)
    Attributes:
        session_id: 会话标识符
        graph: LangGraph 实例
        config: 会话配置（包含 thread_id）
        _needs_review: 当前是否等待人工确认
        _pending_response: 待确认的回复内容
    '''
    def __init__(
            self,
            session_id:str="default",
            db_path:Optional[str]=None,
            auto_init_vacp:bool=True
    ):
        '''
        初始化 ReactAgentV2
        :param session_id:会话唯一标识符
        :param db_path:据库路径（默认使用 checkpoints/agent.db）
        :param auto_init_vacp:否自动初始化 VACP 适配器
        '''
        self.session_id = session_id
        self.thread_id=session_id#使用相同的ID作为thread_id
        #构建graph
        self.graph=build_graph_with_persistence(db_path)
        self.config=get_config(self.thread_id)
        #状态跟踪（私有属性）
        self._needs_review=False
        self._pending_response=None
        self._last_confidence=None
        #自动初始化VACP（如果尚未初始化）
        if auto_init_vacp and get_vacp_adapter() is None:
            self._init_vacp()
        print(f"ReactAgentV2 初始化完成")
        print(f"会话ID: {session_id}")
        print(f"持久化: SQLite")

    def _init_vacp(self):#私有方法，只在构造函数中调用
        '''初始化VACP适配器'''
        try:
            from model.factory import get_local_model
            import torch

            model,tokenizer=get_local_model()
            device="cuda" if torch.cuda.is_available() else "cpu"
            #加载掩码
            vocab_mask_path=Path(__file__).parent.parent/"vacp"/"vocab_mask.pt"
            if vocab_mask_path.exists():
                import torch as torch_pt
                vocab_mask=torch_pt.load(vocab_mask_path)
            else:
                vocab_mask=None
                print("⚠️ 未找到 vocab_mask，VACP 将使用启发式模式")
            init_vacp_adapter(
                model=model,
                tokenizer=tokenizer,
                vocab_mask=vocab_mask,
                temperature=1.0,
                device=device
            )
            print("VACP适配器已初始化")
        except Exception as e:
            print(f"VACP初始化失败: {e}，将使用启发式模式")

    def execute_stream(self,query:str)->Generator[str,None,None]:
        '''
        流式执行用户查询
        :param query:用户输入的问题
        :return:逐字返回 AI 回复内容
        '''
        #重置人工确认状态
        self._needs_review=False
        self._pending_response=None
        #构建初始状态
        initial_state={
            "messages": [HumanMessage(content=query)],
            "report_mode": False,
            "user_id": self.session_id,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "vacp_prediction_set": None,
            "vacp_confidence": None,
            "last_logits": None,
            "needs_human_review": False,
            "review_reason": "",
            "review_status": None,
            "pending_response": None
        }
        #执行graph
        try:
            #使用stream模式获取逐步输出
            for chunk in self.graph.stream(
                initial_state,
                config=self.config,
                stream_mode="values"
            ):
                if "messages" in chunk and chunk["messages"]:
                    #chunk：每一步执行后的状态快照（字典）
                    last_msg=chunk["messages"][-1]
                    if hasattr(last_msg,"content") and last_msg.content:
                        #检查是否人工确认消息
                        if "⚠️ **需要您确认**" in last_msg.content:
                            self._needs_review=True
                            self._pending_response={
                                "content":chunk.get("pending_response",{}).get("content",""),
                                "confidence":chunk.get("vacp_confidence",0),
                                "reason":chunk.get("review_reason","")
                            }
                            yield last_msg.content
                        else:
                            yield last_msg.content
                    #更新置信度
                    if chunk.get("vacp_confidence") is not None:
                        self._last_confidence=chunk["vacp_confidence"]
        except Exception as e:
            print(f"graph执行失败：{e}")
            yield f"抱歉，处理您的问题时出现了错误：{e}"

    def respond_to_review(self,approved:bool)->Dict[str,Any]:
        '''
        响应人工确认请求
        :param approved:True接受AI回复，False拒绝
        :return: 更新后的状态字典
        '''
        result=human_review_node.continue_with_response(approved)
        self._needs_review=False
        if approved:
            print("用户已确认，AI 回复将被使用")
        else:
            print("用户拒绝，建议重新提问")
        return result

    def needs_review(self)->bool:
        '''检查是否正在等待人工确认'''
        return self._needs_review

    def get_pending_response(self)->Optional[Dict]:
        '''获取待确认的回复内容'''
        return self._pending_response

    def get_last_confidence(self)->Optional[float]:
        '''虎丘最后一次回复的置信度'''
        return self._last_confidence

    def get_session_state(self)->Dict[str,Any]:
        '''
        获取当前会话的完整状态
        :return: 包含历史消息、置信度等的字典
        '''
        #从数据库获取最新状态
        try:
            state=self.graph.get_state(self.config)
            if state and state.values:
                messages=state.values.get("messages",[])
                return {
                    "session_id": self.session_id,
                    "message_count":len(messages),
                    "last_confidence":state.values.get("vacp_confidence"),
                    "needs_review":state.values.get("needs_human_review",False),
                    "messages":messages[-5:] if len(messages)>5 else messages#最近五条
                }
        except Exception as e:
            print(f"获取状态失败: {e}")
        return {
            "session_id": self.session_id,
            "message_count": 0,
            "last_confidence": self._last_confidence,
            "needs_review": self._needs_review
        }

    def reset_session(self):
        '''
        重置当前会话（清除历史）
        P.S.这会在数据库中创建新的thread_id
        '''
        new_thread_id=f"{self.session_id}_reset"
        self.thread_id=new_thread_id
        self.config=get_config(new_thread_id)
        self._needs_review=False
        self._pending_response=None
        self._last_confidence=None
        print(f"会话已重置，新 thread_id: {new_thread_id}")

    def get_thread_id(self)->str:
        '''获取当前thread_id'''
        return self.thread_id

    def save_state_to_db(self,state:Dict[str,Any]):
        '''保存状态到数据库'''
        try:
            #使用graph的update_state方法
            self.graph.update_state(self.config,state)
        except Exception as e:
            print(f"保存状态失败：{e}")

    def load_state_from_db(self)->Optional[Dict[str,Any]]:
        '''从数据库加载状态'''
        try:
            state=self.graph.get_state(self.config)
            if state and state.values:
                return state.values
        except Exception as e:
            print(f"加载状态失败：{e}")
        return None

class SessionManager:
    '''会话管理器：管理多个用户的Agent实例'''
    def __init__(self,db_path:Optional[str]=None):
        self._agents:Dict[str,ReactAgentV2]={}
        self.db_path=db_path

    def get_agent(self,session_id:str)->ReactAgentV2:
        '''获取或创建指定会话的Agent'''
        if session_id not in self._agents:
            self._agents[session_id]=ReactAgentV2(
                session_id=session_id,
                db_path=self.db_path,
            )
        return self._agents[session_id]

    def remove_agent(self,session_id:str):
        '''移除指定会话的Agent'''
        if session_id in self._agents:
            del self._agents[session_id]

    def get_all_sessions(self)->list:
        '''获取所有活跃会话ID'''
        return list(self._agents.keys())

#全局会话管理器
_default_session_manager=None

def get_session_manager(db_path:Optional[str]=None)->SessionManager:
    '''获取全局会话管理器：确保整个应用共享同一个 SessionManager 实例'''
    global _default_session_manager
    if _default_session_manager is None:
        _default_session_manager=SessionManager(db_path)
    return _default_session_manager