# 🤖 扫地机器人智能客服——小吴

基于 **LangGraph** 构建的智能客服 Agent，集成 **RAG 知识库检索** 和 **VACP 不确定性量化**，实现"知道自己不知道"的可信 AI 系统。

---

## 📌 项目亮点

| 特性 | 说明 |
|------|------|
| **LangGraph 状态机** | 显式 State 管理，支持条件路由、循环、中断恢复 |
| **VACP 不确定性量化** | 基于词汇表感知的保形预测，精确计算置信度 |
| **人机协作** | 低置信度时自动暂停，等待用户确认 |
| **RAG 知识库** | 支持 PDF/TXT/MD 上传，向量检索增强生成 |
| **多轮工具调用** | 天气、位置、时间、知识库检索等工具 |
| **会话持久化** | SQLite Checkpoint，支持跨会话记忆 |
| **多会话隔离** | 不同用户独立存储，互不干扰 |

---

## 🏗️ 系统架构

```
用户输入
    ↓
┌─────────────────────────────────────────┐
│           LangGraph Agent                │
│                                          │
│  ┌─────────┐    ┌─────────┐             │
│  │  agent  │───→│  tools  │             │
│  │ (LLM)   │←───│ (工具)  │             │
│  └────┬────┘    └─────────┘             │
│       │                                  │
│       ↓                                  │
│  ┌─────────┐                            │
│  │  vacp   │  ← VACP 置信度评估          │
│  └────┬────┘                            │
│       │                                  │
│       ├── 置信度 ≥ 0.7 ──→ END           │
│       │                                  │
│       └── 置信度 < 0.7 ──→ human_review  │
│                              ↓           │
│                         用户确认          │
│                              ↓           │
│                         approved/rejected │
└─────────────────────────────────────────┘
```

![架构图](assets/agent_architecture.png)

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n langgraph_agent python=3.11
conda activate langgraph_agent

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

在 `model/factory.py` 中配置阿里云百炼 API Key，或在环境变量中设置：

```bash
export DASHSCOPE_API_KEY=your_api_key
```

### 3. 构建知识库

```bash
# 将 PDF/TXT 文档放入 data/ 目录
# 构建向量库
python -c "from rag.vector_store import VectorStoreService; VectorStoreService().load_document()"
```

### 4. 构建 VACP 词汇表掩码

```bash
python vacp/build_mask.py
```

### 5. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`

---

## 📁 项目结构

```
LanggraphAgent/
├── agent/                    # 原 LangChain Agent（保留对比）
│   ├── react_agent.py
│   └── tools/
│       ├── agent_tools.py
│       └── middleware.py
│
├── langgraph_agent/          # LangGraph 实现
│   ├── state.py              # State 定义
│   ├── graph.py              # Graph 构建
│   ├── react_agent_v2.py     # Agent 封装
│   └── nodes/
│       ├── call_model.py     # LLM 调用节点
│       ├── tool_executor.py  # 工具执行节点
│       ├── vacp.py           # VACP 评估节点
│       ├── human_review.py   # 人工确认节点
│       └── logger_middleware.py
│
├── vacp/                     # VACP 复现
│   ├── config.py
│   ├── scorer.py
│   ├── calibrator.py
│   ├── temperature_tuning.py
│   ├── vocab_mask.py
│   └── build_mask.py
│
├── evaluation/               # 评估模块
│   └── vacp_adapter.py       # VACP 适配器
│
├── rag/                      # RAG 服务
│   ├── vector_store.py
│   └── rag_service.py
│
├── model/                    # 模型工厂
│   └── factory.py
│
├── utils/                    # 工具函数
│   ├── config_handler.py
│   ├── file_handler.py
│   ├── logger_handler.py
│   ├── path_tool.py
│   ├── prompt_loader.py
│   └── streamlit_session.py  # Streamlit 会话管理
│
├── scripts/                  # 脚本
│   └── visualize_graph.py    # 架构图导出
│
├── config/                   # 配置文件
├── data/                     # 知识库文档
├── prompts/                  # 提示词
├── assets/                   # 架构图等资源
├── docs/                     # 文档
│   ├── ARCHITECTURE.md
│   ├── RESUME.md
│   └── PROJECT_SUMMARY.md
├── app.py                    # Streamlit 主界面
└── requirements.txt
```

---

## 🔬 VACP 不确定性量化

### 什么是 VACP？

VACP（Vocabulary-Aware Conformal Prediction）是一种**词汇表感知的保形预测**方法，用于精确量化大语言模型的输出置信度。

### 核心思想

1. **词汇表过滤**：从 151,936 个 token 中筛选出 ~5,000 个有效 token
2. **温度缩放**：使用最优温度（通义千问为 8.0）锐化概率分布
3. **保形预测**：基于校准集计算阈值，生成预测集
4. **置信度计算**：预测集越小，置信度越高

### 效果

| 指标 | 值 |
|------|-----|
| 有效词汇表 | 5,047 / 151,936 (3.3%) |
| 置信度范围 | 0.0 - 1.0 |
| 人工确认触发阈值 | < 0.7 |

---

## 📊 技术栈

| 层级 | 技术 |
|------|------|
| 编排框架 | LangGraph |
| LLM | DeepSeek-V4.1-Flash（OpenAI 兼容模式） |
| 本地模型 | Qwen2-1.5B-Instruct（VACP logits 来源） |
| 向量数据库 | ChromaDB |
| 嵌入模型 | qwen3.7-text-embedding-flash（OpenAI 兼容模式） |
| 前端 | Streamlit |
| 持久化 | SQLite |
| 不确定性量化 | VACP (Conformal Prediction) |

---

## 🎯 核心代码示例

### 1. State 定义

```python
class MyAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    report_mode: bool
    user_id: Optional[str]
    vacp_confidence: Optional[float]
    last_logits: Optional[List[float]]
    needs_human_review: bool
    review_reason: Optional[str]
    review_status: Optional[str]
    pending_response: Optional[dict]
    session_id: Optional[str]
    thread_id: Optional[str]
```

### 2. 条件路由

```python
def should_continue(state):
    if state["messages"][-1].tool_calls:
        return "tools"
    return "vacp"

builder.add_conditional_edges("agent", should_continue)
builder.add_conditional_edges("vacp", after_vacp)
builder.add_conditional_edges("human_review", after_human_review)
```

### 3. 持久化

```python
conn = sqlite3.connect(str(db_path), check_same_thread=False)
saver = SqliteSaver(conn)
graph = builder.compile(checkpointer=saver)
```

---

## 📈 项目数据

| 指标 | 值 |
|------|-----|
| 代码行数 | ~3,000+ |
| Graph 节点数 | 6 |
| Graph 边数 | 7 |
| 工具数量 | 5 |
| 支持文件格式 | PDF, TXT, MD |
| 知识库文档数 | 326（示例） |

---

## 📄 许可证

MIT License

---

## 👤 作者

小吴 - 智能客服 Agent 项目