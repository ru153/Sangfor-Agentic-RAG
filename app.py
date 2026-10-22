"""
本文件实现了基于 FastAPI 的智能代理 Web API 服务。
主要功能包括：
1. 提供 RESTful API 接口，支持用户与智能代理进行 HTTP 交互
2. 实现请求限流机制（10次/分钟），防止 API 滥用
3. 支持多用户会话管理，通过 thread_id 区分不同用户的对话上下文
4. 实现基于角色的访问控制（RBAC），区分访客和管理员权限
5. 提供完整的执行轨迹追踪，记录代理工作流的每个节点状态
6. 返回结构化的响应数据，包括回答、路由、执行轨迹和检索上下文

API 端点：
- GET /：返回前端交互页面（index.html）
- POST /chat：处理用户对话请求，返回代理回答和执行详情

技术栈：
- FastAPI：高性能异步 Web 框架
- SlowAPI：请求限流中间件
- Pydantic：请求/响应数据验证和序列化
- LangGraph Agent：智能代理工作流引擎
"""

# 从 FastAPI 导入核心类：应用实例、HTTP 异常、请求对象
from fastapi import FastAPI, HTTPException, Request
# 从 FastAPI 导入 HTML 响应类，用于返回网页内容
from fastapi.responses import HTMLResponse
# 从 Pydantic 导入基础模型和字段定义，用于数据验证
from pydantic import BaseModel, Field
# 导入类型注解工具，支持列表、字典、任意类型
from typing import List, Dict, Any
# 从 slowapi 导入限流器和限流异常处理器
from slowapi import Limiter, _rate_limit_exceeded_handler
# 从 slowapi 导入获取远程地址的工具函数，用于限流键值
from slowapi.util import get_remote_address
# 从 slowapi 导入限流异常类
from slowapi.errors import RateLimitExceeded

# 从自定义模块导入 FastAPI 应用实例和智能代理对象
# custom_docs 模块已经初始化了 app 和 agent
from custom_docs import app, agent

# ================= 初始化限流器 =================
# 创建限流器实例，使用客户端 IP 地址作为限流的键值
# 这意味着每个 IP 地址都会独立计算请求频率
limiter = Limiter(key_func=get_remote_address)
# 将限流器绑定到 FastAPI 应用的状态对象
app.state.limiter = limiter
# 注册限流异常处理器，当请求超过限制时自动返回 429 错误
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class ChatRequest(BaseModel):
    """
    对话请求的数据模型，定义了客户端发送对话请求时必须提供的字段。
    属性：
        query: 用户输入的提问内容
        thread_id: 会话 ID，用于标识一个对话线程，支持多轮对话
        user_role: 用户角色，guest=访客，admin=管理员，影响权限控制
    """
    query: str = Field(
        ...,  # ... 表示该字段为必填项
        description="用户的提问内容",
        json_schema_extra={"example": "我想报名"}  # 提供示例值，用于 API 文档
    )
    thread_id: str = Field(
        "default_user",  # 默认值为 "default_user"
        description="会话ID",
        json_schema_extra={"example": "user_001"}
    )
    user_role: str = Field(
        "guest",  # 默认角色为访客
        description="用户角色，guest=访客, admin=管理员",
        json_schema_extra={"example": "admin"}
    )


class TraceStep(BaseModel):
    """
    执行轨迹步骤的数据模型，记录代理工作流中单个节点的执行状态。
    属性：
        node_name: 节点名称（如 guard、router、retrieve 等）
        state_update: 该节点执行后的状态更新字典
    """
    node_name: str
    state_update: Dict[str, Any]


class ChatResponse(BaseModel):
    """
    对话响应的数据模型，定义了返回给客户端的完整响应结构。
    属性：
        answer: 代理生成的最终回答文本
        route: 路由决策（如 rag、register、direct、blocked 等）
        execution_trace: 执行轨迹列表，记录所有节点的执行过程
        retrieved_context: 检索到的相关文档内容（前 200 字摘要）
    """
    answer: str
    route: str
    execution_trace: List[TraceStep]
    retrieved_context: List[str]


@app.get("/", response_class=HTMLResponse)
async def read_index():
    """
    根路径 GET 请求处理函数，返回前端交互页面。
    返回：index.html 文件的内容
    """
    # 以只读模式打开 index.html 文件，使用 UTF-8 编码
    with open("index.html", "r", encoding="utf-8") as f:
        # 读取文件内容并作为 HTML 响应返回
        return f.read()


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")  # 🚨 限流：每个 IP 每分钟最多 10 次请求
def chat_endpoint(request: Request, req: ChatRequest):
    """
    对话 POST 请求处理函数，处理用户与代理的交互。
    参数：
        request: FastAPI 请求对象（限流器需要）
        req: 对话请求数据模型
    返回：
        ChatResponse 对象，包含回答、路由、执行轨迹和上下文
    """
    # 构造代理配置字典，设置线程 ID 用于状态持久化
    # 同一个 thread_id 下的多次请求会共享对话历史
    config = {"configurable": {"thread_id": req.thread_id}}

    # 🚨 构造代理的初始状态字典，传入用户角色信息
    initial_state = {
        "query": req.query,          # 用户查询内容
        "retry_count": 0,            # 重试计数器，初始化为 0
        "user_role": req.user_role   # 用户角色，用于权限控制
    }

    # 初始化执行轨迹列表，用于记录每个节点的执行状态
    trace = []
    # 初始化最终状态字典，用于汇总所有节点的输出
    final_state = {}

    try:
        # 使用 agent.stream() 流式执行代理工作流
        # stream() 会逐个节点地执行，并返回每个节点的输出事件
        for event in agent.stream(initial_state, config):
            # 遍历事件字典，获取节点名称和节点状态
            for node_name, node_state in event.items():
                # 检查节点状态是否有效且为字典类型
                if node_state and isinstance(node_state, dict):
                    # 过滤掉 retrieved_docs 字段（文档对象无法序列化）
                    # 只保留可序列化的状态信息用于追踪
                    safe_state = {k: v for k, v in node_state.items() if k != "retrieved_docs"}
                    # 将节点名称和安全状态添加到执行轨迹
                    trace.append(TraceStep(node_name=node_name, state_update=safe_state))
                # 如果节点状态有效，则更新到最终状态字典
                if node_state:
                    final_state.update(node_state)
    except Exception as e:
        # 捕获代理运行过程中的任何异常，返回 500 错误
        raise HTTPException(status_code=500, detail=f"Agent 运行异常: {str(e)}")

    # 从最终状态中获取检索到的文档列表
    retrieved_docs = final_state.get("retrieved_docs", [])
    # 提取前 3 个文档的内容摘要（每个文档前 200 字 + 省略号）
    # 用于返回给客户端展示检索上下文
    context_texts = [doc.page_content[:200] + "..." for doc in retrieved_docs[:3]]

    # 构造并返回 ChatResponse 对象
    return ChatResponse(
        answer=final_state.get("answer", "抱歉，未能生成回答。"),  # 获取最终回答，如果没有则返回默认提示
        route=final_state.get("route", "unknown"),                 # 获取路由决策
        execution_trace=trace,                                     # 获取完整执行轨迹
        retrieved_context=context_texts                            # 获取检索上下文摘要
    )


# Python 程序入口，当直接运行本文件时启动 Web 服务器
if __name__ == "__main__":
    # 导入 uvicorn ASGI 服务器
    import uvicorn

    # 启动 uvicorn 服务器
    # host="0.0.0.0" 表示监听所有网络接口（允许外部访问）
    # port=8000 表示服务运行在 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)
