"""
本文件实现了基于 LangGraph 的智能代理（Agent）工作流构建。
主要功能包括：
1. 构建一个多阶段的智能代理图（StateGraph），用于处理用户的输入请求
2. 实现工作流程：输入检查 -> 路由分发 -> 知识检索/业务办理/记录查询 -> 答案生成 -> 安全审查
3. 支持多种路由分支：RAG检索增强生成、报名注册、查看记录、直接回答
4. 集成 SQLite 检查点机制，支持对话状态持久化
5. 实现答案质量评估和自动重写机制，提升回答质量
6. 添加安全审查节点（critic），确保输出的合规性
"""

# 导入 SQLite 数据库连接库，用于状态持久化
import sqlite3
# 从 langgraph 导入状态图、起始节点、结束节点，用于构建工作流
from langgraph.graph import StateGraph, START, END
# 从 langgraph 导入 SQLite 检查点保存器，用于保存对话状态
from langgraph.checkpoint.sqlite import SqliteSaver
# 从自定义模块导入代理状态定义
from graph_state import AgentState
# 从自定义模块导入所有工作流节点函数
from graph_nodes import (
    input_guard,          # 输入检查节点，用于拦截非法输入
    router_node,          # 路由节点，决定请求的处理路径
    retrieve_node,        # 检索节点，从知识库检索相关文档
    grader_node,          # 评分节点，评估检索文档的相关性
    rewriter_node,        # 重写节点，优化查询语句
    generate_node,        # 生成节点，生成最终回答
    direct_node,          # 直接回答节点，处理无需检索的问题
    registration_node,    # 报名注册节点，处理报名业务
    view_records_node,    # 查看记录节点，查询用户记录
    critic_node           # 审查节点，对输出进行安全审查
)


def build_agent():
    """
    构建并编译智能代理工作流图
    返回：编译后的代理对象，可直接调用
    """
    # 创建状态图构建器，传入状态类型定义
    builder = StateGraph(AgentState)

    # ==================== 挂载所有节点到图中 ====================
    # 添加输入安全检查节点
    builder.add_node("guard", input_guard)
    # 添加路由决策节点
    builder.add_node("router", router_node)
    # 添加知识检索节点
    builder.add_node("retrieve", retrieve_node)
    # 添加文档相关性评分节点
    builder.add_node("grader", grader_node)
    # 添加查询重写节点
    builder.add_node("rewriter", rewriter_node)
    # 添加答案生成节点
    builder.add_node("generate", generate_node)
    # 添加直接回答节点
    builder.add_node("direct", direct_node)
    # 添加报名注册节点
    builder.add_node("register", registration_node)
    # 添加查看记录节点
    builder.add_node("view_records", view_records_node)
    # 添加安全审查节点
    builder.add_node("critic", critic_node)

    # ==================== 设置工作流的边（连接关系） ====================
    # 从起始节点连接到输入检查节点
    builder.add_edge(START, "guard")

    # 设置输入检查节点的条件边：根据检查结果决定是结束还是继续路由
    # 如果 route 为 "blocked"，则流向 END 结束；否则流向 router 继续处理
    builder.add_conditional_edges(
        "guard", lambda s: "end" if s.get("route") == "blocked" else "router",
        {"end": END, "router": "router"}
    )

    # 设置路由节点的条件边：根据路由结果分发到不同的处理节点
    # 如果 route 为 "rag"，进入知识检索流程
    # 如果 route 为 "register"，进入报名注册流程
    # 如果 route 为 "view_records"，进入查看记录流程
    # 否则进入直接回答流程
    builder.add_conditional_edges(
        "router",
        lambda s: (
            "retrieve" if s["route"] == "rag" else
            ("register" if s["route"] == "register" else
            ("view_records" if s["route"] == "view_records" else "direct"))
        ),
        {
            "retrieve": "retrieve",      # 路由到检索节点
            "register": "register",      # 路由到注册节点
            "view_records": "view_records",  # 路由到查看记录节点
            "direct": "direct"           # 路由到直接回答节点
        }
    )

    # 检索完成后，连接到评分节点评估检索质量
    builder.add_edge("retrieve", "grader")

    # 定义评分后的条件判断函数
    def should_rewrite(state: AgentState):
        """
        根据评分结果决定是否重写查询
        如果评分为 "relevant"，说明检索到相关文档，进入生成阶段
        如果重试次数 >= 1，为避免无限循环，直接进入生成阶段
        否则进入重写阶段，优化查询后重新检索
        """
        if state["grade"] == "relevant": return "generate"
        if state.get("retry_count", 0) >= 1: return "generate"
        return "rewrite"

    # 设置评分节点的条件边：根据评分结果决定生成答案还是重写查询
    builder.add_conditional_edges("grader", should_rewrite, {"generate": "generate", "rewrite": "rewriter"})

    # 重写查询后，重新进入检索流程，形成闭环
    builder.add_edge("rewriter", "retrieve")

    # ==================== 所有回答路径最终都汇聚到审查节点 ====================
    # 生成节点完成后，进入安全审查
    builder.add_edge("generate", "critic")
    # 直接回答完成后，进入安全审查
    builder.add_edge("direct", "critic")
    # 报名注册完成后，进入安全审查
    builder.add_edge("register", "critic")
    # 查看记录完成后，进入安全审查
    builder.add_edge("view_records", "critic")

    # 审查节点完成后，工作流结束
    builder.add_edge("critic", END)

    # ==================== 配置状态持久化 ====================
    # 连接 SQLite 数据库，用于保存对话状态和检查点
    # check_same_thread=False 允许多线程访问
    # isolation_level=None 禁用自动事务管理，提高性能
    conn = sqlite3.connect("agent_memory.db", check_same_thread=False, isolation_level=None)
    # 创建 SQLite 检查点保存器，用于持久化对话状态
    checkpointer = SqliteSaver(conn)

    # 编译图并配置检查点器，返回可执行的代理对象
    return builder.compile(checkpointer=checkpointer)
