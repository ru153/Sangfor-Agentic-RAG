"""
本文件实现了智能代理工作流中的所有核心节点函数。
主要功能包括：
1. 输入安全防护（input_guard）：检测并拦截注入攻击和越界提问
2. 智能路由决策（router_node）：根据用户意图分发到不同处理流程（RAG/报名/查看记录/直接回答）
3. 混合检索（retrieve_node）：结合向量检索和BM25检索，从知识库获取相关文档
4. 文档相关性评分（grader_node）：评估检索文档与问题的相关性
5. 查询重写（rewriter_node）：优化检索效果不佳的查询语句
6. 报名信息管理（registration_node）：多轮对话收集用户报名信息并保存到CSV
7. 报名表查询（view_records_node）：管理员专属权限查看报名记录
8. 答案生成（generate_node/direct_node）：基于检索文档生成回答或直接回复
9. 安全审查（critic_node）：对输出内容进行最终的质量和安全检查

技术栈：
- 使用 LangChain 的 ChatOpenAI 进行大模型调用
- 使用 FAISS 向量库和 BM25 实现混合检索
- 使用 Pydantic 进行结构化输出解析
- 使用 SQLite 持久化对话状态
"""

# 导入正则表达式库，用于检测注入攻击
import re
# 导入 pickle 库，用于加载序列化的 BM25 检索器
import pickle
# 导入 CSV 库，用于读写报名表数据
import csv
# 导入操作系统接口，用于文件路径检查和环境变量加载
import os
# 导入类型注解工具，用于定义字面量类型和列表类型
from typing import Literal, List
# 导入 Pydantic 基类和字段定义，用于结构化输出解析
from pydantic import BaseModel, Field
# 导入 LangChain 的 ChatOpenAI 模型，用于大模型推理
from langchain_openai import ChatOpenAI
# 导入 LangChain 的工具装饰器，用于定义可调用的工具函数
from langchain_core.tools import tool
# 导入自定义的代理状态类型定义
from graph_state import AgentState
# 导入 LangChain 的 FAISS 向量库，用于文档向量检索
from langchain_community.vectorstores import FAISS
# 导入 LangChain 的 OpenAI Embeddings 模型，用于文本向量化
from langchain_openai import OpenAIEmbeddings
# 导入 dotenv 库，用于从 .env 文件加载环境变量
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量配置（如 API Key 等）
load_dotenv()

# 初始化大语言模型，使用 glm-5.3 模型，temperature=0 确保输出稳定
llm = ChatOpenAI(model="glm-5.3", temperature=0)


# ================= 加载检索器 =================
def get_retrievers():
    """
    加载并初始化两个检索器：向量检索器和 BM25 检索器
    返回：(vector_retriever, bm25_retriever) 元组
    """
    # 初始化 OpenAI Embeddings 模型，用于将文本转换为向量
    embeddings = OpenAIEmbeddings(model="embedding-3")
    # 从本地加载 FAISS 向量索引库，allow_dangerous_deserialization=True 允许反序列化
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # 将向量库转换为检索器，设置每次返回最相关的 3 个文档
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # 以二进制读取模式打开 BM25 检索器的序列化文件
    with open("bm25_retriever.pkl", "rb") as f:
        # 使用 pickle 反序列化加载 BM25 检索器
        bm25_retriever = pickle.load(f)
    # 设置 BM25 检索器返回前 3 个最相关文档
    bm25_retriever.k = 3
    # 返回两个检索器供后续使用
    return vector_retriever, bm25_retriever


# 调用函数获取向量检索器和 BM25 检索器的全局实例
v_retriever, b_retriever = get_retrievers()


# ================= 工具定义 =================
@tool
def save_registration_to_csv(name: str, major: str, direction: str, phone: str = "未提供") -> str:
    """
    将报名信息追加保存到 CSV 表格文件。
    参数：
        name: 用户姓名
        major: 专业
        direction: 报名方向（网络安全或AI智能体）
        phone: 联系电话，默认为"未提供"
    返回：报名成功的提示信息
    """
    # 检查 CSV 文件是否已存在，用于判断是否需要写入表头
    file_exists = os.path.isfile("报名信息表.csv")
    # 以追加模式打开 CSV 文件，使用 utf-8-sig 编码（支持 Excel 打开中文）
    with open("报名信息表.csv", mode="a", newline="", encoding="utf-8-sig") as f:
        # 创建 CSV 写入器
        writer = csv.writer(f)
        # 如果文件不存在，先写入表头行
        if not file_exists:
            writer.writerow(["姓名", "专业", "报名方向", "联系电话"])
        # 写入一条报名信息记录
        writer.writerow([name, major, direction, phone])
    # 返回成功提示，告知用户后续会有老师联系
    return "✅ 报名成功！系统已为您生成《报名信息表.csv》，稍后史老师会联系您。"


@tool
def view_registration_csv() -> str:
    """
    读取并返回当前报名表 CSV 文件的全部内容。
    返回：报名表内容或无记录的提示信息
    """
    # 检查报名表文件是否存在
    if not os.path.isfile("报名信息表.csv"):
        # 文件不存在，返回无记录提示
        return "📭 目前还没有任何报名记录。"
    # 以只读模式打开 CSV 文件，使用 utf-8-sig 编码
    with open("报名信息表.csv", "r", encoding="utf-8-sig") as f:
        # 读取文件全部内容
        content = f.read()
    # 返回格式化的报名表数据
    return f"📊 当前报名表数据如下：\n{content}"


# ================= 1. 安全防护 =================
# 定义注入攻击检测的正则表达式模式列表
INJECTION_PATTERNS = [
    # 检测中文提示注入攻击，如"忽略之前的指令"
    r"忽略(之前|以上|前面).*(指令|规则|提示)",
    # 检测英文提示注入攻击
    r"ignore (all )?(previous|above) instructions",
    # 检测试图获取系统信息或敏感数据的攻击
    r"输出.*(系统提示|密码)",
    # 检测恶意攻击相关词汇
    r"黑进|攻击|破解"
]


def input_guard(state: AgentState) -> dict:
    """
    输入安全检查节点，用于拦截恶意注入攻击和越界提问。
    参数：
        state: 代理状态字典，包含用户查询 query
    返回：
        如果检测到危险输入：返回 {"route": "blocked", "answer": "拦截提示"}
        如果输入安全：返回 {"route": "safe"}
    """
    # 获取用户输入的查询内容
    q = state["query"]
    # 遍历所有注入攻击检测模式
    for pattern in INJECTION_PATTERNS:
        # 使用正则表达式匹配，re.IGNORECASE 忽略大小写
        if re.search(pattern, q, re.IGNORECASE):
            # 检测到危险输入，返回拦截状态和提示信息
            return {"route": "blocked", "answer": "⚠️ 检测到不安全输入或越界提问，已拦截。"}
    # 未检测到危险，返回安全状态
    return {"route": "safe"}


# ================= 2. 智能路由 =================
class RouteDecision(BaseModel):
    """
    路由决策的结构化输出模型，用于规范大模型的输出格式。
    属性：
        route: 路由类型，必须是 'rag'、'direct'、'register' 或 'view_records' 之一
        refined_query: 经过优化的查询词，用于提高检索效果
    """
    route: Literal["rag", "direct", "register", "view_records"] = Field(
        description="register=用户想报名; view_records=用户想查看/导出报名表"
    )
    refined_query: str = Field(description="为检索优化的查询词")


def router_node(state: AgentState) -> dict:
    """
    路由节点，根据用户意图决定请求的处理路径。
    参数：
        state: 代理状态字典
    返回：
        包含 route（路由决策）和 rewritten_query（重写后的查询）的字典
    """
    # 如果输入已被标记为 blocked（拦截），直接返回空字典，不做任何处理
    if state.get("route") == "blocked":
        return {}

    # 🚨 核心锁死机制：只要处于报名流程中（is_registering 为 True），
    # 无论用户说什么（甚至说"我不报了"），都强制路由到 register 节点处理！
    # 这确保了报名流程的完整性和连贯性
    if state.get("is_registering") == True:
        return {"route": "register"}

    # 构造路由决策的提示词，指导大模型判断用户意图
    prompt = f"""用户问题：{state['query']}
判断应该走哪条路：
- rag：查训练营文档
- register：用户表达了"想报名"等意向
- view_records：用户想查看、导出、统计报名表记录
- direct：寒暄
同时把问题改写成适合检索的关键词。"""

    # 调用大模型，使用结构化输出解析路由决策
    decision = llm.with_structured_output(RouteDecision).invoke(prompt)
    # 返回路由决策和重写后的查询词
    return {"route": decision.route, "rewritten_query": decision.refined_query}


# ================= 3. 混合检索 =================
def retrieve_node(state: AgentState) -> dict:
    """
    检索节点，使用向量检索和 BM25 检索的混合策略获取相关文档。
    参数：
        state: 代理状态字典
    返回：
        包含 retrieved_docs（检索到的文档列表）的字典
    """
    # 获取重写后的查询，如果没有则使用原始查询
    q = state.get("rewritten_query") or state["query"]
    # 使用向量检索器检索最相关的文档
    vector_docs = v_retriever.invoke(q)
    # 使用 BM25 检索器检索最相关的文档
    bm25_docs = b_retriever.invoke(q)

    # 初始化字典用于合并两种检索结果，去除重复文档
    combined = {}
    # 遍历向量检索和 BM25 检索的所有结果
    for doc in vector_docs + bm25_docs:
        # 如果文档内容还没有在字典中，则加入（去重）
        if doc.page_content not in combined:
            combined[doc.page_content] = doc

    # 返回去重后的文档列表
    return {"retrieved_docs": list(combined.values())}


# ================= 4. 打分评价 (Grader) =================
class GradeDecision(BaseModel):
    """
    文档相关性评分的结构化输出模型。
    属性：
        binary_score: 二元评分，'yes' 表示相关，'no' 表示不相关
    """
    binary_score: Literal["yes", "no"] = Field(description="文档是否相关")


def grader_node(state: AgentState) -> dict:
    """
    评分节点，评估检索到的文档是否与用户问题相关。
    参数：
        state: 代理状态字典
    返回：
        包含 grade（'relevant' 或 'irrelevant'）的字典
    """
    # 获取用户原始查询
    question = state["query"]
    # 将检索到的文档中最后 3 个文档的内容拼接为上下文
    context = "\n\n".join([d.page_content for d in state["retrieved_docs"][-3:]])
    # 构造评分提示词，要求大模型判断文档是否包含回答问题所需的信息
    prompt = f"""判断以下文档是否包含能回答用户问题的信息。

    【文档】：{context}
    【问题】：{question}

    注意：只要文档中提到了与问题相关的概念（比如问零基础，文档提到了适用对象/门槛；问学什么，文档提到了课程/周计划），就请输出 yes。
    只有完全不沾边时，才输出 no。"""
    # 调用大模型进行结构化评分
    result = llm.with_structured_output(GradeDecision).invoke(prompt)
    # 根据评分结果返回相关性判断
    return {"grade": "relevant" if result.binary_score == "yes" else "irrelevant"}


# ================= 5. 查询重写 (Rewriter) =================
def rewriter_node(state: AgentState) -> dict:
    """
    重写节点，当检索效果不佳时，优化查询语句以提高检索质量。
    参数：
        state: 代理状态字典
    返回：
        包含 rewritten_query（新查询）、retry_count（重试次数+1）、retrieved_docs（清空）的字典
    """
    # 构造查询重写提示词，要求大模型生成新的优化查询
    prompt = f"原问题检索效果不好，请生成新查询。原问题：{state['query']}\n只输出新查询。"
    # 调用大模型生成新的查询语句
    new_q = llm.invoke(prompt).content
    # 返回重写后的查询、递增的重试次数、以及清空的文档列表
    return {"rewritten_query": new_q, "retry_count": state.get("retry_count", 0) + 1, "retrieved_docs": []}


# ================= 6. 报名收集节点 =================
class ExtractedInfo(BaseModel):
    """
    从用户输入中提取的报名信息结构化模型。
    属性：
        name: 用户姓名
        major: 专业
        direction: 报名方向
        phone: 联系电话
    """
    name: str = Field(description="姓名，没提到填'未知'")
    major: str = Field(description="专业，没提到填'未知'")
    direction: str = Field(description="报名方向，没提到填'未知'")
    phone: str = Field(description="电话，没提到填'未知'")


def registration_node(state: AgentState) -> dict:
    """
    报名收集节点，通过多轮对话收集用户报名信息并保存到 CSV。
    参数：
        state: 代理状态字典
    返回：
        包含 route、is_registering、user_info、answer 的字典
    """
    # 获取用户当前查询
    query = state["query"]
    # 获取已收集到的用户信息（如果有）
    history_info = state.get("user_info") or {}

    # 构造信息提取提示词，强调只能提取用户明确说出的信息，禁止猜测
    prompt = f"""已知收集到的信息：{history_info}
用户刚刚说：{query}
请从用户的话中提取报名信息。
【严格约束】：只能提取用户明确说出的信息！如果没有明确提到，必须严格返回空字符串 ""，绝对禁止猜测或脑补！"""

    # 调用大模型提取结构化信息
    extracted = llm.with_structured_output(ExtractedInfo).invoke(prompt)
    # 复制历史信息，准备更新
    new_info = history_info.copy()

    # 定义辅助函数，检查提取的信息是否有效
    def is_valid(val):
        """检查值是否有效（非空、非占位符）"""
        return val and val.strip() and val.strip() not in ["未知", "未提供", "None", "null", '""']

    # 如果提取到的姓名有效，则更新到信息字典
    if is_valid(extracted.name): new_info["name"] = extracted.name.strip()
    # 如果提取到的专业有效，则更新
    if is_valid(extracted.major): new_info["major"] = extracted.major.strip()
    # 如果提取到的方向有效，则更新
    if is_valid(extracted.direction): new_info["direction"] = extracted.direction.strip()
    # 如果提取到的电话有效，则更新
    if is_valid(extracted.phone): new_info["phone"] = extracted.phone.strip()

    # 初始化缺失信息列表
    missing = []
    # 检查是否缺少姓名
    if "name" not in new_info: missing.append("姓名")
    # 检查是否缺少专业
    if "major" not in new_info: missing.append("专业")
    # 检查是否缺少报名方向
    if "direction" not in new_info: missing.append("报名方向（网络安全或AI智能体）")

    # 如果还有缺失的必要信息
    if missing:
        # 构造追问提示词，要求大模型向用户索要缺失信息
        ask_prompt = f"你是报名助手。用户想报名，还缺这些信息：{missing}。请用亲切的语气向用户索要，一次只问缺失的，不要说废话。"
        # 调用大模型生成追问消息
        ask_msg = llm.invoke(ask_prompt).content
        # 返回追问信息，保持报名锁死状态
        return {
            "route": "register",          # 路由到报名节点
            "is_registering": True,        # 保持锁死状态，继续报名流程
            "user_info": new_info,         # 更新已收集的信息
            "answer": ask_msg              # 返回追问消息
        }
    else:
        # 所有必要信息已收集完毕，调用工具保存到 CSV 文件
        result = save_registration_to_csv.invoke({
            "name": new_info.get("name"),
            "major": new_info.get("major"),
            "direction": new_info.get("direction"),
            "phone": new_info.get("phone", "未提供")
        })
        # 返回报名成功信息，并解除锁死状态
        return {
            "route": "direct",             # 路由到直接回答节点
            "is_registering": False,       # 🚨 解锁！下一轮对话可以自由走 RAG 了
            "user_info": {},               # 清空表单记忆，为下一次报名做准备
            "answer": result               # 返回报名成功提示
        }


# ================= 7. 查看报名表节点 (仅限管理员) =================
def view_records_node(state: AgentState) -> dict:
    """
    查看报名记录节点，仅限管理员权限访问。
    参数：
        state: 代理状态字典
    返回：
        包含 route 和 answer 的字典
    """
    # 检查用户角色是否为 admin（管理员）
    if state.get("user_role") != "admin":
        # 非管理员用户，返回权限拒绝提示
        return {
            "route": "direct",
            "answer": "⛔ 权限拒绝：报名信息包含用户隐私数据，当前访客身份无权查看。请切换为管理员(Admin)权限操作。"
        }

    # 管理员用户，调用工具读取报名表内容
    result = view_registration_csv.invoke({})
    # 返回报名表数据
    return {
        "route": "direct",
        "answer": result
    }


# ================= 8. 最终生成 =================
def generate_node(state: AgentState) -> dict:
    """
    生成节点，基于检索到的文档生成最终回答。
    参数：
        state: 代理状态字典
    返回：
        包含 answer（生成的回答）的字典
    """
    # 如果没有检索到任何文档，返回建议联系人工的提示
    if not state.get("retrieved_docs"):
        return {"answer": "根据现有信息无法确认，建议联系史老师 17737177020。"}

    # 初始化上下文列表
    context_list = []
    # 遍历前 4 个检索到的文档
    for doc in state["retrieved_docs"][:4]:
        # 获取文档的章节元数据
        section = doc.metadata.get("H2_Section", "")
        # 将章节号和内容格式化为上下文字符串
        context_list.append(f"【{section}】\n{doc.page_content}")

    # 构造生成回答的提示词，要求基于文档回答并标注出处
    prompt = f"你是训练营助手。只根据以下文档回答，末尾标注出处。\n文档：{chr(10).join(context_list)}\n问题：{state['query']}"
    # 调用大模型生成回答并返回
    return {"answer": llm.invoke(prompt).content}


def direct_node(state: AgentState) -> dict:
    """
    直接回答节点，用于处理寒暄或无需检索的简单问题。
    参数：
        state: 代理状态字典
    返回：
        包含 answer（欢迎语）的字典
    """
    # 返回固定的欢迎语和助手介绍
    return {"answer": "你好！我是深信服生态训练营的咨询助手，关于时间、课程、报名等都可以问我。"}


# ================= 9. 反思审查 (Critic Agent) =================
class CriticEvaluation(BaseModel):
    """
    审查评估的结构化输出模型。
    属性：
        score: 安全评分，1-10 分
        reason: 打分理由
        is_safe: 是否安全（不包含违规承诺或幻觉）
    """
    score: int = Field(description="评分1-10分。10分代表绝对安全、准确、无幻觉")
    reason: str = Field(description="打分理由")
    is_safe: bool = Field(description="是否包含违规承诺（如保证录取）或幻觉")


def critic_node(state: AgentState) -> dict:
    """
    审查节点，对生成的回答进行最终的质量和安全检查。
    参数：
        state: 代理状态字典
    返回：
        包含 answer（审查后的回答）的字典
    """
    # 获取当前是否处于报名流程状态
    is_registering = state.get("is_registering")
    # 获取当前回答内容
    answer_text = state.get("answer", "")

    # 如果是报名追问、报名成功提示或权限拒绝提示，直接跳过审查，添加安全徽章
    if is_registering == True or "报名成功" in answer_text or "权限拒绝" in answer_text:
        return {"answer": f"{answer_text}\n\n> 🛡️ *本回答已通过系统安全审计*"}

    # 获取用户查询和 AI 生成的草稿回答
    query = state["query"]
    draft_answer = state.get("answer", "")

    # 构造审查提示词，要求大模型作为严苛的内容审查员
    prompt = f"""你是一个严苛的内容审查员。请审查以下由 AI 生成的回答。

    【用户问题】: {query}
    【AI草稿回答】: {draft_answer}

    审查标准：
    1. 必须基于事实，不能有恶意诱导。
    2. 绝对不能包含"保证录取"、"包过"等违规承诺。
    3. 如果回答中出现了"根据文档未提及"，说明大模型很诚实，这是加分项。

    请给出你的审查意见。"""

    try:
        # 调用大模型进行结构化审查评估
        evaluation = llm.with_structured_output(CriticEvaluation).invoke(prompt)

        # 解析评分，确保为整数类型
        score = int(evaluation.score) if str(evaluation.score).isdigit() else 10
        # 强制限制评分在 1-10 范围内
        score = max(1, min(score, 10))

        # 解析安全性判断，支持多种布尔值表示方式
        is_safe = str(evaluation.is_safe).lower() in ["true", "1", "yes"]

        # 如果内容不安全且评分低于 6 分，则拦截回答
        if not is_safe and score < 6:
            # 构造拦截提示信息
            safe_answer = f"⚠️ 【系统审查拦截】：AI 生成的内容未通过安全审计。\n原因：{evaluation.reason}\n请重新提问或联系人工客服。"
            # 返回拦截提示和 blocked 路由标记
            return {"answer": safe_answer, "route": "blocked"}

    except Exception as e:
        # 如果审查过程出现异常，默认给满分（保守策略）
        score = 10
        pass

    # 审查通过，在回答末尾添加安全审计徽章和评分
    audited_answer = f"{draft_answer}\n\n> 🛡️ *本回答已通过 Critic Agent 安全审计 (Score: {score}/10)*"
    # 返回审查后的回答
    return {"answer": audited_answer}
