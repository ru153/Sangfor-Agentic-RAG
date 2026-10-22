from typing import TypedDict, Annotated, List, Dict
import operator

class AgentState(TypedDict):
    query: str
    rewritten_query: str
    route: str
    retrieved_docs: Annotated[List, operator.add]
    grade: str
    retry_count: int
    answer: str
    user_info: Dict[str, str]
    is_registering: bool
    user_role: str