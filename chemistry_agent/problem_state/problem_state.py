from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

# ProblemState declaration
class ProblemState(BaseModel):
    prompt: Optional[str] = None
    task: Optional[str] = Field(
        default=None,
        description="(optional) which tool to start with"
    )
    level: Literal["gcse", "alevel", "university"] = "gcse"
    topic: str = "general"
    tolerance: float = 0.01  # relative tolerance for comparison
    params: Dict[str, Any]
    expected: str
