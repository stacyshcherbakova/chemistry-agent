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
    # None: score at the precision ``expected`` is written to (see
    # verification.comparator). A float switches to a relative tolerance.
    tolerance: Optional[float] = None
    params: Dict[str, Any]
    expected: str
