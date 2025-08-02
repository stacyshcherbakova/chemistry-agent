from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

# ProblemState declaration
class ProblemState(BaseModel):
    prompt: Optional[str] = None
    task:     Optional[Literal[
                  "molar_mass",
                  "compute_moles",
                  "mass_from_moles",
                  "mix_concentration",
                  "dilution",
                  "percent_yield",
                  "energy_change",
                  "pH_from_H",
                  "calculate_density",
                  "empirical_formula",
                  "titration"
              ]] = Field(
                  default=None,
                  description="(optional) which tool to start with"
              )
    params: Dict[str, Any]
    expected: str 