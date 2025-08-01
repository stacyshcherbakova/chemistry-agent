from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

class ProblemState(BaseModel):
    prompt: Optional[str] = None
    task:     Optional[Literal[
                  "molar_mass",
                  "compute_moles",
                  "mass_from_moles",
                  "mix_concentration",
                  "dilution",
              ]] = Field(
                  default=None,
                  description="(optional) which tool to start with"
              )
    params: Dict[str, Any]
    expected: str 
    
examples: list[ProblemState] = [
    # # 1. Unit conversion: mass
    # ProblemState(
    #     task="convert_units",
    #     params={"value": 1500, "from_unit": "mg", "to_unit": "g"},
    #     expected="1.500 gram",
    # ),

    # # 2. Unit conversion: temperature
    # ProblemState(
    #     task="convert_units",
    #     params={"value": 32, "from_unit": "degF", "to_unit": "degC"},
    #     expected="0.000 °C",
    # ),

    # 3. Molar mass lookup
    ProblemState(
        task="molar_mass",
        params={"formula": "H2O"},
        expected="18.015 g/mol",
    ),

    # 4. Compute moles from mass + formula
    ProblemState(
        task="compute_moles",
        params={"mass": 18.015, "formula": "H2O"},
        expected="1.0000 mol",
    ),

    # 5. Mass from moles
    ProblemState(
        task="mass_from_moles",
        params={"moles": 0.500, "molar_mass": 44.01},
        expected="22.005 g",
    ),

    # 6. Mix concentration of two solutions
    ProblemState(
        task="mix_concentration",
        params={"C1": 0.10, "V1": 0.050, "C2": 0.20, "V2": 0.100},
        expected="0.167 M",
    ),

    # 7. Dilution calculation (C1V1 = C2V2)
    ProblemState(
        task="dilution",
        params={"initial_conc": 0.50, "initial_vol": 0.100, "final_conc": 0.10},
        expected="0.500 L",
    ),
]
