FUNCTIONS = [
    {
      "name": "convert_units",
      "description": "Convert between units (e.g. grams to kilograms or °C to °F).",
      "parameters": {
        "type": "object",
        "properties": {
          "value":      {"type": "number"},
          "from_unit":  {"type": "string"},
          "to_unit":    {"type": "string"},
        },
        "required": ["value", "from_unit", "to_unit"],
      },
    },
    {
      "name": "lookup_molar_mass",
      "description": "Look up the molar mass of a chemical formula.",
      "parameters": {
        "type": "object",
        "properties": {
          "formula": {"type": "string"},
        },
        "required": ["formula"],
      },
    },
    {
      "name": "compute_moles",
      "description": "Compute moles of product from given mass of reactant and stoichiometry.",
      "parameters": {
        "type": "object",
        "properties": {
          "mass":           {"type": "number"},
          "formula":        {"type": "string"},
          "stoich_react":   {"type": "integer"},
          "stoich_prod":    {"type": "integer"},
        },
        "required": ["mass", "formula"],
      },
    },
    {
      "name": "mass_from_moles",
      "description": "Convert moles of a substance to mass in grams.",
      "parameters": {
        "type": "object",
        "properties": {
          "moles":       {"type": "number"},
          "molar_mass":  {"type": "number"},
        },
        "required": ["moles", "molar_mass"],
      },
    },
    {
      "name": "mix_concentration",
      "description": "Compute final concentration when mixing two solutions.",
      "parameters": {
        "type": "object",
        "properties": {
          "C1": {"type": "number"},
          "V1": {"type": "number"},
          "C2": {"type": "number"},
          "V2": {"type": "number"},
        },
        "required": ["C1", "V1", "C2", "V2"],
      },
    },
    {
      "name": "dilution",
      "description": "Compute final volume from a dilution C1·V1 = C2·V2.",
      "parameters": {
        "type": "object",
        "properties": {
          "initial_conc": {"type": "number"},
          "initial_vol":  {"type": "number"},
          "final_conc":   {"type": "number"},
        },
        "required": ["initial_conc", "initial_vol", "final_conc"],
      },
    },
]
