# Chemistry Solver — Architecture Spec

A multi-agent chemistry problem solver spanning **GCSE → A-level → intro-university**,
built on the Anthropic Python SDK with Claude models. This document is the
contract the implementation follows.

## Runtime architecture

```
Problem prompt (+ optional level)
        │
        ▼
   ┌─────────┐   Plan (steps, target_quantity,      ┌──────────┐
   │ Planner │──── expected_units, difficulty) ────▶│  Solver  │
   └─────────┘                                      └────┬─────┘
   messages.parse → Plan          @beta_tool registry +  │ tool_runner loop,
                                  ends with submit_answer│
                                                         ▼
                                                    ┌──────────┐
                       approved ◀───────────────────│ Verifier │
                          │        Verdict          └──────────┘
                          ▼                    messages.parse → Verdict
                    SolveOutcome                (never sees expected answer)
                          │
   rejected & attempts < 3: verifier feedback appended as a NEW user
   message (system prompt + tool list stay byte-stable → prompt caching)
```

- **Orchestrator** (`agents/orchestrator.py`): plain-Python control flow —
  `plan → [solve → verify] × up to 3 attempts`. Accumulates `response.usage`
  per role into `UsageTotals` → USD cost from `config.PRICING`. Emits
  `SolveEvent`s (plan_ready, attempt_start, tool_call, answer_submitted,
  verdict, done, error) consumed by the CLI renderer. A `stop_reason ==
  "refusal"` or a tool-runner exception counts as a failed attempt, not a crash.
- **Planner** (`agents/planner.py`): `client.messages.parse(...,
  output_format=Plan)`. Assesses difficulty; the orchestrator maps difficulty →
  `output_config.effort` for the solver (`gcse: medium`, `alevel: high`,
  `university: xhigh`).
- **Solver** (`agents/solver.py`): `client.beta.messages.tool_runner(...)` over
  `@beta_tool`-wrapped pure functions. Must end with a
  `submit_answer(value, units, formatted_answer, reasoning_summary)` call;
  fallback is one `messages.parse` extraction over the last text. The system
  prompt forbids mental arithmetic — every calculation goes through a tool.
- **Verifier** (`agents/verifier.py`): checks *without* the expected answer —
  units vs plan, dimensional consistency of the tool chain, magnitude
  plausibility, plan-step coverage. Approval/feedback only.
- **Comparator** (`verification/comparator.py`): deterministic, used by evals
  and tests only. pint quantity parsing (`500 mL` matches `0.50 L`),
  `math.isclose` with 1% relative tolerance, unicode/superscript
  normalization, alias table (`M → mol/L`, `dm³ → L`, `%`), formula-string
  fallback for answers like `CH2O`.

## API usage rules (Claude, current reference)

- Default model for every role: **`claude-opus-5`**. Solver may be pointed at
  `claude-sonnet-5` / `claude-haiku-4-5` for eval comparisons.
- **Never** send `temperature` / `top_p` / `top_k` (400 on opus-5). No
  assistant prefills.
- Thinking is on by default on opus-5 — the `thinking` param is omitted
  entirely. Depth is controlled with `output_config={"effort": ...}`.
  `max_tokens` caps thinking + text together, so budgets are generous
  (16000 solver, 4096 planner/verifier).
- Sync `anthropic.Anthropic()` client, key from `ANTHROPIC_API_KEY`. No asyncio.
- `stop_reason == "refusal"` is handled before reading `content`.
- Optional hardening (not implemented): server-side `fallbacks: "default"`
  beta for automatic refusal rerouting.

## Toolkit (pure functions, no SDK imports)

`helper/tools.py` keeps the original 14 GCSE tools (exact output strings —
existing tests must keep passing) with fixed `-> str` annotations and an
AST-safe `calc` (no `eval`). New domains live in `chemistry_agent/toolkit/`:

| Module | Tools |
|---|---|
| `gases.py` | `ideal_gas` (PV=nRT), `gas_volume_rtp` (n×24 dm³), `combined_gas_law`, `partial_pressures` |
| `energetics.py` | `hess_law`, `bond_energy_enthalpy`, `gibbs_free_energy` (ΔG=ΔH−TΔS) |
| `equilibrium.py` | `equilibrium_kc`, `equilibrium_kp` (parallel lists for clean schema gen) |
| `kinetics.py` | `rate_law_order`, `arrhenius` (k=Ae^(−Ea/RT)), `half_life_first_order` |
| `acids_bases.py` | `weak_acid_pH` (Ka), `buffer_pH` (Henderson–Hasselbalch), `pOH`/Kw conversions, `ksp_solubility` |
| `electrochem.py` | `cell_potential`, `nernst_equation`, `electrolysis_mass` (m=MIt/nF) |
| `reactions.py` | `balance_equation` (integer nullspace over `Fraction` — deterministic, no LLM), `limiting_reagent`, `atom_economy` |
| `mathtools.py` | `convert_units` (pint), `solve_equation` (sympy, single unknown), safe `calc` |

Conventions: pure, typed, numeric-in → formatted-string-out, Google-style
docstrings (the tool runner generates schemas from signatures + docstrings).

## Problem bank & evals

- `ProblemState` gains `level` (`gcse`/`alevel`/`university`, default `gcse`),
  `topic`, and `tolerance`; `task` relaxes from a `Literal` to `Optional[str]`.
- `helper/problems.py`: existing 15 problems tagged `gcse` with params
  normalized; ~15 new problems across `alevel` and `university` tiers.
- `evals/harness.py` → `EvalReport`: accuracy overall + per-level + per-topic,
  cost, tokens, per-tool histogram, retry rate. `evals/report.py`: rich tables,
  JSON artifact, `compare(a, b)` for cross-model runs.

## CLI

`chem solve "..." [--level] [--show-work]` — live plan panel, tool trace,
verifier verdict, cost footer. `chem eval [--limit] [--level] [--label]
[--compare a.json b.json]`.

## Testing rules

Everything is green **without** `ANTHROPIC_API_KEY`:
- toolkit numerics vs textbook values; balancer round-trips.
- planner/verifier with `MagicMock` client (`parse.return_value.parsed_output`).
- solver with mocked tool runner; orchestrator with stub agents (constructor
  injection); eval harness with a stub orchestrator.
- legacy `test_tools`, `test_tools_vs_expected`, `test_problem_state` unchanged.
