# Chemistry Agent

[![tests](https://github.com/stacyshcherbakova/chemistry-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/stacyshcherbakova/chemistry-agent/actions/workflows/tests.yml)

A multi-agent chemistry problem solver spanning **GCSE → A-level → intro-university**,
built on Claude with the Anthropic Python SDK.

Instead of one model guessing at an answer, a small team works the problem: a
**Planner** decomposes it, a **Solver** executes every calculation through real
tools, and a **Verifier** independently audits the result before it is shown —
without ever seeing the expected answer. If the verifier rejects, its feedback
goes back to the solver and the loop runs again.

```
Problem (+ optional level)
        │
        ▼
   ┌─────────┐   Plan: steps, target quantity,     ┌──────────┐
   │ Planner │──── expected units, difficulty ────▶│  Solver  │
   └─────────┘                                     └────┬─────┘
                          39 chemistry tools,           │
                          ends with submit_answer       ▼
                                                   ┌──────────┐
                      approved ◀───────────────────│ Verifier │
                         │        Verdict          └──────────┘
                         ▼          units · dimensions · magnitude · coverage
                   Answer + worked solution + cost
                         ▲
   rejected & attempts remain: feedback becomes a new user message
   (system prompt + tool list stay byte-stable → prompt caching)
```

## What it can do

- **Three difficulty tiers.** Difficulty is assessed by the planner and mapped to
  model reasoning effort, so a molar-mass lookup stays cheap while a Nernst
  equation or a Ksp common-ion problem gets deeper reasoning.
- **39 tools, no mental arithmetic.** The solver is forbidden from doing sums in
  its head; every number comes from a pure Python function.

| Domain | Tools |
|---|---|
| Stoichiometry & solutions | molar mass, moles, mass↔moles, concentration, dilution, mixing, titration, percent yield, density, empirical formula |
| Gases | ideal gas (PV=nRT), molar volume at RTP, combined gas law, partial pressures |
| Energetics | Hess's law, bond enthalpies, Gibbs free energy, q=mcΔT |
| Equilibrium | Kc, Kp |
| Kinetics | rate-law orders, Arrhenius, two-temperature Arrhenius, first-order half-life |
| Acids & bases | weak-acid pH, buffer pH, pH/pOH/Kw conversions, Ksp solubility |
| Electrochemistry | cell potential, Nernst equation, electrolysis mass |
| Reactions | equation balancer (exact integer nullspace), limiting reagent, atom economy |
| Math | unit conversion (pint), symbolic solve (sympy), safe arithmetic |

- **Precision-aware scoring.** The old exact-string match is gone. A
  deterministic comparator parses quantities with pint and grades at the
  precision the answer key is written to, the way an examiner rounds: against
  `0.17 M`, `0.1667 M` is right, `0.2 M` is not, and `500 mL` matches `0.50 L`.
  Formulas compare after normalising subscripts, so `CH₂O` matches `CH2O`.
  Answers are graded on the solver's structured `value` and `units` as well as
  its text, so a correct number is never failed for the prose around it.
- **Says when a question can't be answered.** If data is missing, contradictory
  or physically impossible, the solver flags `insufficient_data` instead of
  inventing a number, and the verifier checks that claim.
- **Benchmarking.** A standard 30-problem bank and a hard bank of multi-step and
  deliberately unanswerable problems, tagged by level and topic, with per-level
  and per-topic accuracy, cost, tokens, retries, tool usage and latency. Runs can
  switch the verifier off to measure what it is worth.

## Results

Run on 23 October 2026 over the 30-problem bank (15 GCSE, 7 A-level, 8 university).
The planner and verifier are Claude Opus 5 in both configurations; only the solver
changes.

| Solver | Accuracy | Cost per problem | Cost for 30 | Verifier retries |
|---|---|---|---|---|
| Claude Opus 5 | 30/30 | $0.133 | $3.98 | 0 |
| Claude Sonnet 5 | 30/30 | $0.089 | $2.66 | 1 |

| Level | Opus solver, cost per problem | Sonnet solver, cost per problem | Tool calls per problem (Opus / Sonnet) |
|---|---|---|---|
| GCSE | $0.122 | $0.081 | 3.0 / 2.7 |
| A-level | $0.135 | $0.088 | 3.7 / 3.3 |
| University | $0.151 | $0.102 | 5.4 / 4.5 |

**What this shows:** on this bank a Sonnet solver matches Opus at about two-thirds
of the cost, so it is the better default here. **What it does not show:** both
configurations score 100%, so the bank is too easy to separate them on accuracy,
which is why the hard bank below exists.

Most of the cost is input tokens (about 63k per problem, against about 1k output),
because the 39 tool definitions travel with every call. Cache reads are priced at
10% of the input rate, which is what keeps that affordable.

### Hard bank: multi-step and unanswerable problems

`chem eval --bank hard` runs 15 harder problems: 10 multi-step (buffer pH after a
second addition, Born–Haber cycles, ΔG → K at a new temperature, back-titrations,
combustion analysis to a molecular formula) and 5 that cannot be answered as
written (a missing Ka, a negative concentration, contradictory data, and so on),
where the right answer is to set `insufficient_data` rather than invent a number.
Every key was worked by hand in comments in `problems_hard.py`. Run on 23 October 2026:

| Configuration | Multi-step | Unanswerable | Total | Cost per problem | Median / p95 time |
|---|---|---|---|---|---|
| Opus solver + verifier | 10/10 | 4/5 | 14/15 | $0.180 | 28 s / 41 s |
| Sonnet solver + verifier | 10/10 | 4/5 | 14/15 | $0.129 | 27 s / 45 s |
| Sonnet solver, no verifier | 10/10 | 4/5 | 14/15 | $0.115 | 23 s / 41 s |

**What this shows:** Sonnet still matches Opus, now on problems where the
calculation is the hard part, at about 70% of the cost. All three configurations
spotted four of the five unanswerable problems instead of making up data.

**What the verifier is worth here: nothing measurable.** It never rejected an
answer on this bank, so it added about 12% to the cost and 4 seconds to the
median time without changing a result. On the standard bank it caught one wrong
answer in 60. I am keeping it on by default because a wrong number costs more
than $0.014, but this run does not prove it pays for itself.

**Caveats:** 15 problems is a small sample, and one answer either way moves a
score by 7 points. The one miss, shared by all three, is a judgement call: a gas
problem gives n, V, T *and* p, which disagree by about 20% under pV = nRT. All
three ignored n and used p₁/T₁ = p₂/T₂ to get 188 kPa, which an exam marker
might well accept. The key says the data is contradictory, and the verifier was
told to reject answers that ignore a contradiction, but it didn't.

### What the first runs got wrong, and the fixes

The first smoke run scored 3/5. Every failure turned out to be in the evaluation,
not the agent, so the eval was fixed before trusting any number:

- **The grader failed correct answers.** It choked on `≈0.17 M (0.025 mol / 0.15 L)`,
  `0.50 L (500 mL)` and a typographic minus in `−74.80 kJ/mol`. Grading now also
  uses the solver's structured `value` and `units` from `submit_answer`, and the
  comparator ignores approximation signs, bracketed working and Unicode minus signs.
- **The answer key had the wrong precision.** Ten keys were written to more or fewer
  figures than the question justifies; one asked for 2 significant figures but keyed
  3. Scoring is now precision-aware, so every key was audited and rewritten to the
  precision a marker would require.
- **Three questions were defective.** One gave away its answer in a format example,
  one implied a 125% yield, and one asked for "2 significant figures" on a pH keyed
  to 2 decimal places. Templated questions also printed `0.050` as `0.05`, dropping
  significant figures; all are fixed.
- **A dropped network connection was scored as wrong answers.** Failed calls are now
  reported separately (`n_errors`) and re-run, not counted against the agent.
- **The first hard-bank run hung for an hour.** One tool call spent 55 minutes of
  CPU on exact big-integer arithmetic. `calc` now works in floats only, so huge
  powers fail at once, and `solve_equation` falls back to numeric polynomial
  roots if symbolic solving takes more than 5 seconds. The re-run took 7 minutes.
- **Two more grader false negatives on the hard bank:** a correct `0.000317` with
  units `dimensionless`, and a correct formula followed by bracketed working,
  `C4H8O2 (empirical formula C2H4O, …)`. Both are fixed and covered by tests, and
  every saved report was re-scored with `chem rescore` (no change on the standard bank).

Agent behaviour worth knowing about:

- The verifier rejected one Sonnet answer (the limiting-reagent yield problem); the
  solver corrected it on the second attempt.
- In one Sonnet run the verifier call itself failed, so that answer was graded
  correct but never verified.
- The solver kept passing `log10(...)` to the arithmetic tool, which only accepted
  operators. `calc` now accepts `log10`, `ln`, `exp` and `sqrt`.
- In one hard-bank run the solver correctly said a problem could not be solved but
  wrote that as text without setting the flag. The verifier now rejects that mismatch.

## Setup

```bash
git clone https://github.com/stacyshcherbakova/chemistry-agent.git
cd chemistry-agent
poetry install
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Solve one problem
poetry run chem solve "What is the molar mass of H2O?"

# Harder problem, with the full worked solution
poetry run chem solve --level university --show-work \
  "Calculate the cell potential of a Zn/Cu cell at 298 K with [Zn2+]=0.010 M and [Cu2+]=1.0 M"

# Benchmark
poetry run chem eval --limit 10 --label baseline
poetry run chem eval --level university --label uni-only

# Compare two runs (e.g. opus vs sonnet solver)
poetry run chem eval --solver-model claude-sonnet-5 --label sonnet
poetry run chem compare eval_baseline.json eval_sonnet.json

# Hard bank, and the same run without the verifier
poetry run chem eval --bank hard --label hard
poetry run chem eval --bank hard --no-verifier --label hard-noverify
poetry run chem compare eval_hard.json eval_hard-noverify.json

# Re-grade a saved run after correcting the answer key (no API calls)
poetry run chem rescore eval_baseline.json
```

Library use:

```python
from chemistry_agent.agents.orchestrator import Orchestrator

outcome = Orchestrator().solve("How many moles are in 18.015 g of H2O?")
print(outcome.answer.formatted_answer, f"${outcome.cost_usd:.4f}")
```

## Try it in a browser

A small Streamlit page lets people try the agent and rate its answers. Every
question, answer and rating is logged locally so failures from real use can be
turned into new benchmark problems. See [`app/README.md`](app/README.md).

```bash
poetry install --with app
poetry run streamlit run app/streamlit_app.py
```

## Development

```bash
poetry run pytest          # 412 tests, no API key required — everything is mocked
```

Architecture details, API constraints, and testing rules live in
[`SPEC.md`](SPEC.md).

## Project history

This started as a one-week experiment built on FutureHouse's
[Aviary](https://futurehouse.gitbook.io/futurehouse-cookbook/aviary) and
[LDP](https://futurehouse.gitbook.io/futurehouse-cookbook/ldp-language-decision-processes)
frameworks with GPT-4o: a single `GuidedAgent` looping over 14 GCSE tools in a
`ChemistryEnv`, scored by exact string comparison. The original reflection on
building it is kept below — the rebuild grew out of the "what could be improved"
list at the end of it.

`guided_chemistry_agent.ipynb` is retained as a record of that original version
and no longer runs against the current package.

---

### Personal Reflection on the Original Mini-Project

> *Note: Please don't take this too seriously — this is mostly to document the
> rollercoaster of emotions I experienced during one week of this project, while
> working full-time and moving houses. I'd like to do this for every personal
> project in the future.*

#### Opening Thoughts

I've worked with simple machine learning models and more sophisticated neural
nets like VAEs and GANs (using transfer learning and more), but I'd never
created an agent or used an LLM API for development before. I built my project
by referring to the Aviary and LDP cookbooks and tutorials. The frameworks had
limited documentation and few real-world examples, which made it a challenge to
understand all the available tools. That was my starting point.

#### Game Plan

1. **Define a deterministic problem.** After playing around with generating
   arbitrary text (names for people, fruit, etc.), I realised I needed a clear
   validation mechanism. Deterministic outputs would make testing and debugging
   simpler.
2. **Build a unit-conversion tool.** This worked, but general unit conversion
   proved too broad — there are endless units and edge cases.
3. **Focus on chemistry.** GCSE-level chemistry questions (molarity, density,
   etc.) map neatly to well-defined calculations.
4. **Implement tool selection.** I created a chemistry agent that chose the right
   tool for simple questions. It struggled with multi-step problems because it
   lacked context memory.
5. **Add memory with Agent State.** Following the LDP tutorial, I used
   `compute_graph` and an Agent State store to accumulate previous messages and
   reasoning steps. This was the hardest part to debug — running the agent
   repeatedly ate into my budget, but it was essential for chaining tool calls.
6. **Expand calculation capabilities.** The agent still stumbled on basic
   arithmetic in multi-step questions, so I added a generic expression-evaluation
   tool.

By the end, the agent could select chemistry-specific tools, maintain a reasoning
chain across calls, and evaluate arbitrary calculations. Which is pretty cool!!!!

#### Cost

I set a budget of **£10** (≈$13 USD), since I'd never used an LLM API before. In
total, I spent about **≈£2.14** ($2.84 USD).

![Cost breakdown chart](assets/costs.png)

*OpenAI usage for the original version, July–August 2025. The Claude rebuild is
costed per problem in [Results](#results); building and evaluating it, including
every eval run in this README, came to roughly $20.*

#### What Could Be Improved

- **Structured reasoning steps.** Adopt a more formal plan-and-execute loop.
- **Richer toolset.** Thermodynamics, equilibrium calculations, and more.
- **Better logging.** The logs are "ugly"; print statements were more useful.
- **Enhanced rewards for intermediate steps.**
- **"Exceeded Callbacks" bug** from LiteLLM's `MAX_CALLBACKS` limit.

*(The first three are what the current rebuild addresses: the plan → solve →
verify loop, a 39-tool domain toolkit, and rich-rendered progress with real cost
accounting.)*
