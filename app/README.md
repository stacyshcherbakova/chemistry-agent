# Chemistry Agent web app

A small Streamlit page for letting real people (students, a tutor) try the
solver, with every question logged so you can see where it fails in practice.

## Run it

From the repo root:

```bash
poetry install --with app
poetry run streamlit run app/streamlit_app.py
```

The API key is found the same way as for `chem solve`: `ANTHROPIC_API_KEY` (or
`ANTHROPIC_AUTH_TOKEN`) in the environment, or in a `.env` file in the working
directory or a parent. Without one, the page shows an error and the Solve button
is disabled.

Each question costs roughly $0.10 to $0.15 with the default models (more with
retries), billed to your key. Anyone who can open the page can spend it, so
only share the URL with people you trust.

## What gets logged

`app/usage_log.jsonl` (gitignored), one JSON object per line, append only.
A solve:

```json
{"type": "solve", "id": "342fcba9b49a4f90bf1375267170a9cc", "timestamp": "2026-10-23T19:30:09+00:00", "question": "What mass of CaO forms from 25.0 g CaCO3?", "level": "gcse", "formatted_answer": "14.0 g", "value": 14.0, "units": "g", "insufficient_data": false, "verifier_approved": true, "attempts": 1, "tool_calls": ["molar_mass", "mass_to_moles"], "cost_usd": 0.0123, "error": null}
```

Feedback, written later as its own line and joined to the solve by `solve_id`:

```json
{"type": "feedback", "solve_id": "342fcba9b49a4f90bf1375267170a9cc", "timestamp": "2026-10-23T19:31:40+00:00", "rating": "down", "comment": "expected 14 g to 2 s.f."}
```

`verifier_approved` is `null` when the verifier could not run. `rating` is
`"up"`, `"down"` or `null` (comment only).

To see failures, for example:

```bash
jq -c 'select(.type=="solve" and (.verifier_approved != true or .error != null))' app/usage_log.jsonl
jq -c 'select(.type=="feedback" and .rating=="down")' app/usage_log.jsonl
```

## Privacy

The log stores the question text, the answer and any feedback comment, with a
UTC timestamp. It stores no names, IP addresses, session ids or API keys.
Questions are also sent to the Anthropic API to be solved. The sidebar tells
users their questions are saved and asks them not to type anything personal.
Delete the file to clear the log.
