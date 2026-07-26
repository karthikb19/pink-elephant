# Repository guide

- Use `uv`: `uv sync`, `uv run pytest`, and `uv run ruff check .`.
- Keep production code in `src/pink_elephant/` and tests in `tests/`.
- Write small, typed functions with explicit inputs, outputs, and failures; avoid hidden state.
- Model structured data precisely. Avoid bare `dict`, `Any`, and opaque shapes such as
  `list[dict[str, Any]]`; prefer a typed model (`pydantic`, `TypedDict`, or a named tuple).
  Use `Any` only as a last resort when the schema cannot be expressed.
- Add or update focused tests for every behavior change, including edge and failure cases.
- Tests must be deterministic, fast, and offline; mock network, Modal, clocks, randomness, and expensive Torch work.
- Run `uv run ruff format --check .`, `uv run ruff check .`, and `uv run pytest` before handing off.

## Project knowledge

Keep durable project context, plans, and decisions in `knowledge/`.

- Add architecture decision records (ADRs) to `knowledge/decisions/` as
  `YYYY-MM-DD-<name>.md`, using `knowledge/decisions/TEMPLATE.md`.
- Every ADR must include, in this order: Context, Decision, Alternatives,
  Consequences, and Surface Areas. ADRs are not required for bug fixes,
  documentation fixes, dependency bumps, or changes without a design decision.
  When uncertain, add a concise three-line ADR instead: it is cheap and keeps
  the decision record clear.
- Use `knowledge/scratch/` for informal scratch notes, working plans, and
  temporary project thinking.
