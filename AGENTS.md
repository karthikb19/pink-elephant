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

## Git workflow

- Prefix every branch name with `kb/`.
- Create draft pull requests when work is ready to share, and keep their status
  and links visible in handoffs.
- Write accurate, durable PR descriptions. State what changed, why it changed,
  the user or developer impact, relevant historical context or root cause, and
  the validation performed. Update the description when the scope changes.

### GitHub CLI and PR creation

- GitHub CLI (`gh`) credentials are stored in the host macOS keychain and may
  not be available from the filesystem sandbox. For GitHub API work, run `gh`
  with host access (`require_escalated`); do not refresh or expose the token
  through `GH_TOKEN`.
- When asked to open a PR, inspect the working tree and stage only the intended
  files. Commit intentionally, push the `kb/` branch with upstream tracking,
  then open a draft PR unless the user explicitly asks for a ready-for-review
  PR.
- If host-side `gh` authentication fails, ask the user to run
  `gh auth login -h github.com` rather than attempting to repair credentials.

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
