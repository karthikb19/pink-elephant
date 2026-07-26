# Repository guide

- Use `uv`: `uv sync`, `uv run pytest`, and `uv run ruff check .`.
- Keep production code in `src/pink_elephant/` and tests in `tests/`.
- Write small, typed functions with explicit inputs, outputs, and failures; avoid hidden state.
- Add or update focused tests for every behavior change, including edge and failure cases.
- Tests must be deterministic, fast, and offline; mock network, Modal, clocks, randomness, and expensive Torch work.
- Run `uv run ruff format --check .`, `uv run ruff check .`, and `uv run pytest` before handing off.
