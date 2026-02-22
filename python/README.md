# functiongemma python

Python smart-home orchestrator CLI for FunctionGemma model interaction.

## Files
- `main.py`: interactive chat orchestrator CLI
- `action_executor.py`: webhook action execution (live)
- `pyproject.toml` / `uv.lock`: `uv`-managed Python environment

## Environment
This directory is managed with `uv`.

```bash
cd python
uv sync
```

## Run Interactive Orchestrator
```bash
uv run python main.py --model <model_name> --port 8080 --debug
```
