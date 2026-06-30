# Contributing to Hermes Edge

## Setup

```bash
git clone https://github.com/simpliibarrii-crypto/hermes-edge
cd hermes-edge
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/ -v
```

Ensure all tests pass before submitting a PR.

## Code Style

- Python: `ruff check hermes/ scripts/ tests/`
- Line length: 100
- Target: Python 3.11+
- Type hints are required for all public functions

## PR Process

1. Fork the repo and create a feature branch
2. Make your changes
3. Run tests and lint
4. Open a pull request against `main`
5. Ensure CI passes

## Commit Messages

Use conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, etc.
