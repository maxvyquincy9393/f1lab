# Contributing

Thank you for considering contributing to this project.

## Development Setup

```bash
git clone https://github.com/your-username/f1-analytics.git
cd f1-analytics
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Workflow

1. Create a feature branch from `main`
2. Make changes following PEP 8 guidelines
3. Add or update tests as needed
4. Run the test suite: `pytest tests/ -v`
5. Format code: `black src/ tests/ && isort src/ tests/`
6. Submit a pull request

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation only
- `test:` — Test additions/changes
- `refactor:` — Code restructuring

## Code Style

- Follow PEP 8
- Use type hints for function signatures
- Add docstrings for public functions
- Keep functions focused and under 50 lines when practical

## Code Style

- Use type hints for function parameters and returns
- Maximum line length: 100 characters
- Use descriptive variable names

## Questions?

Open an issue if you have any questions!

---
Maintained by **maxvy**
