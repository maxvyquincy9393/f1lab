# Contributing to F1 Visualization

Thank you for your interest in contributing to this project!

## How to Contribute

### 1. Fork the Repository
```bash
git clone https://github.com/maxvyquincy9393/hub.git
cd "f1 visualization"
```

### 2. Set Up Development Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Update tests if necessary

### 5. Run Tests
```bash
pytest tests/
```

### 6. Commit Your Changes
```bash
git add .
git commit -m "feat: add your feature description"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `test:` adding tests
- `refactor:` code refactoring

### 7. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## Code Style

- Use type hints for function parameters and returns
- Maximum line length: 100 characters
- Use descriptive variable names

## Questions?

Open an issue if you have any questions!

---
Maintained by **maxvy**
