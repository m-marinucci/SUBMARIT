# Contributing to SUBMARIT

Thank you for your interest in contributing to SUBMARIT! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/yourusername/submarit.git
   cd submarit
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e ".[dev,docs]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Standards

### Python Style Guide
- Follow PEP 8 with a line length limit of 88 characters
- Use Black for code formatting
- Use isort for import sorting
- Use type hints for all functions
- Write descriptive docstrings for all public APIs using Google style
- Pre-commit hooks will automatically check and format your code

### Testing
- Write tests for all new functionality
- Maintain test coverage above 80%
- Use pytest for testing
- Include both unit tests and integration tests
- Test edge cases and error conditions

### Documentation
- Update docstrings for any modified functions
- Update the relevant documentation in `docs/`
- Include examples in docstrings where appropriate
- Keep the changelog updated

#### Building Documentation Locally
```bash
cd docs
make clean html
```

View the built documentation at `docs/_build/html/index.html`.

For live preview during development:
```bash
cd docs
make livehtml
```

Our documentation is automatically built and deployed to GitHub Pages:
https://m-marinucci.github.io/SubmarketIdentificationTesting/

## Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Follow conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test additions or modifications
   - `refactor:` for code refactoring
   - `perf:` for performance improvements
   - `chore:` for maintenance tasks

3. Run tests locally:
   ```bash
   pytest
   ```

4. Run linting and formatting:
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

5. Push your changes and create a pull request

## Pull Request Process

1. Ensure all tests pass and coverage requirements are met
2. Update the README.md if needed
3. Update the changelog in CHANGELOG.md
4. Request review from maintainers
5. Address any feedback
6. Once approved, the PR will be merged

## Testing Guidelines

### Unit Tests
- Test individual functions and methods
- Mock external dependencies
- Test edge cases and error conditions
- Keep tests focused and independent

### Integration Tests
- Test interactions between components
- Use realistic test data
- Test the full workflow

### Numerical Testing
- Use appropriate tolerances for floating-point comparisons
- Test against MATLAB reference outputs where available
- Document expected numerical differences

## Performance Considerations

- Profile code before optimizing
- Document performance characteristics
- Consider using Numba for hot loops
- Vectorize operations where possible
- Benchmark against MATLAB implementation

## Questions?

Feel free to open an issue for any questions or concerns.