# Contributing to Liquid Edge LLN Kit

Thank you for your interest in contributing! This project focuses on bringing efficient liquid neural networks to edge robotics.

## Development Setup

```bash
# Clone and setup
git clone https://github.com/liquid-edge/liquid-edge-lln-kit.git
cd liquid-edge-lln-kit
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Standards

- **Python 3.10+** required
- **Code style**: Black formatter, Ruff linter
- **Type hints**: Required for all public APIs
- **Tests**: pytest with >90% coverage
- **Documentation**: Docstrings in Google style

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=liquid_edge --cov-report=html

# Specific test
pytest tests/test_core.py::test_liquid_nn
```

## Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes with tests
4. Run quality checks: `pre-commit run --all-files`
5. Submit PR with clear description

## Priority Areas

- **MCU platforms**: STM32, ESP32, nRF52 support
- **Optimization**: Energy efficiency improvements
- **Examples**: Real robot applications
- **Documentation**: Tutorials and guides

## Issues

- Use issue templates
- Include reproduction steps
- Specify hardware/platform details
- Check existing issues first

## Code of Conduct

Be respectful, inclusive, and constructive. Focus on technical merit and project goals.

## License

By contributing, you agree your code will be licensed under MIT License.