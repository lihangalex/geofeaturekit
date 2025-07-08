# Contributing to GeoFeatureKit

We welcome contributions to GeoFeatureKit! This guide will help you get started.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/geofeaturekit.git
   cd geofeaturekit
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv test_env
   source test_env/bin/activate  # On Windows: test_env\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-test.txt
   pip install -e .
   ```

## 🧪 Running Tests

We use `tox` for testing across multiple Python versions:

```bash
# Run tests for your Python version
tox -e py39  # or py310, py311, py312

# Run all tests
tox

# Run with coverage
tox -e coverage
```

## 📝 Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings for all public functions
- Keep functions focused and well-documented

## 🐛 Reporting Issues

Before creating an issue:
- Check if it already exists in our [issue tracker](https://github.com/lihangalex/geofeaturekit/issues)
- Use our issue templates when available
- Include minimal reproduction steps
- Provide system information (OS, Python version, package versions)

## ✨ Submitting Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write tests for new functionality
   - Update documentation if needed
   - Ensure all tests pass

3. **Commit with a clear message**:
   ```bash
   git commit -m "Add feature: brief description of what you added"
   ```

4. **Push and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## 📋 Pull Request Guidelines

- **Title**: Clear, descriptive summary
- **Description**: What changes you made and why
- **Tests**: Include tests for new features
- **Documentation**: Update README or docstrings if needed
- **Scope**: Keep PRs focused on a single feature/fix

## 🎯 Areas We Need Help

- **New feature implementations** (see roadmap in README)
- **Performance optimizations**
- **Documentation improvements**
- **Test coverage expansion**
- **Bug fixes**

## 📚 Development Tips

- **Cache management**: Be mindful of OSM data caching during development
- **Test data**: Use small areas for faster testing
- **Performance**: Profile memory usage for large datasets
- **Dependencies**: Minimize new dependencies, prefer lightweight alternatives

## 💬 Getting Help

- **Questions**: Open a discussion on GitHub
- **Feature requests**: Contact lihangalex@pm.me
- **Bugs**: Use the issue tracker

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to GeoFeatureKit!** 🌍 