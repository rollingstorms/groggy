# Contributing to Groggy

Thank you for your interest in contributing to Groggy! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Rust 1.70 or later
- Python 3.8 or later
- Git

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/groggy.git
   cd groggy
   ```

3. Set up the development environment:
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -r requirements-dev.txt

   # Build the Rust extension
   maturin develop
   ```

4. Run tests to ensure everything is working:
   ```bash
   pytest tests/
   cargo test
   ```

## Development Workflow

### Making Changes

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes in the appropriate directories:
   - `src/` - Rust code
   - `python/groggy/` - Python bindings and utilities
   - `tests/` - Test files
   - `docs/` - Documentation

3. Write tests for your changes
4. Run the test suite to ensure nothing is broken
5. Update documentation if necessary

### Code Style

#### Rust Code
- Follow standard Rust formatting (`cargo fmt`)
- Run Clippy for linting (`cargo clippy`)
- Add documentation comments for public APIs

#### Python Code
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings for functions and classes

### Testing

- Write unit tests for new functionality
- Ensure all existing tests pass
- Add integration tests for complex features
- Test performance-critical code with benchmarks

### Running Tests

```bash
# Python tests
pytest tests/

# Rust tests
cargo test

# Run specific test files
pytest tests/test_specific_feature.py
cargo test specific_module
```

## Continuous Integration & Deployment

Groggy uses GitHub Actions to safeguard the project across languages:

- `.github/workflows/test.yml` runs the full lint/test matrix. It enforces `cargo fmt -- --check`, `cargo clippy -- -D warnings`, `cargo test`, and builds/tests the Python extension on Ubuntu, macOS, and Windows for Python 3.8–3.12.
- `.github/workflows/ci.yml` is a lighter Python regression sweep on pushes/PRs to `main`, rebuilding with Maturin and running `pytest tests/` for Python 3.8–3.12.
- `.github/workflows/docs.yml` rebuilds the extension and publishes the MkDocs site when `main` updates.
- `.github/workflows/publish.yml` produces release wheels/SDists for all supported targets and uploads them to PyPI on tagged releases.

Knowing what CI enforces lets you mirror the same checks locally and avoid round-trips during review.

### Pre-PR Checklist

Run the commands below before pushing to ensure parity with CI and keep warnings out of review threads:

```bash
# === Core Rust Package (src/) ===
# Always run fmt last after any debugging/development to ensure consistent formatting
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo check --all-features
cargo test --all-targets

# === Python Extension Package (python-groggy/) ===
# Rebuild the Python extension after Rust/FFI edits
maturin develop --release

# Run clippy/fmt on the Python extension crate separately
cd python-groggy
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cd ..

# === Python Code ===
# Python formatting and linting
black .
isort .
pre-commit run --all-files

# Python tests
pytest tests -q
```

**Important Notes:**
- Always run `cargo fmt` **last** after debugging/making changes - it ensures consistent formatting
- Run clippy/fmt in both the root crate (`src/`) and the Python extension crate (`python-groggy/`)
- The Python extension has its own Cargo.toml and needs separate checks
- Rebuild with `maturin develop --release` before running Python tests


## Submitting Changes

### Pull Request Process

1. Ensure your code follows the style guidelines
2. Add or update tests as needed
3. Update documentation if you've changed APIs
4. Write a clear commit message describing your changes
5. Push to your fork and submit a pull request

### Pull Request Guidelines

- **Title**: Clear, descriptive title summarizing the change
- **Description**: Detailed description of what the PR does and why
- **Testing**: Describe how you tested your changes
- **Breaking Changes**: Clearly note any breaking changes

### Commit Message Format

```
type(scope): brief description

Longer description if needed

- Bullet points for multiple changes
- Reference issues: Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Environment**: OS, Python version, Rust version
- **Steps to Reproduce**: Clear, minimal steps
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Error Messages**: Full error messages and stack traces

### Feature Requests

For feature requests, please include:

- **Use Case**: Why this feature would be useful
- **Proposed Solution**: Your idea for implementation
- **Alternatives**: Other solutions you've considered

## Performance Considerations

Groggy is designed for high performance. When contributing:

- Consider memory usage and allocation patterns
- Profile performance-critical code
- Add benchmarks for new algorithms
- Avoid unnecessary copying of large data structures

## Documentation

- Update docstrings for any changed APIs
- Add examples for new features
- Update the README if needed
- Consider adding tutorials for complex features

## Code Review Process

1. All submissions require review
2. Maintainers will review your code for:
   - Correctness
   - Style consistency
   - Test coverage
   - Performance implications
   - Documentation quality

3. Address feedback and update your PR
4. Once approved, a maintainer will merge your changes

## Community Guidelines

- Be respectful and constructive
- Help others learn and contribute
- Focus on the code, not the person
- Assume good intentions

## Questions?

If you have questions about contributing:

- Check existing issues and PRs
- Start a discussion on GitHub
- Reach out to maintainers

Thank you for contributing to Groggy!
