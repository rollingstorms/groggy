# Groggy Documentation

This directory contains the complete Sphinx documentation for Groggy (Graph Language Interface).

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r docs-requirements.txt
```

### Building

```bash
# Build HTML documentation
make html

# Build and serve locally
make html && make serve

# Watch for changes and rebuild automatically
make watch

# Clean build files
make clean
```

### Development

For development, use the development build with warnings as errors:

```bash
make dev-build
```

## Documentation Structure

```
docs/
├── source/
│   ├── _static/           # Static assets (CSS, images)
│   ├── _templates/        # Custom templates
│   ├── api/              # API reference documentation
│   ├── examples/         # Usage examples
│   ├── rust/             # Rust backend documentation
│   ├── conf.py           # Sphinx configuration
│   ├── index.rst         # Main documentation index
│   ├── installation.rst  # Installation guide
│   ├── quickstart.rst    # Quick start guide
│   ├── performance.rst   # Performance guide
│   ├── architecture.rst  # Architecture overview
│   ├── testing.rst       # Testing guide
│   ├── contributing.rst  # Contributing guide
│   └── changelog.rst     # Change log
├── build/                # Generated documentation (git-ignored)
├── Makefile              # Build automation
├── docs-requirements.txt # Documentation dependencies
└── README.md            # This file
```

## Key Features

- **Complete API Reference**: Auto-generated from docstrings
- **Performance Guides**: Optimization techniques and benchmarks
- **Architecture Documentation**: Detailed system design
- **Examples**: Practical usage examples
- **Rust Backend**: Specialized documentation for Rust backend

## Contributing to Documentation

1. **Edit Source Files**: Modify `.rst` files in `source/`
2. **Add Examples**: Include practical code examples
3. **Test Locally**: Use `make dev-build` to test changes
4. **Check Links**: Run `make linkcheck` to verify external links

### Writing Guidelines

- Use clear, concise language
- Include practical examples
- Add code comments for complex examples
- Use consistent formatting
- Cross-reference related sections

### API Documentation

API documentation is automatically generated from Python docstrings. To improve it:

1. Write comprehensive docstrings
2. Include parameter descriptions
3. Add usage examples
4. Document exceptions
5. Use proper type hints

Example docstring format:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """Brief description of the function.
    
    Longer description with more details about what the function does,
    its behavior, and any important notes.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter with default value
        
    Returns:
        Description of the return value
        
    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer
        
    Example:
        >>> result = example_function("hello", 20)
        >>> print(result)
        True
    """
    pass
```

## Advanced Features

### Custom CSS

The documentation includes custom CSS for:
- Better code highlighting
- Performance metrics styling
- Responsive design
- Groggy-specific branding

### Cross-References

Use Sphinx cross-references to link between sections:

```rst
See :doc:`quickstart` for basic usage.
Reference :func:`groggy.Graph.add_node` for details.
```

### Code Examples

Include testable code examples:

```rst
.. code-block:: python

   from groggy import Graph
   
   g = Graph()
   alice = g.add_node(name="Alice")
   print(f"Added node: {alice}")
```

## Deployment

The documentation can be deployed to various platforms:

- **GitHub Pages**: Via GitHub Actions
- **Read the Docs**: Automatic builds from repository
- **Internal Hosting**: Static HTML files in `build/html/`

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install with `pip install -r docs-requirements.txt`
2. **Build Errors**: Use `make dev-build` for detailed error messages
3. **Broken Links**: Run `make linkcheck` to identify issues
4. **Import Errors**: Ensure Groggy is installed in the environment

### Performance

For large documentation builds:
- Use `make html` instead of `make dev-build` for faster builds
- Consider using `sphinx-autobuild` for live reloading
- Cache can be cleared with `make clean`

## Additional Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [MyST Parser](https://myst-parser.readthedocs.io/) for Markdown support
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
