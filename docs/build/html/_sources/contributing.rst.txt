Contributing to GLI
==================

We welcome contributions to GLI! This guide will help you get started with contributing code, documentation, or bug reports.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~

1. **Fork and Clone**

   .. code-block:: bash

      # Fork the repository on GitHub, then:
      git clone https://github.com/your-username/gli.git
      cd gli

2. **Set up Development Environment**

   .. code-block:: bash

      # Create virtual environment
      python -m venv gli-dev
      source gli-dev/bin/activate  # On Windows: gli-dev\\Scripts\\activate
      
      # Install development dependencies
      pip install -e ".[dev]"
      
      # Install Rust (if not already installed)
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
      source ~/.cargo/env
      
      # Install maturin for Rust-Python bindings
      pip install maturin

3. **Build Rust Backend**

   .. code-block:: bash

      # Build in development mode
      maturin develop
      
      # Or build in release mode for performance testing
      maturin develop --release

4. **Verify Installation**

   .. code-block:: bash

      # Run tests to verify everything works
      python -m pytest tests/test_backend_parity.py
      
      # Check both backends are available
      python -c "from groggy import get_available_backends; print(get_available_backends())"

Development Workflow
-------------------

Branch Strategy
~~~~~~~~~~~~~~

We use a standard Git flow:

.. code-block:: bash

   # Create feature branch
   git checkout -b feature/your-feature-name
   
   # Make changes and commit
   git add .
   git commit -m "Add feature: brief description"
   
   # Push and create pull request
   git push origin feature/your-feature-name

Code Standards
~~~~~~~~~~~~~

**Python Code Style**

We follow PEP 8 with some modifications:

.. code-block:: bash

   # Install code formatting tools
   pip install black isort flake8 mypy
   
   # Format code
   black gli/ tests/
   isort gli/ tests/
   
   # Check style
   flake8 gli/ tests/
   mypy gli/

**Rust Code Style**

.. code-block:: bash

   # Format Rust code
   cd src/
   cargo fmt
   
   # Check with clippy
   cargo clippy

**Documentation Style**

.. code-block:: bash

   # Build documentation
   cd docs/
   make html
   
   # Check for broken links
   make linkcheck

Types of Contributions
---------------------

Bug Reports
~~~~~~~~~~

When reporting bugs, please include:

1. **Minimal Reproduction Case**

   .. code-block:: python

      from groggy import Graph
      
      # Minimal code that demonstrates the bug
      g = Graph()
      # ... specific steps that cause the issue

2. **Environment Information**

   .. code-block:: bash

      # Include this information
      python --version
      python -c "import groggy; print(groggy.__version__)"
      python -c "from groggy import get_available_backends; print(get_available_backends())"

3. **Expected vs Actual Behavior**

   Clearly describe what you expected to happen and what actually happened.

Feature Requests
~~~~~~~~~~~~~~~

For new features, please:

1. **Check Existing Issues**: See if it's already been requested
2. **Provide Use Case**: Explain why this feature would be useful
3. **Consider Alternatives**: Are there existing ways to achieve this?
4. **API Design**: Suggest how the feature should work

.. code-block:: python

   # Example feature request
   # Proposed API for new feature
   g = Graph()
   result = g.new_proposed_method(parameter=value)

Code Contributions
~~~~~~~~~~~~~~~~~

**Small Changes**
- Bug fixes
- Documentation improvements
- Performance optimizations

**Medium Changes**
- New utility functions
- Additional graph algorithms
- API improvements

**Large Changes**
- New backends
- Major architectural changes
- New core features

Please discuss large changes in an issue first.

Contribution Guidelines
----------------------

Code Quality
~~~~~~~~~~~

1. **Tests Required**: All new code must include tests

   .. code-block:: python

      def test_new_feature():
          \"\"\"Test description\"\"\"
          g = Graph()
          # Test implementation
          assert expected_result == actual_result

2. **Backend Parity**: Features must work on both backends

   .. code-block:: python

      @pytest.mark.parametrize("backend", ["python", "rust"])
      def test_feature_both_backends(backend):
          g = Graph(backend=backend)
          # Test same behavior on both backends

3. **Documentation**: Public APIs need docstrings

   .. code-block:: python

      def new_method(self, parameter: str) -> str:
          \"\"\"Brief description.
          
          Args:
              parameter: Description of parameter
              
          Returns:
              Description of return value
              
          Raises:
              ValueError: When parameter is invalid
              
          Example:
              >>> g = Graph()
              >>> result = g.new_method("value")
              >>> print(result)
              "processed_value"
          \"\"\"
          return f"processed_{parameter}"

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Benchmark New Features**

   .. code-block:: python

      def test_performance_new_feature():
          import time
          
          g = Graph(backend="rust")
          
          start = time.time()
          for i in range(10000):
              g.new_feature(f"item_{i}")
          duration = time.time() - start
          
          # Should complete within reasonable time
          assert duration < 1.0  # 1 second max

2. **Memory Efficiency**

   .. code-block:: python

      import tracemalloc
      
      def test_memory_usage_new_feature():
          tracemalloc.start()
          
          g = Graph()
          for i in range(1000):
              g.new_feature(f"item_{i}")
          
          current, peak = tracemalloc.get_traced_memory()
          tracemalloc.stop()
          
          # Should not use excessive memory
          assert peak < 10 * 1024 * 1024  # 10MB max

3. **Rust Backend Integration**

   For performance-critical features, implement in Rust:

   .. code-block:: rust

      #[pymethods]
      impl FastGraph {
          fn new_fast_method(&mut self, parameter: String) -> PyResult<String> {
              // High-performance implementation
              Ok(format!("processed_{}", parameter))
          }
      }

Specific Contribution Areas
--------------------------

Documentation
~~~~~~~~~~~~

We always need help with documentation:

1. **API Documentation**: Improve docstrings and examples
2. **Tutorials**: Create learning materials
3. **Architecture Docs**: Explain design decisions
4. **Performance Guides**: Document optimization techniques

.. code-block:: bash

   # Build documentation locally
   cd docs/
   pip install sphinx sphinx-rtd-theme myst-parser
   make html
   # Open docs/build/html/index.html

Testing
~~~~~~

Help improve our test coverage:

1. **Edge Cases**: Test unusual inputs and conditions
2. **Error Conditions**: Test error handling
3. **Performance Tests**: Add benchmark tests
4. **Integration Tests**: Test with external systems

.. code-block:: python

   # Example edge case test
   def test_edge_case_empty_attributes():
       g = Graph()
       node_id = g.add_node()  # No attributes
       node_data = g.get_node(node_id)
       assert len(node_data.attributes) == 0

Performance
~~~~~~~~~~

Help make GLI faster:

1. **Profiling**: Find performance bottlenecks
2. **Optimization**: Improve hot code paths
3. **Memory Usage**: Reduce memory consumption
4. **Rust Implementation**: Port Python code to Rust

.. code-block:: python

   # Performance profiling example
   import cProfile
   
   def profile_operation():
       g = Graph()
       for i in range(10000):
           g.add_node(f"node_{i}")
   
   cProfile.run('profile_operation()')

New Features
~~~~~~~~~~~

Popular feature requests:

1. **Graph Algorithms**: Shortest path, centrality measures, etc.
2. **Import/Export**: Additional file formats (GraphML, GEXF, etc.)
3. **Visualization**: Integration with plotting libraries
4. **Streaming**: Real-time graph updates

Review Process  
--------------

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~

1. **Clear Description**: Explain what the PR does and why
2. **Small, Focused Changes**: One feature/fix per PR
3. **Tests Included**: All changes must have tests
4. **Documentation Updated**: Update docs for user-facing changes

.. code-block:: markdown

   ## Pull Request Template
   
   ### Description
   Brief description of the changes
   
   ### Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ### Testing
   - [ ] Tests pass locally
   - [ ] New tests added for changes
   - [ ] Both backends tested
   
   ### Documentation
   - [ ] Docstrings updated
   - [ ] User documentation updated (if needed)

Code Review Process
~~~~~~~~~~~~~~~~~~

1. **Automated Checks**: CI must pass
2. **Peer Review**: At least one maintainer review
3. **Testing**: Verify tests are comprehensive
4. **Documentation**: Check docs are updated

Community Guidelines
-------------------

Code of Conduct
~~~~~~~~~~~~~~~

We follow the Python Community Code of Conduct:

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment
- Report inappropriate behavior

Communication
~~~~~~~~~~~~

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions and discussions
- **Discussions**: General questions and ideas

Getting Help
~~~~~~~~~~~

If you need help:

1. **Check Documentation**: Most questions are answered in docs
2. **Search Issues**: See if others have asked similar questions
3. **Create Issue**: Ask specific questions with context
4. **Join Discussions**: Participate in community discussions

Recognition
----------

Contributors are recognized through:

- **Changelog**: All contributors mentioned in releases
- **Contributors File**: GitHub contributors page
- **Special Thanks**: Major contributors highlighted

Thank you for contributing to GLI! 🚀
