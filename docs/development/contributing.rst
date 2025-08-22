Contributing to Groggy
=====================

Thank you for your interest in contributing to Groggy! This guide covers everything you need to know to contribute effectively to the project.

Getting Started
---------------

Development Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Prerequisites**:
- Rust 1.70+ (install via `rustup <https://rustup.rs/>`_)
- Python 3.8+ 
- Git
- C compiler (GCC, Clang, or MSVC)

**Clone and Build**:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/groggy-dev/groggy.git
   cd groggy

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -r requirements-dev.txt

   # Build the project
   maturin develop --release

   # Run tests to verify installation
   python -m pytest tests/

Development Tools
~~~~~~~~~~~~~~~~

**Required Tools**:

.. code-block:: bash

   # Rust development tools
   rustup component add rustfmt clippy
   cargo install cargo-audit cargo-deny

   # Python development tools
   pip install black isort mypy pre-commit
   pip install pytest pytest-cov pytest-benchmark

   # Documentation tools
   pip install sphinx sphinx-rtd-theme

**Pre-commit Hooks**:

.. code-block:: bash

   # Install pre-commit hooks
   pre-commit install

   # Run hooks manually
   pre-commit run --all-files

Code Style and Standards
-----------------------

Rust Code Style
~~~~~~~~~~~~~~~

We follow standard Rust conventions:

**Formatting**:

.. code-block:: bash

   # Format Rust code
   cargo fmt --all

   # Check formatting
   cargo fmt --all -- --check

**Linting**:

.. code-block:: bash

   # Run Clippy linter
   cargo clippy --all-targets --all-features -- -D warnings

**Example Rust Code Style**:

.. code-block:: rust

   // Good: Clear, documented function
   /// Calculate betweenness centrality for all nodes in the graph.
   ///
   /// # Arguments
   /// * `graph` - The graph to analyze
   /// * `normalized` - Whether to normalize the centrality values
   ///
   /// # Returns
   /// A HashMap mapping node IDs to centrality scores
   ///
   /// # Panics
   /// Panics if the graph is empty
   pub fn betweenness_centrality(
       graph: &GraphCore,
       normalized: bool,
   ) -> Result<HashMap<NodeId, f64>, GraphError> {
       if graph.node_count() == 0 {
           return Err(GraphError::EmptyGraph);
       }

       let mut centrality = HashMap::new();
       
       // Implementation...
       
       Ok(centrality)
   }

   // Bad: Unclear, undocumented function
   pub fn bc(g: &GraphCore, n: bool) -> HashMap<NodeId, f64> {
       // Implementation...
   }

Python Code Style
~~~~~~~~~~~~~~~~~

We use Black for formatting and follow PEP 8:

**Formatting**:

.. code-block:: bash

   # Format Python code
   black .
   isort .

   # Type checking
   mypy src/

**Example Python Code Style**:

.. code-block:: python

   # Good: Type hints, clear docstring, proper error handling
   def compute_centrality(
       graph: Graph, 
       algorithm: str = "pagerank",
       **kwargs: Any
   ) -> Dict[str, float]:
       """Compute centrality measure for all nodes.
       
       Args:
           graph: The graph to analyze
           algorithm: Centrality algorithm to use
           **kwargs: Algorithm-specific parameters
           
       Returns:
           Dictionary mapping node IDs to centrality scores
           
       Raises:
           ValueError: If algorithm is not supported
           RuntimeError: If computation fails
       """
       if algorithm not in SUPPORTED_ALGORITHMS:
           raise ValueError(f"Unsupported algorithm: {algorithm}")
       
       try:
           return _compute_centrality_impl(graph, algorithm, **kwargs)
       except Exception as e:
           raise RuntimeError(f"Centrality computation failed: {e}") from e

   # Bad: No types, unclear purpose, poor error handling
   def compute(g, alg="pr", **kw):
       return _compute_impl(g, alg, **kw)

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~

**Code Documentation**:
- All public functions must have docstrings
- Rust functions should use `///` doc comments
- Python functions should use Google-style docstrings
- Include examples for complex functions

**Commit Messages**:

.. code-block:: text

   # Good commit message format
   feat: add betweenness centrality algorithm

   - Implement Brandes algorithm for betweenness centrality
   - Add parallel processing support
   - Include comprehensive tests
   - Update documentation

   Fixes #123

   # Bad commit messages
   fix bug
   update code
   wip

**Commit Prefixes**:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Test additions/changes
- `build:` - Build system changes

Contributing Process
-------------------

Finding Issues to Work On
~~~~~~~~~~~~~~~~~~~~~~~~~

**Good First Issues**:
- Look for issues labeled `good first issue`
- Documentation improvements
- Adding tests for existing functionality
- Small bug fixes

**Medium Complexity**:
- New algorithm implementations
- Performance optimizations
- API improvements

**Advanced Issues**:
- Core architecture changes
- FFI layer improvements
- Memory management optimizations

Issue Workflow
~~~~~~~~~~~~~

1. **Check Existing Issues**: Search for related issues before creating new ones

2. **Create/Comment on Issue**: 
   - Describe the problem/feature clearly
   - Include minimal reproduction case for bugs
   - Discuss approach before major changes

3. **Get Assignment**: Comment on issue to get assigned

4. **Development**:
   - Create feature branch: `git checkout -b feature/your-feature-name`
   - Make changes following code standards
   - Add tests for new functionality
   - Update documentation

5. **Testing**:
   - Run full test suite: `python -m pytest`
   - Run Rust tests: `cargo test --all`
   - Check performance regressions: `python -m pytest --benchmark-only`

6. **Submit Pull Request**:
   - Create PR with clear description
   - Link to related issues
   - Include testing information

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~

**PR Description Template**:

.. code-block:: markdown

   ## Description
   Brief description of the changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Performance improvement
   - [ ] Documentation update
   - [ ] Refactoring

   ## Testing
   - [ ] Added/updated unit tests
   - [ ] Added/updated integration tests
   - [ ] Tested on multiple platforms
   - [ ] Benchmarked performance impact

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests pass locally

   ## Related Issues
   Fixes #123

**Review Process**:
1. Automated checks must pass (CI, formatting, tests)
2. Code review by maintainers
3. Address feedback
4. Final approval and merge

Testing Guidelines
-----------------

Test Structure
~~~~~~~~~~~~~

.. code-block:: text

   tests/
   ├── unit/                 # Unit tests
   │   ├── rust/            # Rust unit tests
   │   └── python/          # Python unit tests
   ├── integration/         # Integration tests
   ├── benchmarks/          # Performance benchmarks
   └── fixtures/            # Test data and utilities

Writing Tests
~~~~~~~~~~~~

**Rust Tests**:

.. code-block:: rust

   #[cfg(test)]
   mod tests {
       use super::*;
       use crate::test_utils::create_test_graph;

       #[test]
       fn test_pagerank_basic() {
           let graph = create_test_graph();
           let result = pagerank(&graph, 0.85, 100, 1e-6).unwrap();
           
           // Check that all nodes have positive centrality
           for &centrality in result.values() {
               assert!(centrality > 0.0);
           }
           
           // Check that centralities sum to node count (approximately)
           let sum: f64 = result.values().sum();
           assert!((sum - graph.node_count() as f64).abs() < 1e-3);
       }

       #[test]
       fn test_pagerank_empty_graph() {
           let graph = GraphCore::new(true);
           let result = pagerank(&graph, 0.85, 100, 1e-6);
           
           assert!(result.is_err());
           assert!(matches!(result.unwrap_err(), GraphError::EmptyGraph));
       }
   }

**Python Tests**:

.. code-block:: python

   import pytest
   import groggy as gr

   class TestPageRank:
       def test_pagerank_basic(self):
           """Test basic PageRank functionality"""
           g = gr.Graph()
           g.add_nodes(['A', 'B', 'C'])
           g.add_edges([('A', 'B'), ('B', 'C'), ('C', 'A')])
           
           result = g.centrality.pagerank()
           
           # All nodes should have positive centrality
           assert all(score > 0 for score in result.values())
           
           # Centralities should sum to 1 (approximately)
           assert abs(sum(result.values()) - 1.0) < 1e-6

       def test_pagerank_empty_graph(self):
           """Test PageRank on empty graph"""
           g = gr.Graph()
           
           with pytest.raises(ValueError, match="empty"):
               g.centrality.pagerank()

       @pytest.mark.parametrize("alpha", [0.5, 0.85, 0.95])
       def test_pagerank_different_alphas(self, alpha):
           """Test PageRank with different damping factors"""
           g = self._create_test_graph()
           result = g.centrality.pagerank(alpha=alpha)
           
           assert len(result) == g.node_count()
           assert all(score > 0 for score in result.values())

**Benchmark Tests**:

.. code-block:: python

   import pytest
   import groggy as gr

   class TestPerformance:
       @pytest.mark.benchmark(group="pagerank")
       def test_pagerank_small_graph(self, benchmark):
           """Benchmark PageRank on small graph"""
           g = gr.random_graph(1000, edge_probability=0.01)
           
           result = benchmark(g.centrality.pagerank)
           
           assert len(result) == 1000

       @pytest.mark.benchmark(group="pagerank")
       def test_pagerank_large_graph(self, benchmark):
           """Benchmark PageRank on larger graph"""
           g = gr.random_graph(10000, edge_probability=0.001)
           
           result = benchmark(g.centrality.pagerank)
           
           assert len(result) == 10000

Running Tests
~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   python -m pytest

   # Run specific test file
   python -m pytest tests/unit/python/test_centrality.py

   # Run with coverage
   python -m pytest --cov=groggy

   # Run benchmarks
   python -m pytest --benchmark-only

   # Run Rust tests
   cargo test --all

   # Run specific Rust test
   cargo test pagerank

Performance Considerations
-------------------------

Benchmarking
~~~~~~~~~~~

Always benchmark changes that might affect performance:

.. code-block:: python

   # Create benchmark test
   def test_new_algorithm_performance(benchmark):
       g = create_large_test_graph()
       result = benchmark(g.new_algorithm)
       validate_result(result)

   # Run baseline
   python -m pytest tests/benchmarks/test_new_algorithm.py --benchmark-save=baseline

   # Run with changes
   python -m pytest tests/benchmarks/test_new_algorithm.py --benchmark-compare=baseline

Memory Usage
~~~~~~~~~~~

Monitor memory usage for algorithms:

.. code-block:: python

   import tracemalloc

   def test_memory_usage():
       tracemalloc.start()
       
       g = create_large_test_graph()
       result = g.memory_intensive_algorithm()
       
       current, peak = tracemalloc.get_traced_memory()
       tracemalloc.stop()
       
       # Memory usage should be reasonable
       assert peak < 1024 * 1024 * 1024  # 1GB

Profiling
~~~~~~~~

Use profiling to identify bottlenecks:

.. code-block:: bash

   # Profile Python code
   python -m cProfile -o profile.stats script.py
   python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

   # Profile Rust code
   cargo build --release
   perf record target/release/benchmark
   perf report

Documentation Contributions
---------------------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install documentation dependencies
   pip install -r docs/requirements.txt

   # Build documentation
   cd docs
   make html

   # View documentation
   open _build/html/index.html

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~

**API Documentation**:
- All public APIs must be documented
- Include usage examples
- Document parameters and return values
- Note any exceptions that can be raised

**Tutorials**:
- Step-by-step instructions
- Complete, runnable examples
- Explain the "why" not just the "how"
- Include expected output

**Architecture Documentation**:
- High-level overviews
- Detailed technical specifications
- Design decisions and trade-offs
- Performance characteristics

Release Process
--------------

Version Management
~~~~~~~~~~~~~~~~~

We use semantic versioning (SemVer):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

Release Checklist
~~~~~~~~~~~~~~~~~

1. **Update Version Numbers**:
   - `Cargo.toml`
   - `pyproject.toml`
   - `__init__.py`

2. **Update Documentation**:
   - CHANGELOG.md
   - Release notes
   - API documentation

3. **Testing**:
   - Full test suite on all platforms
   - Performance regression tests
   - Integration tests

4. **Build and Package**:
   - Build wheels for all platforms
   - Test installation from packages
   - Generate documentation

5. **Release**:
   - Tag release in Git
   - Upload to PyPI
   - Create GitHub release
   - Update documentation website

Getting Help
-----------

**Community Channels**:
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: Questions and general discussion
- Documentation: Comprehensive guides and API reference

**Maintainer Contact**:
- For security issues: security@groggy.dev
- For urgent matters: maintainers@groggy.dev

**Response Times**:
- Issues: Within 1-2 business days
- Pull requests: Within 3-5 business days
- Security issues: Within 24 hours

Recognition
----------

All contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Documentation acknowledgments

Significant contributors may be invited to join the core team.

Thank you for contributing to Groggy! Your efforts help make high-performance graph analysis accessible to the Python community.