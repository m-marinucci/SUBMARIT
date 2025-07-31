Contributing Guide
==================

Thank you for your interest in contributing to SUBMARIT! This guide will help you get started.

.. note::
   This is the Sphinx documentation version of our contributing guide. 
   For the most up-to-date version, please see the `CONTRIBUTING.md <https://github.com/yourusername/submarit/blob/main/CONTRIBUTING.md>`_ file in the repository.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. Fork and clone the repository:

.. code-block:: bash

    git clone https://github.com/yourusername/submarit.git
    cd submarit

2. Create a virtual environment:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install in development mode:

.. code-block:: bash

    pip install -e ".[dev]"

4. Install pre-commit hooks:

.. code-block:: bash

    pre-commit install

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

The ``[dev]`` extras include:

- **Testing**: pytest, pytest-cov, pytest-benchmark
- **Linting**: flake8, black, isort, mypy
- **Documentation**: sphinx, sphinx-rtd-theme
- **Profiling**: memory-profiler, line-profiler

Code Style
----------

We follow PEP 8 with these specifics:

- Line length: 88 characters (Black default)
- Use type hints for function signatures
- Write docstrings for all public functions

Example:

.. code-block:: python

    from typing import Optional, Tuple
    import numpy as np
    
    def calculate_metrics(
        data: np.ndarray,
        clusters: np.ndarray,
        metric: str = "euclidean"
    ) -> Tuple[float, float]:
        """Calculate clustering metrics.
        
        Parameters
        ----------
        data : np.ndarray
            Input data matrix of shape (n_samples, n_features).
        clusters : np.ndarray
            Cluster assignments of shape (n_samples,).
        metric : str, optional
            Distance metric to use, by default "euclidean".
            
        Returns
        -------
        Tuple[float, float]
            Silhouette score and Davies-Bouldin index.
            
        Examples
        --------
        >>> X = np.random.rand(100, 10)
        >>> clusters = np.random.randint(0, 5, 100)
        >>> sil, db = calculate_metrics(X, clusters)
        """
        # Implementation here
        pass

Testing
-------

Writing Tests
~~~~~~~~~~~~~

All new features must include tests:

.. code-block:: python

    # tests/test_new_feature.py
    import pytest
    import numpy as np
    from submarit.new_module import new_function
    
    class TestNewFeature:
        def test_basic_functionality(self):
            """Test basic use case."""
            result = new_function([1, 2, 3])
            assert result == expected_value
            
        def test_edge_cases(self):
            """Test edge cases."""
            with pytest.raises(ValueError):
                new_function([])
                
        @pytest.mark.parametrize("input,expected", [
            ([1, 2], 3),
            ([0, 0], 0),
            ([-1, 1], 0),
        ])
        def test_various_inputs(self, input, expected):
            """Test various input combinations."""
            assert new_function(input) == expected

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

    # Run all tests
    pytest
    
    # Run with coverage
    pytest --cov=submarit --cov-report=html
    
    # Run specific test file
    pytest tests/test_algorithms.py
    
    # Run benchmarks
    pytest benchmarks/ --benchmark-only

Documentation
-------------

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

1. **Docstrings**: Use NumPy style docstrings
2. **User Guide**: Update relevant .rst files in docs/source/
3. **Examples**: Add to docstring examples or create notebook

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd docs
    make clean
    make html
    # View at docs/build/html/index.html

Adding Examples
~~~~~~~~~~~~~~~

Create Jupyter notebooks in ``examples/``:

.. code-block:: python

    # examples/new_feature_demo.ipynb
    """
    # New Feature Demo
    
    This notebook demonstrates the new feature.
    """
    
    import numpy as np
    from submarit import new_feature
    
    # Step-by-step demonstration
    # ...

Pull Request Process
--------------------

1. Create a Feature Branch
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git checkout -b feature/your-feature-name

2. Make Your Changes
~~~~~~~~~~~~~~~~~~~~

- Write code following style guidelines
- Add tests for new functionality
- Update documentation
- Add entry to CHANGELOG.md

3. Commit Your Changes
~~~~~~~~~~~~~~~~~~~~~~

Use clear, descriptive commit messages:

.. code-block:: bash

    git add .
    git commit -m "Add new clustering metric
    
    - Implement Dunn index calculation
    - Add tests for edge cases
    - Update documentation with examples"

4. Run Quality Checks
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Format code
    black submarit tests
    isort submarit tests
    
    # Run linters
    flake8 submarit tests
    mypy submarit
    
    # Run tests
    pytest
    
    # Check documentation
    cd docs && make doctest

5. Push and Create PR
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git push origin feature/your-feature-name

Then create a pull request on GitHub with:

- Clear description of changes
- Link to related issue (if any)
- Screenshots (for visualizations)
- Performance comparisons (if relevant)

Development Guidelines
----------------------

Adding New Algorithms
~~~~~~~~~~~~~~~~~~~~~

1. Inherit from ``BaseClusterer``:

.. code-block:: python

    from submarit.core.base import BaseClusterer
    
    class MyAlgorithm(BaseClusterer):
        def __init__(self, n_clusters, **kwargs):
            super().__init__(n_clusters=n_clusters)
            # Initialize parameters
            
        def fit(self, X):
            # Implement fitting logic
            self.labels_ = ...
            return self
            
        def predict(self, X):
            # Optional: for new data
            return self.labels_

2. Add tests in ``tests/test_algorithms.py``
3. Add documentation in ``docs/source/api/algorithms.rst``
4. Add example in docstring or notebook

Adding New Metrics
~~~~~~~~~~~~~~~~~~

1. Create function in appropriate module:

.. code-block:: python

    def new_metric(S, clusters):
        """Calculate new metric.
        
        Parameters
        ----------
        S : array-like
            Substitution matrix
        clusters : array-like
            Cluster assignments
            
        Returns
        -------
        float
            Metric value
        """
        # Implementation

2. Add to ``ClusterEvaluator`` if appropriate
3. Add tests with known results
4. Document mathematical formula

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Use NumPy vectorization over loops
- Consider memory usage for large datasets
- Add benchmarks for performance-critical code
- Profile before optimizing

Release Process
---------------

Version Numbering
~~~~~~~~~~~~~~~~~

We follow Semantic Versioning (MAJOR.MINOR.PATCH):

- MAJOR: Incompatible API changes
- MINOR: New functionality, backwards compatible
- PATCH: Bug fixes

Release Checklist
~~~~~~~~~~~~~~~~~

1. Update version in ``submarit/__init__.py``
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test documentation
5. Create release branch
6. Tag release
7. Build and upload to PyPI

Community
---------

Code of Conduct
~~~~~~~~~~~~~~~

We follow the `Contributor Covenant Code of Conduct <https://www.contributor-covenant.org/>`_.
Be respectful and inclusive.

Getting Help
~~~~~~~~~~~~

- **Questions**: Use GitHub Discussions
- **Bugs**: Open an Issue with reproducible example
- **Features**: Discuss in Issue before implementing

Recognition
~~~~~~~~~~~

Contributors are recognized in:

- AUTHORS.md file
- Release notes
- Documentation credits

Thank you for contributing to SUBMARIT!