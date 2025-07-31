Installation Guide
==================

This guide covers the installation of SUBMARIT for different use cases and platforms.

Requirements
------------

* Python 3.8 or higher
* NumPy >= 1.20.0
* SciPy >= 1.7.0
* pandas >= 1.3.0
* scikit-learn >= 0.24.0
* matplotlib >= 3.3.0
* seaborn >= 0.11.0

Quick Install
-------------

The simplest way to install SUBMARIT is using pip:

.. code-block:: bash

    pip install submarit

For the latest development version:

.. code-block:: bash

    pip install git+https://github.com/yourusername/submarit.git

Installation from Source
------------------------

Clone the repository and install in development mode:

.. code-block:: bash

    git clone https://github.com/yourusername/submarit.git
    cd submarit
    pip install -e .[dev]

This installs the package in editable mode along with development dependencies.

Platform-Specific Instructions
------------------------------

Windows
~~~~~~~

On Windows, you may need to install Visual C++ Build Tools for some dependencies:

1. Download and install `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_
2. Install SUBMARIT using pip

macOS
~~~~~

On macOS, ensure you have Xcode Command Line Tools installed:

.. code-block:: bash

    xcode-select --install
    pip install submarit

Linux
~~~~~

Most Linux distributions come with the necessary build tools. If not:

.. code-block:: bash

    # Ubuntu/Debian
    sudo apt-get install python3-dev build-essential
    
    # Fedora/RedHat
    sudo dnf install python3-devel gcc

Conda Environment
-----------------

For scientific computing, we recommend using conda:

.. code-block:: bash

    conda create -n submarit python=3.9
    conda activate submarit
    conda install numpy scipy pandas scikit-learn matplotlib seaborn
    pip install submarit

Optional Dependencies
---------------------

For enhanced functionality:

.. code-block:: bash

    # For Jupyter notebook support
    pip install jupyterlab ipywidgets
    
    # For parallel processing
    pip install joblib dask
    
    # For advanced visualization
    pip install plotly bokeh
    
    # For MATLAB compatibility
    pip install matlab.engine  # Requires MATLAB installation

Verification
------------

Verify your installation:

.. code-block:: python

    import submarit
    print(submarit.__version__)
    
    # Run a simple test
    from submarit.core import create_substitution_matrix
    import numpy as np
    
    # Create sample data
    X = np.random.rand(100, 10)
    S = create_substitution_matrix(X)
    print(f"Substitution matrix shape: {S.shape}")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **ImportError: No module named 'submarit'**
   
   Ensure the package is installed in the correct environment:
   
   .. code-block:: bash
   
       pip list | grep submarit

2. **NumPy/SciPy build errors**
   
   Install pre-built wheels:
   
   .. code-block:: bash
   
       pip install --only-binary :all: numpy scipy

3. **Memory errors with large datasets**
   
   Consider installing with memory-efficient options:
   
   .. code-block:: bash
   
       pip install submarit[sparse]

4. **MATLAB compatibility issues**
   
   Ensure MATLAB Engine API for Python is properly installed:
   
   .. code-block:: bash
   
       cd "matlabroot/extern/engines/python"
       python setup.py install

Getting Help
------------

If you encounter issues:

1. Check the `FAQ <faq.html>`_
2. Search `GitHub Issues <https://github.com/yourusername/submarit/issues>`_
3. Ask on `Stack Overflow <https://stackoverflow.com/questions/tagged/submarit>`_ with tag 'submarit'
4. Contact the maintainers