SUBMARIT Documentation
======================

Welcome to SUBMARIT's documentation!

.. note::

   The canonical documentation for SUBMARIT is hosted on GitHub Pages at:
   https://m-marinucci.github.io/SubmarketIdentificationTesting/
   
   This ensures documentation stays in sync with the latest code changes.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   api
   algorithms
   performance

.. toctree::
   :maxdepth: 2
   :caption: Resources:

   migration_guide
   faq
   contributing

Overview
--------

SUBMARIT (SUBMARket Identification and Testing) is a Python package for identifying
and analyzing submarkets based on product substitution patterns. It provides:

* **Efficient clustering algorithms** - State-of-the-art local search with multiple optimization strategies
* **Statistical evaluation methods** - Comprehensive metrics for assessing submarket quality
* **Validation techniques** - Cross-validation, stability testing, and bootstrap methods
* **MATLAB compatibility** - Seamless migration from MATLAB implementations
* **Performance optimization** - Support for large-scale datasets with GPU and distributed computing
* **Real-world applications** - Ready for production use in retail, e-commerce, and market research

Key Features
~~~~~~~~~~~~

* **Scalable**: Handle datasets from 100 to 1,000,000+ products
* **Fast**: Optimized implementations with parallel processing
* **Flexible**: Multiple algorithms and customization options
* **Validated**: Extensive testing and benchmarking
* **Well-documented**: Comprehensive guides and API reference
* **Production-ready**: Cloud deployment and monitoring tools

Use Cases
~~~~~~~~~

SUBMARIT is ideal for:

* **Retail Analytics**: Understanding product competition and substitution
* **Pricing Strategy**: Identifying products that compete on price
* **Inventory Management**: Grouping substitutable products for stock optimization
* **Market Research**: Analyzing market structure and competition
* **Recommendation Systems**: Finding substitute products for out-of-stock items

Getting Help
~~~~~~~~~~~~

* **Installation Issues**: See the :doc:`installation` guide
* **Quick Tutorial**: Start with the :doc:`quickstart` guide
* **API Details**: Browse the :doc:`api` reference
* **Performance**: Check the :doc:`performance` guide
* **Questions**: See the :doc:`faq` or file an issue on GitHub

Credits and Acknowledgments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This Python implementation is based on the original MATLAB SUBMARIT package.

**Original MATLAB Implementation**

* Stephen France, Mississippi State University (RandIndex4.m, 2012)
* Additional contributors (names unknown)

**Academic Foundations**

The SUBMARIT methodology is based on submarket identification research from marketing science:

* Rand, W.M. (1971) - Objective criteria for the evaluation of clustering methods
* Hubert, L. and Arabie, P. (1985) - Comparing partitions (Adjusted Rand Index)
* Urban, G.L., Johnson, P.L., and Hauser, J.R. - Market structure analysis
* Tibshirani, R., Walther, G., and Hastie, T. (2001) - Estimating the number of clusters via the gap statistic

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`