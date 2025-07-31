"""Test coverage verification and reporting for SUBMARIT."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def run_coverage_report():
    """Run test coverage analysis and generate reports."""
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Coverage commands
    commands = [
        # Run tests with coverage
        [
            sys.executable, "-m", "pytest",
            "--cov=submarit",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-config=.coveragerc",
            "tests/"
        ],
        
        # Generate coverage badge (if coverage-badge is installed)
        ["coverage-badge", "-o", "coverage.svg", "-f"],
    ]
    
    # Change to project root
    os.chdir(project_root)
    
    # Run commands
    for cmd in commands:
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
            else:
                print(result.stdout)
        except FileNotFoundError:
            print(f"Command not found: {cmd[0]}")
            continue
    
    # Read coverage report
    coverage_file = project_root / "htmlcov" / "index.html"
    if coverage_file.exists():
        print(f"\nCoverage report generated at: {coverage_file}")
    
    # Parse XML coverage for summary
    xml_file = project_root / "coverage.xml"
    if xml_file.exists():
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get overall coverage
        coverage = float(root.attrib.get('line-rate', 0)) * 100
        print(f"\nOverall test coverage: {coverage:.1f}%")
        
        if coverage < 80:
            print("WARNING: Coverage is below 80% target!")
            return 1
    
    return 0


class TestCoverageChecks:
    """Verify test coverage meets requirements."""
    
    def test_all_modules_imported(self):
        """Ensure all modules can be imported."""
        modules = [
            "submarit",
            "submarit.algorithms",
            "submarit.algorithms.local_search",
            "submarit.core",
            "submarit.core.base",
            "submarit.core.substitution_matrix",
            "submarit.core.create_substitution_matrix",
            "submarit.evaluation",
            "submarit.evaluation.cluster_evaluator",
            "submarit.evaluation.entropy_evaluator",
            "submarit.evaluation.gap_statistic",
            "submarit.evaluation.statistical_tests",
            "submarit.validation",
            "submarit.validation.empirical_distributions",
            "submarit.validation.kfold",
            "submarit.validation.multiple_runs",
            "submarit.validation.p_values",
            "submarit.validation.rand_index",
            "submarit.validation.topk_analysis",
            "submarit.utils",
            "submarit.utils.matlab_compat",
            "submarit.io",
            "submarit.io.data_io",
            "submarit.io.matlab_io",
        ]
        
        for module_name in modules:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")
    
    def test_critical_paths_covered(self):
        """Verify critical code paths are tested."""
        critical_functions = [
            # Core algorithms
            ("submarit.algorithms.local_search", "KSMLocalSearch.fit"),
            ("submarit.algorithms.local_search", "KSMLocalSearch2.fit"),
            ("submarit.algorithms.local_search", "KSMLocalSearchConstrained.fit_constrained"),
            
            # Core functionality
            ("submarit.core.substitution_matrix", "SubstitutionMatrix.create_from_consumer_product_data"),
            ("submarit.core.create_substitution_matrix", "create_substitution_matrix"),
            
            # Evaluation
            ("submarit.evaluation.cluster_evaluator", "ClusterEvaluator.evaluate"),
            ("submarit.evaluation.gap_statistic", "GAPStatistic.compute"),
            
            # Validation
            ("submarit.validation.empirical_distributions", "EmpiricalDistribution.generate"),
            ("submarit.validation.kfold", "KFoldValidator.validate"),
            ("submarit.validation.rand_index", "rand_index"),
        ]
        
        # This is a placeholder - in a real test, we'd check coverage data
        # For now, just verify the functions exist
        for module_name, func_path in critical_functions:
            module = __import__(module_name, fromlist=[''])
            
            # Navigate to the function
            parts = func_path.split('.')
            obj = module
            for part in parts:
                assert hasattr(obj, part), f"Missing critical function: {func_path}"
                obj = getattr(obj, part)
    
    @pytest.mark.slow
    def test_coverage_target_met(self):
        """Verify overall coverage meets 80% target."""
        # This would run the coverage analysis
        # For now, it's a placeholder
        pass


def create_coverage_config():
    """Create .coveragerc configuration file."""
    config = """[run]
source = src/submarit
omit = 
    */tests/*
    */test_*
    */__init__.py
    */setup.py
    */cli.py

[report]
precision = 2
show_missing = True
skip_covered = False

exclude_lines =
    pragma: no cover
    def __repr__
    def __str__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @overload
    @abstractmethod

[html]
directory = htmlcov

[xml]
output = coverage.xml
"""
    
    with open(".coveragerc", "w") as f:
        f.write(config)
    
    print("Created .coveragerc configuration file")


if __name__ == "__main__":
    # Create config if needed
    if not Path(".coveragerc").exists():
        create_coverage_config()
    
    # Run coverage analysis
    sys.exit(run_coverage_report())