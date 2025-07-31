#!/usr/bin/env python
"""Comprehensive test runner for SUBMARIT test suite."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest


class TestRunner:
    """Run different categories of tests with reporting."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = {}
        self.start_time = time.time()
    
    def run_unit_tests(self):
        """Run unit tests."""
        print("\n" + "="*60)
        print("Running UNIT TESTS...")
        print("="*60)
        
        start = time.time()
        result = pytest.main([
            "tests/",
            "-v" if self.verbose else "-q",
            "-m", "not integration and not slow and not benchmark",
            "--tb=short"
        ])
        
        self.results["unit_tests"] = {
            "passed": result == 0,
            "duration": time.time() - start,
            "exit_code": result
        }
        
        return result == 0
    
    def run_integration_tests(self):
        """Run integration tests."""
        print("\n" + "="*60)
        print("Running INTEGRATION TESTS...")
        print("="*60)
        
        start = time.time()
        result = pytest.main([
            "tests/",
            "-v" if self.verbose else "-q",
            "-m", "integration",
            "--tb=short"
        ])
        
        self.results["integration_tests"] = {
            "passed": result == 0,
            "duration": time.time() - start,
            "exit_code": result
        }
        
        return result == 0
    
    def run_property_tests(self):
        """Run property-based tests."""
        print("\n" + "="*60)
        print("Running PROPERTY-BASED TESTS...")
        print("="*60)
        
        start = time.time()
        result = pytest.main([
            "tests/test_property_based.py",
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--hypothesis-show-statistics"
        ])
        
        self.results["property_tests"] = {
            "passed": result == 0,
            "duration": time.time() - start,
            "exit_code": result
        }
        
        return result == 0
    
    def run_regression_tests(self):
        """Run regression tests."""
        print("\n" + "="*60)
        print("Running REGRESSION TESTS...")
        print("="*60)
        
        start = time.time()
        result = pytest.main([
            "tests/test_regression.py",
            "-v" if self.verbose else "-q",
            "--tb=short"
        ])
        
        self.results["regression_tests"] = {
            "passed": result == 0,
            "duration": time.time() - start,
            "exit_code": result
        }
        
        return result == 0
    
    def run_edge_case_tests(self):
        """Run edge case tests."""
        print("\n" + "="*60)
        print("Running EDGE CASE TESTS...")
        print("="*60)
        
        start = time.time()
        result = pytest.main([
            "tests/test_edge_cases.py",
            "-v" if self.verbose else "-q",
            "--tb=short"
        ])
        
        self.results["edge_case_tests"] = {
            "passed": result == 0,
            "duration": time.time() - start,
            "exit_code": result
        }
        
        return result == 0
    
    def run_performance_tests(self):
        """Run performance benchmark tests."""
        print("\n" + "="*60)
        print("Running PERFORMANCE BENCHMARKS...")
        print("="*60)
        
        start = time.time()
        result = pytest.main([
            "tests/test_performance.py",
            "-v",  # Always verbose for benchmarks
            "-m", "benchmark",
            "--tb=short"
        ])
        
        self.results["performance_tests"] = {
            "passed": result == 0,
            "duration": time.time() - start,
            "exit_code": result
        }
        
        return result == 0
    
    def run_coverage_analysis(self):
        """Run test coverage analysis."""
        print("\n" + "="*60)
        print("Running COVERAGE ANALYSIS...")
        print("="*60)
        
        start = time.time()
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "--cov=submarit",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-report=json",
            "tests/",
            "-q"
        ], capture_output=True, text=True)
        
        # Parse coverage percentage
        coverage_pct = None
        for line in result.stdout.splitlines():
            if "TOTAL" in line:
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        coverage_pct = float(part.rstrip('%'))
                        break
        
        self.results["coverage"] = {
            "passed": result.returncode == 0 and (coverage_pct or 0) >= 80,
            "duration": time.time() - start,
            "exit_code": result.returncode,
            "coverage_percentage": coverage_pct
        }
        
        if coverage_pct:
            print(f"\nTotal coverage: {coverage_pct}%")
            if coverage_pct < 80:
                print("WARNING: Coverage below 80% target!")
        
        return result.returncode == 0
    
    def run_slow_tests(self):
        """Run slow tests (optional)."""
        print("\n" + "="*60)
        print("Running SLOW TESTS...")
        print("="*60)
        
        start = time.time()
        result = pytest.main([
            "tests/",
            "-v",
            "-m", "slow",
            "--tb=short"
        ])
        
        self.results["slow_tests"] = {
            "passed": result == 0,
            "duration": time.time() - start,
            "exit_code": result
        }
        
        return result == 0
    
    def generate_report(self):
        """Generate final test report."""
        total_duration = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("TEST SUITE SUMMARY")
        print("="*60)
        
        all_passed = True
        for test_type, result in self.results.items():
            status = "PASSED" if result["passed"] else "FAILED"
            duration = result["duration"]
            print(f"{test_type:.<40} {status} ({duration:.2f}s)")
            
            if test_type == "coverage" and result.get("coverage_percentage"):
                print(f"  Coverage: {result['coverage_percentage']}%")
            
            if not result["passed"]:
                all_passed = False
        
        print("-"*60)
        print(f"Total duration: {total_duration:.2f}s")
        print(f"Overall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        
        # Save results to JSON
        results_file = Path("test_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "results": self.results,
                "total_duration": total_duration,
                "all_passed": all_passed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Generate HTML report if coverage was run
        if "coverage" in self.results and self.results["coverage"]["passed"]:
            print("\nCoverage HTML report: htmlcov/index.html")
        
        return all_passed


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Run SUBMARIT test suite")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Verbose output")
    parser.add_argument("--unit", action="store_true",
                      help="Run only unit tests")
    parser.add_argument("--integration", action="store_true",
                      help="Run only integration tests")
    parser.add_argument("--property", action="store_true",
                      help="Run only property-based tests")
    parser.add_argument("--regression", action="store_true",
                      help="Run only regression tests")
    parser.add_argument("--edge", action="store_true",
                      help="Run only edge case tests")
    parser.add_argument("--performance", action="store_true",
                      help="Run only performance tests")
    parser.add_argument("--coverage", action="store_true",
                      help="Run only coverage analysis")
    parser.add_argument("--slow", action="store_true",
                      help="Include slow tests")
    parser.add_argument("--all", action="store_true",
                      help="Run all tests including slow ones")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(verbose=args.verbose)
    
    # Determine which tests to run
    if args.all or not any([args.unit, args.integration, args.property, 
                           args.regression, args.edge, args.performance, 
                           args.coverage]):
        # Run all tests
        tests_to_run = [
            ("unit", runner.run_unit_tests),
            ("integration", runner.run_integration_tests),
            ("property", runner.run_property_tests),
            ("regression", runner.run_regression_tests),
            ("edge", runner.run_edge_case_tests),
            ("coverage", runner.run_coverage_analysis),
        ]
        
        if args.all or args.performance:
            tests_to_run.append(("performance", runner.run_performance_tests))
        
        if args.all or args.slow:
            tests_to_run.append(("slow", runner.run_slow_tests))
    else:
        # Run selected tests
        tests_to_run = []
        if args.unit:
            tests_to_run.append(("unit", runner.run_unit_tests))
        if args.integration:
            tests_to_run.append(("integration", runner.run_integration_tests))
        if args.property:
            tests_to_run.append(("property", runner.run_property_tests))
        if args.regression:
            tests_to_run.append(("regression", runner.run_regression_tests))
        if args.edge:
            tests_to_run.append(("edge", runner.run_edge_case_tests))
        if args.performance:
            tests_to_run.append(("performance", runner.run_performance_tests))
        if args.coverage:
            tests_to_run.append(("coverage", runner.run_coverage_analysis))
        if args.slow:
            tests_to_run.append(("slow", runner.run_slow_tests))
    
    # Run selected tests
    for test_name, test_func in tests_to_run:
        try:
            test_func()
        except Exception as e:
            print(f"\nERROR running {test_name}: {e}")
            runner.results[test_name] = {
                "passed": False,
                "duration": 0,
                "exit_code": -1,
                "error": str(e)
            }
    
    # Generate report
    all_passed = runner.generate_report()
    
    # Return appropriate exit code
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())