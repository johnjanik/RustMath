#!/usr/bin/env python3
"""
RustMath vs SymPy Benchmark Runner

Automated benchmark suite comparing RustMath and SymPy performance.
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Any, List

import sympy as sp
from sympy import symbols, diff, simplify, sin, cos, exp, expand


class BenchmarkRunner:
    """Runs and manages benchmarks comparing RustMath and SymPy"""

    def __init__(self, iterations: int = 1000, warmup: int = 10):
        self.iterations = iterations
        self.warmup = warmup
        self.results = []
        self.rustmath_bin = Path(__file__).parent.parent / 'target' / 'release'
        self.results_dir = Path(__file__).parent / 'results'

        # Create results directory if it doesn't exist
        self.results_dir.mkdir(exist_ok=True)

    def run_rustmath_benchmark(self, binary: str, test_name: str) -> Dict[str, Any]:
        """Run RustMath benchmark via subprocess"""
        cmd = [
            str(self.rustmath_bin / binary),
            '--test', test_name,
            '--iterations', str(self.iterations),
            '--json'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running {binary} {test_name}: {e}")
            print(f"stderr: {e.stderr}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from {binary} {test_name}: {e}")
            print(f"stdout: {result.stdout}")
            raise

    def benchmark_sympy(self, func: Callable) -> float:
        """Benchmark a SymPy operation"""
        # Warmup
        for _ in range(self.warmup):
            func()

        # Benchmark
        start = time.perf_counter()
        for _ in range(self.iterations):
            func()
        end = time.perf_counter()

        return (end - start) * 1000.0 / self.iterations

    def benchmark_test(
        self,
        name: str,
        category: str,
        rustmath_binary: str,
        rustmath_test: str,
        sympy_func: Callable
    ) -> Dict[str, Any]:
        """Compare RustMath vs SymPy for a single test"""
        print(f"Running: {name}...", end=" ", flush=True)

        # Run RustMath
        rustmath_result = self.run_rustmath_benchmark(rustmath_binary, rustmath_test)
        rustmath_time = rustmath_result['avg_time_ms']

        # Run SymPy
        sympy_time = self.benchmark_sympy(sympy_func)

        speedup = sympy_time / rustmath_time if rustmath_time > 0 else 0

        result = {
            'name': name,
            'category': category,
            'rustmath_binary': rustmath_binary,
            'rustmath_test': rustmath_test,
            'rustmath_ms': rustmath_time,
            'sympy_ms': sympy_time,
            'speedup': speedup,
            'iterations': self.iterations
        }

        self.results.append(result)
        print(f"✓ ({speedup:.1f}x speedup)")

        return result

    def run_all_benchmarks(self):
        """Run all defined benchmarks"""
        x = symbols('x')
        y = symbols('y')

        print("=" * 80)
        print("RustMath vs SymPy Benchmarks")
        print("=" * 80)
        print(f"Iterations: {self.iterations}")
        print(f"Warmup: {self.warmup}")
        print(f"SymPy version: {sp.__version__}")
        print("=" * 80)
        print()

        # Symbolic differentiation tests
        print("Category: Differentiation")
        print("-" * 80)

        self.benchmark_test(
            name="Polynomial d/dx (5x⁵ + 3x² - 7x + 2)",
            category="Differentiation",
            rustmath_binary="bench_symbolic",
            rustmath_test="diff_polynomial",
            sympy_func=lambda: diff(5*x**5 + 3*x**2 - 7*x + 2, x)
        )

        self.benchmark_test(
            name="Trig d/dx (sin(x) * cos(x))",
            category="Differentiation",
            rustmath_binary="bench_symbolic",
            rustmath_test="diff_trig",
            sympy_func=lambda: diff(sin(x) * cos(x), x)
        )

        self.benchmark_test(
            name="Nested d/dx (exp(sin(x²)))",
            category="Differentiation",
            rustmath_binary="bench_symbolic",
            rustmath_test="diff_nested",
            sympy_func=lambda: diff(exp(sin(x**2)), x)
        )

        self.benchmark_test(
            name="Product chain d/dx (x³ * sin(x) * exp(x))",
            category="Differentiation",
            rustmath_binary="bench_symbolic",
            rustmath_test="diff_product_chain",
            sympy_func=lambda: diff(x**3 * sin(x) * exp(x), x)
        )

        self.benchmark_test(
            name="High-order d^10/dx^10 (x^20)",
            category="Differentiation",
            rustmath_binary="bench_symbolic",
            rustmath_test="diff_high_order",
            sympy_func=lambda: diff(x**20, x, 10)
        )

        print()

        # Simplification tests
        print("Category: Simplification")
        print("-" * 80)

        self.benchmark_test(
            name="Simplify sin²(x) + cos²(x)",
            category="Simplification",
            rustmath_binary="bench_symbolic",
            rustmath_test="simplify_trig",
            sympy_func=lambda: simplify(sin(x)**2 + cos(x)**2)
        )

        self.benchmark_test(
            name="Expand (x + y)^10",
            category="Simplification",
            rustmath_binary="bench_symbolic",
            rustmath_test="expand_binomial",
            sympy_func=lambda: expand((x + y)**10)
        )

        self.benchmark_test(
            name="Simplify (x² - 1)/(x - 1)",
            category="Simplification",
            rustmath_binary="bench_symbolic",
            rustmath_test="simplify_rational",
            sympy_func=lambda: simplify((x**2 - 1) / (x - 1))
        )

        self.benchmark_test(
            name="Substitute x = 2 into x⁵ + 3x² - 7x + 2",
            category="Simplification",
            rustmath_binary="bench_symbolic",
            rustmath_test="substitution",
            sympy_func=lambda: (x**5 + 3*x**2 - 7*x + 2).subs(x, 2)
        )

        print()

    def print_summary(self):
        """Print benchmark results summary"""
        if not self.results:
            print("No results to display")
            return

        print()
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print()

        # Print table header
        print(f"{'Test Name':<45} {'SymPy (ms)':>12} {'RustMath (ms)':>14} {'Speedup':>8}")
        print("-" * 80)

        # Print results
        for result in self.results:
            name = result['name'][:44]  # Truncate if too long
            sympy_time = result['sympy_ms']
            rustmath_time = result['rustmath_ms']
            speedup = result['speedup']

            print(f"{name:<45} {sympy_time:>12.6f} {rustmath_time:>14.6f} {speedup:>7.1f}x")

        print("=" * 80)

        # Calculate statistics
        speedups = [r['speedup'] for r in self.results]
        avg_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)

        print()
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Min Speedup:     {min_speedup:.2f}x")
        print(f"Max Speedup:     {max_speedup:.2f}x")
        print(f"Tests Faster:    {sum(1 for s in speedups if s > 1)}/{len(speedups)}")
        print()

    def save_results(self, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        output_path = self.results_dir / filename

        # Calculate summary statistics
        speedups = [r['speedup'] for r in self.results]

        data = {
            'timestamp': datetime.now().isoformat(),
            'iterations': self.iterations,
            'warmup': self.warmup,
            'num_tests': len(self.results),
            'avg_speedup': sum(speedups) / len(speedups) if speedups else 0,
            'min_speedup': min(speedups) if speedups else 0,
            'max_speedup': max(speedups) if speedups else 0,
            'sympy_version': sp.__version__,
            'results': self.results
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Also save as latest.json
        latest_path = self.results_dir / 'latest.json'
        with open(latest_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Results saved to: {output_path}")
        print(f"✓ Latest results: {latest_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Run RustMath vs SymPy benchmarks')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of iterations per test (default: 1000)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: auto-generated)')

    args = parser.parse_args()

    # Create runner and execute benchmarks
    runner = BenchmarkRunner(iterations=args.iterations, warmup=args.warmup)

    try:
        runner.run_all_benchmarks()
        runner.print_summary()
        runner.save_results(args.output)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        if runner.results:
            print("Saving partial results...")
            runner.save_results("partial_results.json")
    except Exception as e:
        print(f"\nError during benchmarking: {e}")
        raise


if __name__ == '__main__':
    main()
