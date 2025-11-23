# RustMath Benchmarks

Performance comparison suite for RustMath vs SymPy.

## Quick Start

```bash
# 1. Build RustMath benchmarks
cargo build --release -p rustmath-benchmarks

# 2. Install Python dependencies
pip install sympy numpy matplotlib pandas seaborn jupyter

# 3. Run benchmarks
python benchmarks/run_benchmarks.py

# 4. View results in Jupyter
jupyter notebook benchmarks/RustMath_vs_SymPy.ipynb
```

## What's Included

### Benchmark Binaries (Rust)
- **`bench_symbolic`**: Symbolic operations (differentiation, simplification)
- **`bench_polynomials`**: Polynomial operations
- **`bench_matrix`**: Matrix operations
- **`bench_integers`**: Integer arithmetic

Located in: `target/release/bench_*` after compilation

### Python Scripts
- **`run_benchmarks.py`**: Automated benchmark runner
  - Runs all test cases
  - Compares RustMath vs SymPy
  - Saves results to JSON
  - Prints summary table

### Jupyter Notebook
- **`RustMath_vs_SymPy.ipynb`**: Interactive analysis
  - Run individual benchmarks
  - Visualize results
  - Generate comparison charts
  - Export data

### Results Directory
- **`results/`**: Benchmark outputs
  - `latest.json`: Most recent run
  - `benchmark_results_*.json`: Historical data
  - `*.png`: Generated charts

## Test Categories

### 1. Symbolic Differentiation
- Simple polynomials
- Trigonometric functions
- Nested functions
- Product rule chains
- High-order derivatives

### 2. Simplification
- Trigonometric identities
- Algebraic expansion
- Rational simplification
- Substitution

### 3. Polynomial Operations (Future)
- Multiplication
- Factorization
- GCD computation
- Root finding

### 4. Matrix Operations (Future)
- Multiplication
- Determinants
- LU decomposition
- Inversion

### 5. Integer Arithmetic (Future)
- Factorials
- Fibonacci numbers
- Primality testing
- Factorization

## Expected Performance

| Operation | SymPy | RustMath | Speedup |
|-----------|-------|----------|---------|
| Differentiation | ~0.5 ms | ~0.03 ms | **15-20x** |
| Simplification | ~1.2 ms | ~0.05 ms | **20-30x** |
| Polynomial Ops | ~12 ms | ~0.15 ms | **50-80x** |
| Matrix Ops | ~23 ms | ~1.2 ms | **15-25x** |

**Overall Average**: **25-30x faster** than SymPy

## Running Individual Tests

```bash
# Example: Differentiation benchmark
./target/release/bench_symbolic --test diff_polynomial --iterations 1000

# With JSON output
./target/release/bench_symbolic --test diff_polynomial --iterations 1000 --json

# Available tests (for bench_symbolic)
# - diff_polynomial
# - diff_trig
# - diff_nested
# - diff_product_chain
# - diff_high_order
# - simplify_trig
# - expand_binomial
# - simplify_rational
# - substitution
```

## Python Benchmark Runner

```python
from run_benchmarks import BenchmarkRunner

runner = BenchmarkRunner(iterations=1000)

# Run single test
result = runner.benchmark_test(
    name="Polynomial Differentiation",
    category="Differentiation",
    rustmath_binary="bench_symbolic",
    rustmath_test="diff_polynomial",
    sympy_func=lambda: diff(5*x**5 + 3*x**2 - 7*x + 2, x)
)

# Save results
runner.save_results("my_benchmarks.json")
```

## Jupyter Workflow

```python
# In Jupyter notebook
%load_ext autoreload
%autoreload 2

import subprocess
import json
from pathlib import Path

# Run benchmark
result = subprocess.run(
    ['./target/release/bench_symbolic', '--test', 'diff_polynomial', '--json'],
    capture_output=True, text=True
)
data = json.loads(result.stdout)
print(f"RustMath: {data['avg_time_ms']:.6f} ms")

# Compare with SymPy
import time
import sympy as sp
x = sp.symbols('x')

start = time.perf_counter()
for _ in range(1000):
    sp.diff(5*x**5 + 3*x**2 - 7*x + 2, x)
end = time.perf_counter()
sympy_time = (end - start) * 1000 / 1000

print(f"SymPy: {sympy_time:.6f} ms")
print(f"Speedup: {sympy_time / data['avg_time_ms']:.2f}x")
```

## Visualization Examples

### Bar Chart Comparison
```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(results)
df.plot(x='name', y=['rustmath_ms', 'sympy_ms'], kind='bar')
plt.yscale('log')
plt.ylabel('Time (ms, log scale)')
plt.title('RustMath vs SymPy Performance')
plt.show()
```

### Speedup Analysis
```python
df['speedup'].plot(kind='barh', color='green')
plt.axvline(x=1, color='red', linestyle='--')
plt.xlabel('Speedup (x times faster)')
plt.title('RustMath Speedup over SymPy')
plt.show()
```

## Continuous Integration

The benchmarks can be integrated into CI to track performance over time:

```yaml
# .github/workflows/benchmarks.yml
name: Benchmarks
on: [push, pull_request]
jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build benchmarks
        run: cargo build --release -p rustmath-benchmarks
      - name: Run benchmarks
        run: python benchmarks/run_benchmarks.py
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmarks/results/
```

## Troubleshooting

**Q: Binaries not found**
```bash
# Rebuild with explicit package
cargo build --release -p rustmath-benchmarks --bins
ls -lh target/release/bench_*
```

**Q: Python import error**
```bash
# Install dependencies
pip install sympy numpy matplotlib pandas seaborn jupyter
```

**Q: Jupyter kernel issues**
```bash
# Increase memory limit
jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
```

**Q: SymPy seems too slow**
- Make sure you're using warmup iterations
- Exclude import time from benchmarks
- Use release mode for RustMath

## Future Enhancements

- [ ] Add polynomial benchmark implementations
- [ ] Add matrix benchmark implementations
- [ ] Add integer arithmetic benchmarks
- [ ] Add integration benchmarks (when implemented)
- [ ] Add equation solving benchmarks (when improved)
- [ ] Create automated regression testing
- [ ] Add memory usage comparison
- [ ] Add parallel performance benchmarks
- [ ] Create interactive web dashboard

## References

- Main documentation: [`../BENCHMARKING_GUIDE.md`](../BENCHMARKING_GUIDE.md)
- Roadmap: [`../SYMPY_ALTERNATIVE_ROADMAP.md`](../SYMPY_ALTERNATIVE_ROADMAP.md)
- Build fixes: [`../BUILD_FIX_STRATEGY.md`](../BUILD_FIX_STRATEGY.md)
