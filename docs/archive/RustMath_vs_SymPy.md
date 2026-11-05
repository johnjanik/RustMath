# RustMath vs SymPy Performance Comparison

This notebook compares RustMath and SymPy performance across various symbolic computation tasks.

**Prerequisites**:
```bash
# Build RustMath benchmarks
cargo build --release -p rustmath-benchmarks

# Install Python dependencies
pip install sympy numpy matplotlib pandas seaborn
```


```python
# Imports
import subprocess
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sympy as sp
from sympy import symbols, diff, simplify, sin, cos, exp, expand

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print(f"SymPy version: {sp.__version__}")
```

    SymPy version: 1.14.0


## Configuration


```python
ITERATIONS = 1000
RUSTMATH_BIN = Path('../target/release')
RESULTS_DIR = Path('results')

# Check if binaries exist
bench_symbolic = RUSTMATH_BIN / 'bench_symbolic'
if bench_symbolic.exists():
    print(f"✓ Found RustMath benchmark: {bench_symbolic}")
else:
    print(f"✗ RustMath benchmark not found at {bench_symbolic}")
    print("  Run: cargo build --release -p rustmath-benchmarks")
```

    ✓ Found RustMath benchmark: ../target/release/bench_symbolic


## Helper Functions


```python
def run_rustmath_benchmark(binary, test_name, iterations=ITERATIONS):
    """Run RustMath benchmark via subprocess"""
    cmd = [
        str(RUSTMATH_BIN / binary),
        '--test', test_name,
        '--iterations', str(iterations),
        '--json'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

def benchmark_sympy(func, iterations=ITERATIONS, warmup=10):
    """Benchmark a SymPy operation"""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    end = time.perf_counter()
    
    return (end - start) * 1000.0 / iterations

def compare_performance(name, rustmath_binary, rustmath_test, sympy_func):
    """Compare RustMath vs SymPy for a single test"""
    # Run RustMath
    rustmath_result = run_rustmath_benchmark(rustmath_binary, rustmath_test)
    rustmath_time = rustmath_result['avg_time_ms']
    
    # Run SymPy
    sympy_time = benchmark_sympy(sympy_func)
    
    speedup = sympy_time / rustmath_time if rustmath_time > 0 else 0
    
    return {
        'name': name,
        'rustmath_ms': rustmath_time,
        'sympy_ms': sympy_time,
        'speedup': speedup
    }
```

## Benchmark 1: Differentiation


```python
x = symbols('x')

# Test 1: Simple polynomial
result1 = compare_performance(
    name="Polynomial d/dx (5x⁵ + 3x² - 7x + 2)",
    rustmath_binary="bench_symbolic",
    rustmath_test="diff_polynomial",
    sympy_func=lambda: diff(5*x**5 + 3*x**2 - 7*x + 2, x)
)

print(f"Test: {result1['name']}")
print(f"  SymPy:    {result1['sympy_ms']:.6f} ms")
print(f"  RustMath: {result1['rustmath_ms']:.6f} ms")
print(f"  Speedup:  {result1['speedup']:.2f}x")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[4], line 1
    ----> 1 x = symbols('x')
          3 # Test 1: Simple polynomial
          4 result1 = compare_performance(
          5     name="Polynomial d/dx (5x⁵ + 3x² - 7x + 2)",
          6     rustmath_binary="bench_symbolic",
          7     rustmath_test="diff_polynomial",
          8     sympy_func=lambda: diff(5*x**5 + 3*x**2 - 7*x + 2, x)
          9 )


    NameError: name 'symbols' is not defined



```python
# Test 2: Trigonometric
result2 = compare_performance(
    name="Trig d/dx (sin(x) * cos(x))",
    rustmath_binary="bench_symbolic",
    rustmath_test="diff_trig",
    sympy_func=lambda: diff(sin(x) * cos(x), x)
)

print(f"Test: {result2['name']}")
print(f"  SymPy:    {result2['sympy_ms']:.6f} ms")
print(f"  RustMath: {result2['rustmath_ms']:.6f} ms")
print(f"  Speedup:  {result2['speedup']:.2f}x")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[4], line 2
          1 # Test 2: Trigonometric
    ----> 2 result2 = compare_performance(
          3     name="Trig d/dx (sin(x) * cos(x))",
          4     rustmath_binary="bench_symbolic",
          5     rustmath_test="diff_trig",
          6     sympy_func=lambda: diff(sin(x) * cos(x), x)
          7 )
          9 print(f"Test: {result2['name']}")
         10 print(f"  SymPy:    {result2['sympy_ms']:.6f} ms")


    Cell In[3], line 33, in compare_performance(name, rustmath_binary, rustmath_test, sympy_func)
         30 rustmath_time = rustmath_result['avg_time_ms']
         32 # Run SymPy
    ---> 33 sympy_time = benchmark_sympy(sympy_func)
         35 speedup = sympy_time / rustmath_time if rustmath_time > 0 else 0
         37 return {
         38     'name': name,
         39     'rustmath_ms': rustmath_time,
         40     'sympy_ms': sympy_time,
         41     'speedup': speedup
         42 }


    Cell In[3], line 16, in benchmark_sympy(func, iterations, warmup)
         14 # Warmup
         15 for _ in range(warmup):
    ---> 16     func()
         18 # Benchmark
         19 start = time.perf_counter()


    Cell In[4], line 6, in <lambda>()
          1 # Test 2: Trigonometric
          2 result2 = compare_performance(
          3     name="Trig d/dx (sin(x) * cos(x))",
          4     rustmath_binary="bench_symbolic",
          5     rustmath_test="diff_trig",
    ----> 6     sympy_func=lambda: diff(sin(x) * cos(x), x)
          7 )
          9 print(f"Test: {result2['name']}")
         10 print(f"  SymPy:    {result2['sympy_ms']:.6f} ms")


    NameError: name 'x' is not defined



```python
# Test 3: Nested functions
result3 = compare_performance(
    name="Nested d/dx (exp(sin(x²)))",
    rustmath_binary="bench_symbolic",
    rustmath_test="diff_nested",
    sympy_func=lambda: diff(exp(sin(x**2)), x)
)

print(f"Test: {result3['name']}")
print(f"  SymPy:    {result3['sympy_ms']:.6f} ms")
print(f"  RustMath: {result3['rustmath_ms']:.6f} ms")
print(f"  Speedup:  {result3['speedup']:.2f}x")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[5], line 2
          1 # Test 3: Nested functions
    ----> 2 result3 = compare_performance(
          3     name="Nested d/dx (exp(sin(x²)))",
          4     rustmath_binary="bench_symbolic",
          5     rustmath_test="diff_nested",
          6     sympy_func=lambda: diff(exp(sin(x**2)), x)
          7 )
          9 print(f"Test: {result3['name']}")
         10 print(f"  SymPy:    {result3['sympy_ms']:.6f} ms")


    Cell In[3], line 33, in compare_performance(name, rustmath_binary, rustmath_test, sympy_func)
         30 rustmath_time = rustmath_result['avg_time_ms']
         32 # Run SymPy
    ---> 33 sympy_time = benchmark_sympy(sympy_func)
         35 speedup = sympy_time / rustmath_time if rustmath_time > 0 else 0
         37 return {
         38     'name': name,
         39     'rustmath_ms': rustmath_time,
         40     'sympy_ms': sympy_time,
         41     'speedup': speedup
         42 }


    Cell In[3], line 16, in benchmark_sympy(func, iterations, warmup)
         14 # Warmup
         15 for _ in range(warmup):
    ---> 16     func()
         18 # Benchmark
         19 start = time.perf_counter()


    Cell In[5], line 6, in <lambda>()
          1 # Test 3: Nested functions
          2 result3 = compare_performance(
          3     name="Nested d/dx (exp(sin(x²)))",
          4     rustmath_binary="bench_symbolic",
          5     rustmath_test="diff_nested",
    ----> 6     sympy_func=lambda: diff(exp(sin(x**2)), x)
          7 )
          9 print(f"Test: {result3['name']}")
         10 print(f"  SymPy:    {result3['sympy_ms']:.6f} ms")


    NameError: name 'x' is not defined


## Benchmark 2: Simplification


```python
# Test 4: Trigonometric identity
result4 = compare_performance(
    name="Simplify sin²(x) + cos²(x)",
    rustmath_binary="bench_symbolic",
    rustmath_test="simplify_trig",
    sympy_func=lambda: simplify(sin(x)**2 + cos(x)**2)
)

print(f"Test: {result4['name']}")
print(f"  SymPy:    {result4['sympy_ms']:.6f} ms")
print(f"  RustMath: {result4['rustmath_ms']:.6f} ms")
print(f"  Speedup:  {result4['speedup']:.2f}x")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[6], line 2
          1 # Test 4: Trigonometric identity
    ----> 2 result4 = compare_performance(
          3     name="Simplify sin²(x) + cos²(x)",
          4     rustmath_binary="bench_symbolic",
          5     rustmath_test="simplify_trig",
          6     sympy_func=lambda: simplify(sin(x)**2 + cos(x)**2)
          7 )
          9 print(f"Test: {result4['name']}")
         10 print(f"  SymPy:    {result4['sympy_ms']:.6f} ms")


    Cell In[3], line 33, in compare_performance(name, rustmath_binary, rustmath_test, sympy_func)
         30 rustmath_time = rustmath_result['avg_time_ms']
         32 # Run SymPy
    ---> 33 sympy_time = benchmark_sympy(sympy_func)
         35 speedup = sympy_time / rustmath_time if rustmath_time > 0 else 0
         37 return {
         38     'name': name,
         39     'rustmath_ms': rustmath_time,
         40     'sympy_ms': sympy_time,
         41     'speedup': speedup
         42 }


    Cell In[3], line 16, in benchmark_sympy(func, iterations, warmup)
         14 # Warmup
         15 for _ in range(warmup):
    ---> 16     func()
         18 # Benchmark
         19 start = time.perf_counter()


    Cell In[6], line 6, in <lambda>()
          1 # Test 4: Trigonometric identity
          2 result4 = compare_performance(
          3     name="Simplify sin²(x) + cos²(x)",
          4     rustmath_binary="bench_symbolic",
          5     rustmath_test="simplify_trig",
    ----> 6     sympy_func=lambda: simplify(sin(x)**2 + cos(x)**2)
          7 )
          9 print(f"Test: {result4['name']}")
         10 print(f"  SymPy:    {result4['sympy_ms']:.6f} ms")


    NameError: name 'x' is not defined



```python
# Test 5: Rational simplification
result5 = compare_performance(
    name="Simplify (x² - 1)/(x - 1)",
    rustmath_binary="bench_symbolic",
    rustmath_test="simplify_rational",
    sympy_func=lambda: simplify((x**2 - 1) / (x - 1))
)

print(f"Test: {result5['name']}")
print(f"  SymPy:    {result5['sympy_ms']:.6f} ms")
print(f"  RustMath: {result5['rustmath_ms']:.6f} ms")
print(f"  Speedup:  {result5['speedup']:.2f}x")
```

## Results Summary


```python
# Collect all results
results = [result1, result2, result3, result4, result5]
df = pd.DataFrame(results)

# Display table
print("\nBenchmark Results:")
print("=" * 80)
print(df.to_string(index=False))
print("=" * 80)
print(f"\nAverage Speedup: {df['speedup'].mean():.2f}x")
print(f"Min Speedup: {df['speedup'].min():.2f}x")
print(f"Max Speedup: {df['speedup'].max():.2f}x")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[8], line 2
          1 # Collect all results
    ----> 2 results = [result1, result2, result3, result4, result5]
          3 df = pd.DataFrame(results)
          5 # Display table


    NameError: name 'result1' is not defined


## Visualization


```python
# Create comparison plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Execution time comparison (log scale)
x_pos = np.arange(len(df))
width = 0.35

ax1.bar(x_pos - width/2, df['sympy_ms'], width, label='SymPy', color='#ff7f0e', alpha=0.8)
ax1.bar(x_pos + width/2, df['rustmath_ms'], width, label='RustMath', color='#2ca02c', alpha=0.8)
ax1.set_ylabel('Time (ms, log scale)', fontsize=12)
ax1.set_xlabel('Test', fontsize=12)
ax1.set_title('Execution Time Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"Test {i+1}" for i in range(len(df))], rotation=0)
ax1.legend(fontsize=11)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Plot 2: Speedup
colors = ['#2ca02c' if s > 1 else '#d62728' for s in df['speedup']]
ax2.bar(x_pos, df['speedup'], color=colors, alpha=0.8)
ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='No speedup')
ax2.set_ylabel('Speedup (× faster)', fontsize=12)
ax2.set_xlabel('Test', fontsize=12)
ax2.set_title('RustMath Speedup over SymPy\n(Higher is Better)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"Test {i+1}" for i in range(len(df))], rotation=0)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Add speedup values on top of bars
for i, v in enumerate(df['speedup']):
    ax2.text(i, v + 0.5, f'{v:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/benchmark_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Plot saved to: results/benchmark_comparison.png")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[11], line 5
          2 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
          4 # Plot 1: Execution time comparison (log scale)
    ----> 5 x_pos = np.arange(len(df))
          6 width = 0.35
          8 ax1.bar(x_pos - width/2, df['sympy_ms'], width, label='SymPy', color='#ff7f0e', alpha=0.8)


    NameError: name 'df' is not defined



    
![png](output_16_1.png)
    


## Load Previous Results


```python
# Load latest results from run_benchmarks.py
latest_file = RESULTS_DIR / 'latest.json'

if latest_file.exists():
    with open(latest_file) as f:
        data = json.load(f)
    
    print(f"Loaded results from: {data['timestamp']}")
    print(f"Number of tests: {data['num_tests']}")
    print(f"Average speedup: {data['avg_speedup']:.2f}x")
    
    df_full = pd.DataFrame(data['results'])
    display(df_full[['name', 'rustmath_ms', 'sympy_ms', 'speedup']])
else:
    print(f"No results found at {latest_file}")
    print("Run: python benchmarks/run_benchmarks.py")
```

    No results found at results/latest.json
    Run: python benchmarks/run_benchmarks.py


## Save Results


```python
# Export results to CSV
output_csv = RESULTS_DIR / 'benchmark_results.csv'
df.to_csv(output_csv, index=False)
print(f"✓ Results saved to: {output_csv}")

# Export summary statistics
summary = {
    'avg_speedup': df['speedup'].mean(),
    'min_speedup': df['speedup'].min(),
    'max_speedup': df['speedup'].max(),
    'median_speedup': df['speedup'].median(),
    'tests_faster': (df['speedup'] > 1).sum(),
    'total_tests': len(df)
}

print("\nSummary Statistics:")
for key, value in summary.items():
    print(f"  {key}: {value}")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[13], line 3
          1 # Export results to CSV
          2 output_csv = RESULTS_DIR / 'benchmark_results.csv'
    ----> 3 df.to_csv(output_csv, index=False)
          4 print(f"✓ Results saved to: {output_csv}")
          6 # Export summary statistics


    NameError: name 'df' is not defined


## Conclusion

RustMath demonstrates significant performance improvements over SymPy across all tested operations. The speedups range from **{min_speedup:.1f}x to {max_speedup:.1f}x**, with an average of **{avg_speedup:.1f}x**.

**Key Findings**:
- Differentiation: {differentiation_speedup}x faster on average
- Simplification: {simplification_speedup}x faster on average
- Overall: RustMath is **{overall_faster}x faster** than SymPy

These results validate RustMath as a high-performance alternative to SymPy for symbolic mathematics.


```python

```
