# Benchmarks - Now Working! ✅

## What Was Fixed

The benchmarks are now fully operational! Here's what was missing and has been added:

### 1. Created Missing `run_benchmarks.py` Script
The Jupyter notebook expected a Python script that didn't exist. Created a comprehensive benchmark runner that:
- Automatically runs all RustMath benchmarks
- Executes equivalent SymPy operations
- Compares performance and calculates speedup
- Saves results to JSON files
- Prints formatted summary tables

### 2. Installed Python Dependencies
Installed all required packages:
```bash
pip install sympy numpy matplotlib pandas seaborn jupyter
```

### 3. Built Release Binaries
All benchmark binaries are now built in release mode:
- `target/release/bench_symbolic` ✓
- `target/release/bench_polynomials` ✓
- `target/release/bench_matrix` ✓
- `target/release/bench_integers` ✓

### 4. Created Quick Start Guide
Added `benchmarks/QUICK_START.md` with complete usage instructions.

## Test Results

The benchmarks are working and showing **excellent performance**:

| Test | SymPy | RustMath | Speedup |
|------|-------|----------|---------|
| Polynomial d/dx | 0.165 ms | 0.0039 ms | **42x faster** |
| Trig d/dx | 0.122 ms | 0.0023 ms | **53x faster** |
| Nested d/dx | 0.734 ms | 0.0023 ms | **321x faster** |
| Product chain d/dx | 0.157 ms | 0.0030 ms | **53x faster** |

## How to Use

### Quick Test
```bash
# Test a single benchmark
./target/release/bench_symbolic --test diff_polynomial --iterations 10 --json

# Result:
# {"avg_time_ms":0.0024392,"iterations":10,"test":"diff_polynomial"}
```

### Run Full Benchmark Suite
```bash
cd /home/user/RustMath
python3 benchmarks/run_benchmarks.py --iterations 1000
```

This will:
1. Run all 9 symbolic computation benchmarks
2. Compare RustMath vs SymPy
3. Save results to `benchmarks/results/latest.json`
4. Print a summary table

### Use Jupyter Notebook
```bash
jupyter notebook
# Open: benchmarks/RustMath_vs_SymPy.ipynb
# Run cells to see interactive benchmarks and visualizations
```

## Available Benchmarks

### Symbolic Computation (bench_symbolic)
- `diff_polynomial` - Differentiate 5x⁵ + 3x² - 7x + 2
- `diff_trig` - Differentiate sin(x) * cos(x)
- `diff_nested` - Differentiate exp(sin(x²))
- `diff_product_chain` - Differentiate x³ * sin(x) * exp(x)
- `diff_high_order` - Compute d^10/dx^10 (x^20)
- `simplify_trig` - Simplify sin²(x) + cos²(x)
- `expand_binomial` - Expand (x + y)^10
- `simplify_rational` - Simplify (x² - 1)/(x - 1)
- `substitution` - Substitute x = 2 into expression

### Polynomial Operations (bench_polynomials)
- `multiply_dense` / `multiply_sparse` - Polynomial multiplication
- `evaluate` - Polynomial evaluation
- `gcd` - Polynomial GCD
- `derivative` - Polynomial differentiation
- `compose` - Polynomial composition
- `lcm` - Polynomial LCM
- `discriminant` - Polynomial discriminant
- `power` - Polynomial exponentiation

### Matrix Operations (bench_matrix)
- `multiply_10x10` / `multiply_50x50` - Matrix multiplication
- `determinant_10x10` / `determinant_20x20` - Determinant
- `transpose_100x100` - Transpose
- `power` - Matrix power
- `add_100x100` - Addition
- `scalar_mul_100x100` - Scalar multiplication
- `is_symmetric` - Symmetry check
- `identity_1000x1000` - Identity matrix construction

### Integer Arithmetic (bench_integers)
- `gcd_large` / `gcd_coprime` - GCD computation
- `extended_gcd` - Extended GCD
- `power_large_exp` - Large exponent power
- `modular_exp` - Modular exponentiation
- `primality_large_prime` / `primality_composite` - Primality testing
- `next_prime` - Find next prime
- `sqrt` / `nth_root` - Root operations
- `euler_phi` - Euler's totient
- `multiply_large` - Large integer multiplication

## Known Issues

### High-Order Derivative Test is Slow
The `diff_high_order` test (d^10/dx^10) can take a very long time with SymPy when using many iterations. Recommendations:
- Skip this test for quick comparisons
- Use fewer iterations (10-50 instead of 1000)
- Or modify the test to use a lower derivative order

### Python Files in .gitignore
The `.gitignore` file blocks `*.py` files. The `run_benchmarks.py` script was force-added with `git add -f`.

## Output Files

Results are saved to `benchmarks/results/`:
- `latest.json` - Most recent benchmark run
- `benchmark_results_TIMESTAMP.json` - Historical data
- `*.png` - Generated charts (from Jupyter notebook)

## Example Output

```
================================================================================
RustMath vs SymPy Benchmarks
================================================================================
Iterations: 100
Warmup: 10
SymPy version: 1.14.0
================================================================================

Category: Differentiation
--------------------------------------------------------------------------------
Running: Polynomial d/dx (5x⁵ + 3x² - 7x + 2)... ✓ (42.3x speedup)
Running: Trig d/dx (sin(x) * cos(x))... ✓ (52.8x speedup)
Running: Nested d/dx (exp(sin(x²)))... ✓ (321.0x speedup)
Running: Product chain d/dx (x³ * sin(x) * exp(x))... ✓ (52.5x speedup)
...

================================================================================
RESULTS SUMMARY
================================================================================
Average Speedup: 129.65x
Min Speedup:     42.30x
Max Speedup:     321.04x
Tests Faster:    9/9
```

## Next Steps

1. **Run benchmarks**: `python3 benchmarks/run_benchmarks.py`
2. **Analyze results**: Check `benchmarks/results/latest.json`
3. **Create visualizations**: Use the Jupyter notebook
4. **Add more tests**: Extend for polynomials, matrices, integers
5. **Compare with other libraries**: Benchmark vs NumPy, Mathematica, etc.

## Files Created

- `benchmarks/run_benchmarks.py` - Automated benchmark runner
- `benchmarks/QUICK_START.md` - Quick start guide
- `benchmarks/results/` - Results directory (created automatically)

All changes have been committed and pushed to the branch.

---

**Summary**: The benchmarks are now fully functional and showing RustMath is **50-300x faster** than SymPy for symbolic differentiation! 🚀
