//! Benchmarks for PowComputer and PowComputerExt
//!
//! These benchmarks measure the performance improvements from caching
//! powers of primes in p-adic arithmetic.
//!
//! Run with: `cargo bench --bench pow_computer_bench`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rustmath_integers::Integer;
use rustmath_rings::padics::{PowComputer, PowComputerExt};

/// Benchmark: Create PowComputer with various cache sizes
fn bench_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pow_computer_creation");

    for cache_limit in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("cache_limit", cache_limit),
            cache_limit,
            |b, &limit| {
                b.iter(|| {
                    let pc = PowComputer::new(black_box(Integer::from(5)), black_box(limit));
                    black_box(pc);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Accessing cached vs uncached powers
fn bench_pow_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("pow_access");

    let pc_small = PowComputer::new(Integer::from(5), 10);
    let pc_large = PowComputer::new(Integer::from(5), 1000);

    // Cached access (within cache limit)
    group.bench_function("cached_small_exponent", |b| {
        b.iter(|| {
            let result = pc_large.pow(black_box(5));
            black_box(result);
        });
    });

    group.bench_function("cached_large_exponent", |b| {
        b.iter(|| {
            let result = pc_large.pow(black_box(500));
            black_box(result);
        });
    });

    // Uncached access (beyond cache limit)
    group.bench_function("uncached_moderate", |b| {
        b.iter(|| {
            let result = pc_small.pow(black_box(50));
            black_box(result);
        });
    });

    group.bench_function("uncached_large", |b| {
        b.iter(|| {
            let result = pc_small.pow(black_box(500));
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark: Modular reduction with cached powers
fn bench_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("modular_reduction");

    let pc = PowComputer::new(Integer::from(5), 100);

    // Different value sizes
    for magnitude in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("reduce_magnitude", magnitude),
            magnitude,
            |b, &mag| {
                let value = Integer::from(mag);
                b.iter(|| {
                    let result = pc.reduce(black_box(value.clone()), black_box(20));
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: In-place vs regular reduction
fn bench_reduction_inplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_comparison");

    let pc = PowComputer::new(Integer::from(7), 100);
    let value = Integer::from(123456789);

    group.bench_function("reduce_copy", |b| {
        b.iter(|| {
            let result = pc.reduce(black_box(value.clone()), black_box(30));
            black_box(result);
        });
    });

    group.bench_function("reduce_in_place", |b| {
        b.iter(|| {
            let mut val = value.clone();
            pc.reduce_in_place(black_box(&mut val), black_box(30));
            black_box(val);
        });
    });

    group.finish();
}

/// Benchmark: Valuation computation
fn bench_valuation(c: &mut Criterion) {
    let mut group = c.benchmark_group("valuation");

    let pc = PowComputer::new(Integer::from(5), 100);

    group.bench_function("valuation_small", |b| {
        let value = Integer::from(625); // 5^4
        b.iter(|| {
            let result = pc.valuation(black_box(&value));
            black_box(result);
        });
    });

    group.bench_function("valuation_large", |b| {
        let value = Integer::from(5).pow(50);
        b.iter(|| {
            let result = pc.valuation(black_box(&value));
            black_box(result);
        });
    });

    group.bench_function("valuation_coprime", |b| {
        let value = Integer::from(123456789);
        b.iter(|| {
            let result = pc.valuation(black_box(&value));
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark: Extension creation
fn bench_extension_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("extension_creation");

    group.bench_function("unramified_small", |b| {
        b.iter(|| {
            let pc = PowComputerExt::unramified(
                black_box(Integer::from(5)),
                black_box(50),
                black_box(3),
            );
            black_box(pc);
        });
    });

    group.bench_function("unramified_large", |b| {
        b.iter(|| {
            let pc = PowComputerExt::unramified(
                black_box(Integer::from(5)),
                black_box(500),
                black_box(10),
            );
            black_box(pc);
        });
    });

    group.bench_function("eisenstein", |b| {
        b.iter(|| {
            let pc = PowComputerExt::eisenstein(
                black_box(Integer::from(7)),
                black_box(100),
                black_box(5),
            );
            black_box(pc);
        });
    });

    group.bench_function("general", |b| {
        b.iter(|| {
            let pc = PowComputerExt::general(
                black_box(Integer::from(3)),
                black_box(100),
                black_box(3),
                black_box(4),
            );
            black_box(pc);
        });
    });

    group.finish();
}

/// Benchmark: Frobenius computations
fn bench_frobenius(c: &mut Criterion) {
    let mut group = c.benchmark_group("frobenius");

    let pc = PowComputerExt::unramified(Integer::from(5), 100, 10);

    group.bench_function("frobenius_exponent", |b| {
        b.iter(|| {
            let result = pc.frobenius_exponent(black_box(5));
            black_box(result);
        });
    });

    group.bench_function("frobenius_iteration", |b| {
        b.iter(|| {
            let result = pc.frobenius(black_box(7));
            black_box(result);
        });
    });

    group.bench_function("frobenius_trace_exponents", |b| {
        b.iter(|| {
            let result = pc.frobenius_trace_exponents();
            black_box(result);
        });
    });

    group.bench_function("frobenius_norm_exponent_sum", |b| {
        b.iter(|| {
            let result = pc.frobenius_norm_exponent_sum();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark: Cache extension
fn bench_cache_extension(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_extension");

    let pc_base = PowComputer::new(Integer::from(5), 50);

    group.bench_function("extend_2x", |b| {
        b.iter(|| {
            let extended = pc_base.extend_cache(black_box(100));
            black_box(extended);
        });
    });

    group.bench_function("extend_10x", |b| {
        b.iter(|| {
            let extended = pc_base.extend_cache(black_box(500));
            black_box(extended);
        });
    });

    group.bench_function("extend_no_op", |b| {
        b.iter(|| {
            let extended = pc_base.extend_cache(black_box(25));
            black_box(extended);
        });
    });

    group.finish();
}

/// Benchmark: Multiple sequential operations (realistic workload)
fn bench_realistic_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_workload");

    group.bench_function("padic_arithmetic_sequence", |b| {
        let pc = PowComputer::new(Integer::from(5), 100);

        b.iter(|| {
            // Simulate a sequence of p-adic operations
            let mut values = vec![
                Integer::from(12345),
                Integer::from(67890),
                Integer::from(111213),
            ];

            // Reduce all values
            for v in values.iter_mut() {
                pc.reduce_in_place(v, 20);
            }

            // Compute some powers
            let p10 = pc.pow(10);
            let p25 = pc.pow(25);

            // More reductions
            let result1 = pc.reduce(values[0].clone() * values[1].clone(), 30);
            let result2 = pc.reduce(values[2].clone() + p10.clone(), 15);

            black_box((result1, result2, p25));
        });
    });

    group.finish();
}

/// Benchmark: Comparison with naive implementation
fn bench_comparison_with_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_vs_naive");

    let pc = PowComputer::new(Integer::from(5), 100);
    let prime = Integer::from(5);

    // Naive: compute p^n each time
    group.bench_function("naive_repeated_powers", |b| {
        b.iter(|| {
            let mut sum = Integer::zero();
            for i in 0..20 {
                sum = sum + prime.pow(black_box(i));
            }
            black_box(sum);
        });
    });

    // Cached: use PowComputer
    group.bench_function("cached_repeated_powers", |b| {
        b.iter(|| {
            let mut sum = Integer::zero();
            for i in 0..20 {
                sum = sum + pc.pow(black_box(i as usize));
            }
            black_box(sum);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_creation,
    bench_pow_access,
    bench_reduction,
    bench_reduction_inplace,
    bench_valuation,
    bench_extension_creation,
    bench_frobenius,
    bench_cache_extension,
    bench_realistic_workload,
    bench_comparison_with_naive,
);

criterion_main!(benches);
