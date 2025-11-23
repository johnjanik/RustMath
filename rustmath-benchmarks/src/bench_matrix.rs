use clap::Parser;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use serde_json::json;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "bench_matrix")]
#[command(about = "RustMath matrix computation benchmarks")]
struct Args {
    /// Test name to run
    #[arg(long)]
    test: String,

    /// Number of iterations
    #[arg(long, default_value = "100")]
    iterations: usize,

    /// Output results as JSON
    #[arg(long)]
    json: bool,
}

/// Benchmark: Multiply two 10x10 matrices
fn bench_matrix_multiply_10x10(iterations: usize) -> f64 {
    let data1: Vec<Integer> = (1..=100).map(|i| Integer::from(i)).collect();
    let data2: Vec<Integer> = (1..=100).map(|i| Integer::from(101 - i)).collect();

    let m1 = Matrix::from_vec(10, 10, data1).unwrap();
    let m2 = Matrix::from_vec(10, 10, data2).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = m1.mul(&m2).unwrap();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Multiply two 50x50 matrices
fn bench_matrix_multiply_50x50(iterations: usize) -> f64 {
    let data1: Vec<Integer> = (1..=2500).map(|i| Integer::from(i % 100)).collect();
    let data2: Vec<Integer> = (1..=2500).map(|i| Integer::from((i * 7) % 100)).collect();

    let m1 = Matrix::from_vec(50, 50, data1).unwrap();
    let m2 = Matrix::from_vec(50, 50, data2).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = m1.mul(&m2).unwrap();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Compute determinant of 10x10 matrix
fn bench_matrix_determinant_10x10(iterations: usize) -> f64 {
    let data: Vec<Integer> = (1..=100).map(|i| Integer::from(i)).collect();
    let m = Matrix::from_vec(10, 10, data).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = m.determinant().unwrap();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Compute determinant of 20x20 matrix
fn bench_matrix_determinant_20x20(iterations: usize) -> f64 {
    let data: Vec<Integer> = (1..=400).map(|i| Integer::from(i % 50)).collect();
    let m = Matrix::from_vec(20, 20, data).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = m.determinant().unwrap();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Transpose a 100x100 matrix
fn bench_matrix_transpose_100x100(iterations: usize) -> f64 {
    let data: Vec<Integer> = (1..=10000).map(|i| Integer::from(i % 100)).collect();
    let m = Matrix::from_vec(100, 100, data).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = m.transpose();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Matrix power (10x10)^5
fn bench_matrix_power(iterations: usize) -> f64 {
    let data: Vec<Integer> = (1..=100).map(|i| Integer::from(i % 10)).collect();
    let m = Matrix::from_vec(10, 10, data).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = m.pow(5).unwrap();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Matrix addition 100x100
fn bench_matrix_add_100x100(iterations: usize) -> f64 {
    let data1: Vec<Integer> = (1..=10000).map(|i| Integer::from(i % 100)).collect();
    let data2: Vec<Integer> = (1..=10000).map(|i| Integer::from((i * 3) % 100)).collect();

    let m1 = Matrix::from_vec(100, 100, data1).unwrap();
    let m2 = Matrix::from_vec(100, 100, data2).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = m1.clone() + m2.clone();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Matrix scalar multiplication 100x100
fn bench_matrix_scalar_mul_100x100(iterations: usize) -> f64 {
    let data: Vec<Integer> = (1..=10000).map(|i| Integer::from(i % 100)).collect();
    let m = Matrix::from_vec(100, 100, data).unwrap();
    let scalar = Integer::from(7);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = m.scalar_mul(&scalar);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Check matrix symmetry 100x100
fn bench_matrix_is_symmetric(iterations: usize) -> f64 {
    let mut data: Vec<Integer> = vec![Integer::from(0); 10000];
    for i in 0..100 {
        for j in 0..100 {
            let val = Integer::from(((i + j) % 50) as i64);
            data[i * 100 + j] = val.clone();
            data[j * 100 + i] = val;
        }
    }
    let m = Matrix::from_vec(100, 100, data).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = m.is_symmetric();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Identity matrix construction 1000x1000
fn bench_matrix_identity_1000x1000(iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = Matrix::<Integer>::identity(1000);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn main() {
    let args = Args::parse();

    let avg_time_ms = match args.test.as_str() {
        "multiply_10x10" => bench_matrix_multiply_10x10(args.iterations),
        "multiply_50x50" => bench_matrix_multiply_50x50(args.iterations),
        "determinant_10x10" => bench_matrix_determinant_10x10(args.iterations),
        "determinant_20x20" => bench_matrix_determinant_20x20(args.iterations),
        "transpose_100x100" => bench_matrix_transpose_100x100(args.iterations),
        "power" => bench_matrix_power(args.iterations),
        "add_100x100" => bench_matrix_add_100x100(args.iterations),
        "scalar_mul_100x100" => bench_matrix_scalar_mul_100x100(args.iterations),
        "is_symmetric" => bench_matrix_is_symmetric(args.iterations),
        "identity_1000x1000" => bench_matrix_identity_1000x1000(args.iterations),
        _ => {
            eprintln!("Unknown test: {}", args.test);
            eprintln!("Available tests:");
            eprintln!("  - multiply_10x10");
            eprintln!("  - multiply_50x50");
            eprintln!("  - determinant_10x10");
            eprintln!("  - determinant_20x20");
            eprintln!("  - transpose_100x100");
            eprintln!("  - power");
            eprintln!("  - add_100x100");
            eprintln!("  - scalar_mul_100x100");
            eprintln!("  - is_symmetric");
            eprintln!("  - identity_1000x1000");
            std::process::exit(1);
        }
    };

    if args.json {
        println!("{}", json!({
            "test": args.test,
            "iterations": args.iterations,
            "avg_time_ms": avg_time_ms,
            "total_time_s": avg_time_ms * args.iterations as f64 / 1000.0,
        }));
    } else {
        println!("Test: {}", args.test);
        println!("Iterations: {}", args.iterations);
        println!("Average time: {:.6} ms", avg_time_ms);
        println!("Total time: {:.3} s", avg_time_ms * args.iterations as f64 / 1000.0);
    }
}
