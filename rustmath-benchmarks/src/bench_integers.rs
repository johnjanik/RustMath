use clap::Parser;
use rustmath_integers::{Integer, prime};
use serde_json::json;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "bench_integers")]
#[command(about = "RustMath integer computation benchmarks")]
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

/// Benchmark: GCD of two large integers
fn bench_gcd_large(iterations: usize) -> f64 {
    let a = Integer::from(2).pow(100) + Integer::from(123456789);
    let b = Integer::from(2).pow(100) + Integer::from(987654321);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = a.gcd(&b);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: GCD of coprime integers
fn bench_gcd_coprime(iterations: usize) -> f64 {
    let a = Integer::from(2).pow(64) - Integer::from(59);
    let b = Integer::from(2).pow(64) - Integer::from(17);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = a.gcd(&b);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Extended GCD
fn bench_extended_gcd(iterations: usize) -> f64 {
    let a = Integer::from(12345678);
    let b = Integer::from(87654321);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = a.extended_gcd(&b);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Integer power (small base, large exponent)
fn bench_power_large_exp(iterations: usize) -> f64 {
    let base = Integer::from(3);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = base.pow(1000);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Modular exponentiation (cryptographic size)
fn bench_modular_exp(iterations: usize) -> f64 {
    let base = Integer::from(2).pow(100) + Integer::from(7);
    let exp = Integer::from(2).pow(50);
    let modulus = Integer::from(2).pow(128) - Integer::from(159);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = base.mod_pow(&exp, &modulus).unwrap();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Primality test (Miller-Rabin) on large prime
fn bench_primality_large_prime(iterations: usize) -> f64 {
    // 2^127 - 1 (Mersenne prime)
    let n = Integer::from(2).pow(127) - Integer::from(1);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = prime::is_prime(&n);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Primality test on composite number
fn bench_primality_composite(iterations: usize) -> f64 {
    let n = Integer::from(2).pow(100) + Integer::from(1);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = prime::is_prime(&n);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Find next prime after large number
fn bench_next_prime(iterations: usize) -> f64 {
    let n = Integer::from(2).pow(50);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = prime::next_prime(&n);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Integer square root
fn bench_sqrt(iterations: usize) -> f64 {
    let n = Integer::from(2).pow(256);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = n.sqrt().unwrap();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Integer nth root
fn bench_nth_root(iterations: usize) -> f64 {
    let n = Integer::from(2).pow(100);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = n.nth_root(5).unwrap();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Euler's totient function
fn bench_euler_phi(iterations: usize) -> f64 {
    let n = Integer::from(123456789);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = n.euler_phi().unwrap();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Multiply large integers
fn bench_multiply_large(iterations: usize) -> f64 {
    let a = Integer::from(2).pow(128) + Integer::from(12345);
    let b = Integer::from(2).pow(128) + Integer::from(67890);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = a.clone() * b.clone();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn main() {
    let args = Args::parse();

    let avg_time_ms = match args.test.as_str() {
        "gcd_large" => bench_gcd_large(args.iterations),
        "gcd_coprime" => bench_gcd_coprime(args.iterations),
        "extended_gcd" => bench_extended_gcd(args.iterations),
        "power_large_exp" => bench_power_large_exp(args.iterations),
        "modular_exp" => bench_modular_exp(args.iterations),
        "primality_large_prime" => bench_primality_large_prime(args.iterations),
        "primality_composite" => bench_primality_composite(args.iterations),
        "next_prime" => bench_next_prime(args.iterations),
        "sqrt" => bench_sqrt(args.iterations),
        "nth_root" => bench_nth_root(args.iterations),
        "euler_phi" => bench_euler_phi(args.iterations),
        "multiply_large" => bench_multiply_large(args.iterations),
        _ => {
            eprintln!("Unknown test: {}", args.test);
            eprintln!("Available tests:");
            eprintln!("  - gcd_large");
            eprintln!("  - gcd_coprime");
            eprintln!("  - extended_gcd");
            eprintln!("  - power_large_exp");
            eprintln!("  - modular_exp");
            eprintln!("  - primality_large_prime");
            eprintln!("  - primality_composite");
            eprintln!("  - next_prime");
            eprintln!("  - sqrt");
            eprintln!("  - nth_root");
            eprintln!("  - euler_phi");
            eprintln!("  - multiply_large");
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
