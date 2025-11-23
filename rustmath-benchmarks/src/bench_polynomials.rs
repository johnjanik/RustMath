use clap::Parser;
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use serde_json::json;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "bench_polynomials")]
#[command(about = "RustMath polynomial computation benchmarks")]
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

/// Benchmark: Multiply two dense polynomials of degree 50
fn bench_poly_multiply_dense(iterations: usize) -> f64 {
    let coeffs1: Vec<Integer> = (0..51).map(|i| Integer::from(i)).collect();
    let coeffs2: Vec<Integer> = (0..51).map(|i| Integer::from(51 - i)).collect();
    let p1 = UnivariatePolynomial::new(coeffs1);
    let p2 = UnivariatePolynomial::new(coeffs2);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = p1.clone() * p2.clone();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Multiply two sparse polynomials
fn bench_poly_multiply_sparse(iterations: usize) -> f64 {
    let mut coeffs1 = vec![Integer::from(0); 101];
    coeffs1[0] = Integer::from(1);
    coeffs1[50] = Integer::from(5);
    coeffs1[100] = Integer::from(3);

    let mut coeffs2 = vec![Integer::from(0); 101];
    coeffs2[0] = Integer::from(2);
    coeffs2[25] = Integer::from(4);
    coeffs2[100] = Integer::from(7);

    let p1 = UnivariatePolynomial::new(coeffs1);
    let p2 = UnivariatePolynomial::new(coeffs2);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = p1.clone() * p2.clone();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Evaluate polynomial at multiple points
fn bench_poly_evaluate(iterations: usize) -> f64 {
    let coeffs: Vec<Integer> = (0..51).map(|i| Integer::from(i)).collect();
    let p = UnivariatePolynomial::new(coeffs);
    let point = Integer::from(5);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = p.evaluate(&point);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Compute GCD of two polynomials
fn bench_poly_gcd(iterations: usize) -> f64 {
    // Create p1 = (x - 1)(x - 2)(x - 3)
    let x_minus_1 = UnivariatePolynomial::new(vec![Integer::from(-1), Integer::from(1)]);
    let x_minus_2 = UnivariatePolynomial::new(vec![Integer::from(-2), Integer::from(1)]);
    let x_minus_3 = UnivariatePolynomial::new(vec![Integer::from(-3), Integer::from(1)]);
    let p1 = x_minus_1.clone() * x_minus_2.clone() * x_minus_3.clone();

    // Create p2 = (x - 2)(x - 3)(x - 4)
    let x_minus_4 = UnivariatePolynomial::new(vec![Integer::from(-4), Integer::from(1)]);
    let p2 = x_minus_2.clone() * x_minus_3.clone() * x_minus_4;

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = p1.gcd(&p2);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Compute derivative of polynomial
fn bench_poly_derivative(iterations: usize) -> f64 {
    let coeffs: Vec<Integer> = (0..51).map(|i| Integer::from(i * i)).collect();
    let p = UnivariatePolynomial::new(coeffs);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = p.derivative();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Compose two polynomials p(q(x))
fn bench_poly_compose(iterations: usize) -> f64 {
    let p = UnivariatePolynomial::new(vec![
        Integer::from(1),
        Integer::from(2),
        Integer::from(3),
    ]);
    let q = UnivariatePolynomial::new(vec![
        Integer::from(5),
        Integer::from(-2),
        Integer::from(1),
    ]);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = p.compose(&q);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Compute LCM of two polynomials
fn bench_poly_lcm(iterations: usize) -> f64 {
    let p1 = UnivariatePolynomial::new(vec![
        Integer::from(6),
        Integer::from(11),
        Integer::from(6),
        Integer::from(1),
    ]);
    let p2 = UnivariatePolynomial::new(vec![
        Integer::from(2),
        Integer::from(5),
        Integer::from(3),
    ]);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = p1.lcm(&p2);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Compute polynomial discriminant
fn bench_poly_discriminant(iterations: usize) -> f64 {
    let p = UnivariatePolynomial::new(vec![
        Integer::from(1),
        Integer::from(-6),
        Integer::from(11),
        Integer::from(-6),
    ]);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = p.discriminant();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Power of polynomial p^10
fn bench_poly_power(iterations: usize) -> f64 {
    let p = UnivariatePolynomial::new(vec![
        Integer::from(1),
        Integer::from(1),
    ]);

    let start = Instant::now();
    for _ in 0..iterations {
        let mut result = p.clone();
        for _ in 1..10 {
            result = result.clone() * p.clone();
        }
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn main() {
    let args = Args::parse();

    let avg_time_ms = match args.test.as_str() {
        "multiply_dense" => bench_poly_multiply_dense(args.iterations),
        "multiply_sparse" => bench_poly_multiply_sparse(args.iterations),
        "evaluate" => bench_poly_evaluate(args.iterations),
        "gcd" => bench_poly_gcd(args.iterations),
        "derivative" => bench_poly_derivative(args.iterations),
        "compose" => bench_poly_compose(args.iterations),
        "lcm" => bench_poly_lcm(args.iterations),
        "discriminant" => bench_poly_discriminant(args.iterations),
        "power" => bench_poly_power(args.iterations),
        _ => {
            eprintln!("Unknown test: {}", args.test);
            eprintln!("Available tests:");
            eprintln!("  - multiply_dense");
            eprintln!("  - multiply_sparse");
            eprintln!("  - evaluate");
            eprintln!("  - gcd");
            eprintln!("  - derivative");
            eprintln!("  - compose");
            eprintln!("  - lcm");
            eprintln!("  - discriminant");
            eprintln!("  - power");
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
