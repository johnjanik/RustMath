use clap::Parser;
use rustmath_symbolic::{Expr, Symbol, operators::derivative, simplify::simplify, walker::substitute};
use serde_json::json;
use std::time::Instant;
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "bench_symbolic")]
#[command(about = "RustMath symbolic computation benchmarks")]
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

/// Benchmark: Differentiate a simple polynomial
fn bench_diff_polynomial(iterations: usize) -> f64 {
    let x = Expr::symbol("x");
    let x_sym = Symbol::new("x");
    let expr = Expr::from(5) * x.clone().pow(Expr::from(5))
             + Expr::from(3) * x.clone().pow(Expr::from(2))
             - Expr::from(7) * x.clone()
             + Expr::from(2);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = derivative(expr.clone(), &x_sym, 1);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Differentiate trigonometric expression
fn bench_diff_trig(iterations: usize) -> f64 {
    let x = Expr::symbol("x");
    let x_sym = Symbol::new("x");
    let sin_x = Expr::sin(x.clone());
    let cos_x = Expr::cos(x.clone());
    let expr = sin_x * cos_x;

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = derivative(expr.clone(), &x_sym, 1);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Differentiate nested exponential/sin
fn bench_diff_nested(iterations: usize) -> f64 {
    let x = Expr::symbol("x");
    let x_sym = Symbol::new("x");
    let x_squared = x.clone().pow(Expr::from(2));
    let sin_x2 = Expr::sin(x_squared);
    let expr = Expr::exp(sin_x2);

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = derivative(expr.clone(), &x_sym, 1);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Product rule chain (x³ * sin(x) * exp(x))
fn bench_diff_product_chain(iterations: usize) -> f64 {
    let x = Expr::symbol("x");
    let x_sym = Symbol::new("x");
    let x_cubed = x.clone().pow(Expr::from(3));
    let sin_x = Expr::sin(x.clone());
    let exp_x = Expr::exp(x.clone());
    let expr = x_cubed * sin_x * exp_x;

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = derivative(expr.clone(), &x_sym, 1);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: High-order derivative d^10/dx^10 (x^20)
fn bench_diff_high_order(iterations: usize) -> f64 {
    let x = Expr::symbol("x");
    let x_sym = Symbol::new("x");
    let expr = x.clone().pow(Expr::from(20));

    let start = Instant::now();
    for _ in 0..iterations {
        let mut result = expr.clone();
        for _ in 0..10 {
            result = derivative(result, &x_sym, 1);
        }
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Simplify trigonometric identity sin²(x) + cos²(x)
fn bench_simplify_trig(iterations: usize) -> f64 {
    let x = Expr::symbol("x");
    let sin_x = Expr::sin(x.clone());
    let cos_x = Expr::cos(x.clone());
    let expr = sin_x.pow(Expr::from(2)) + cos_x.pow(Expr::from(2));

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = simplify(&expr);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Expand (x + y)^10
fn bench_expand_binomial(iterations: usize) -> f64 {
    let x = Expr::symbol("x");
    let y = Expr::symbol("y");
    let expr = (x + y).pow(Expr::from(10));

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = expr.expand();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Simplify rational expression (x² - 1)/(x - 1)
fn bench_simplify_rational(iterations: usize) -> f64 {
    let x = Expr::symbol("x");
    let numerator = x.clone().pow(Expr::from(2)) - Expr::from(1);
    let denominator = x.clone() - Expr::from(1);
    let expr = numerator / denominator;

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = simplify(&expr);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark: Substitute x = 2 into large expression
fn bench_substitution(iterations: usize) -> f64 {
    let x = Expr::symbol("x");
    let x_sym = Symbol::new("x");
    let expr = x.clone().pow(Expr::from(5))
             + Expr::from(3) * x.clone().pow(Expr::from(2))
             - Expr::from(7) * x.clone()
             + Expr::from(2);

    let mut replacements = HashMap::new();
    replacements.insert(x_sym, Expr::from(2));

    let start = Instant::now();
    for _ in 0..iterations {
        let _result = substitute(&expr, &replacements);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn main() {
    let args = Args::parse();

    let avg_time_ms = match args.test.as_str() {
        "diff_polynomial" => bench_diff_polynomial(args.iterations),
        "diff_trig" => bench_diff_trig(args.iterations),
        "diff_nested" => bench_diff_nested(args.iterations),
        "diff_product_chain" => bench_diff_product_chain(args.iterations),
        "diff_high_order" => bench_diff_high_order(args.iterations),
        "simplify_trig" => bench_simplify_trig(args.iterations),
        "expand_binomial" => bench_expand_binomial(args.iterations),
        "simplify_rational" => bench_simplify_rational(args.iterations),
        "substitution" => bench_substitution(args.iterations),
        _ => {
            eprintln!("Unknown test: {}", args.test);
            eprintln!("Available tests:");
            eprintln!("  - diff_polynomial");
            eprintln!("  - diff_trig");
            eprintln!("  - diff_nested");
            eprintln!("  - diff_product_chain");
            eprintln!("  - diff_high_order");
            eprintln!("  - simplify_trig");
            eprintln!("  - expand_binomial");
            eprintln!("  - simplify_rational");
            eprintln!("  - substitution");
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
