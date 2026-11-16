//! S1 implementation, fix-up round 5: review 5's blocker B2 (resources lens,
//! `fu2/s1impl/S1.review5-resources.md`), as regression tests written before the fix.
//!
//! B2: round 4's memo of `simplify` counts the parents of a node inside the walked DAG and keeps
//! the value of a node with two or more parents until its last use. It keeps it also when the
//! first use throws the value away (`0 * q`), and whatever its size. HEAD's recursive `simplify`
//! recomputes a shared node at each occurrence and holds one value at a time. Library DAGs (P one
//! integer leaf of 2^20 bits, q_i = Neg(P), the expression names each q_i twice, as `0 * q_i` in
//! two sums): m = 4 000, HEAD +3 MB in 0.046 s, round 4 +517 MB; m = 40 000 under a 4 GB address
//! space limit, HEAD 46 MB in 0.45 s, round 4 aborted (`memory allocation of 131080 bytes
//! failed`) in debug and release; P of 2^23 bits and m = 2 000, round 4 +2 008 MB. With
//! q_i = P*(i+2) and S1 - S2: HEAD +3 MB, round 4 +502 MB. Through the engine, the derivative of an
//! accepted result names each link of a kernel's source chain twice:
//! `sin((2^254+1)^4096*3*...*3)/(1-x)` with 6 000 factors (12 KB), `taylor` `Ok` in 0.20 s, then
//! the REPL's `diff(t, x)` peaked at 839 MB (debug); with 11 000 factors (22 KB),
//! `simplify(R.differentiate(x))` peaked at 1 422 MB (release).
//!
//! S1_BAR (b), a regression against HEAD on an input HEAD handles (an abort included), and (c)
//! through the engine. The required property: `simplify` keeps HEAD's memory up to a bounded
//! payload on every input, and keeps round 3's sharing for results.
//!
//! VmHWM is the peak of the whole process, so these tests are a test binary of their own, they
//! run one at a time (a lock), and each resets the peak before it measures (writing 5 to
//! `/proc/self/clear_refs`, Linux 4.0 and later; without it the absolute peak is compared, which
//! is only stricter). The abort cases run in a child process of this test binary under
//! `ulimit -v` (the shell's limit, no unsafe code). Both profiles:
//! `cargo test -p rustmath-symbolic --test s1_review5_memory`; the `#[ignore]` test is the release
//! measurement through the engine:
//! `cargo test --release -p rustmath-symbolic --test s1_review5_memory -- --ignored --test-threads=1`.
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`,
//! §49.1.3-49.1.4: 1/(1-x) has every coefficient 1, here times sin of a constant); S1 design §2.10
//! (`simplify`), S1_BAR (b) and (c).

mod s1_common;

use rustmath_integers::Integer;
use rustmath_symbolic::{BinaryOp, UnaryOp};
use s1_common::api::*;
use s1_common::check::*;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// One measurement at a time: VmHWM is the peak of the process.
static SERIAL: Mutex<()> = Mutex::new(());

/// The growth of the peak that `simplify` of the DAGs below may cause (HEAD: 3 MB).
const MEMO_GROWTH_BYTES: u64 = 64 << 20;

/// S1_BAR (c): 20 s and 512 MB.
const BAR_SECONDS: f64 = 20.0;
const BAR_BYTES: u64 = 512 << 20;

/// Reset the peak resident set to the current one and return it (bytes).
fn reset_peak() -> u64 {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
    vm_hwm_bytes().unwrap_or(0)
}

/// The shapes of review 5's `memo/memo_zero.rs` and `memo/memo_dag.rs`.
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// q_i = Neg(P); E = Z1 + Z2, Z = sum of 0 * q_i, both sums over the same q_i `Arc`s
    ZeroTimes,
    /// q_i = P * (i + 2); E = S1 - S2, S = sum of q_i, both over the same q_i `Arc`s
    SumMinusSum,
}

/// P = 2^bits + 12345 (a leaf of bits + 1 bits, held by one `Arc`), and the DAG of `shape` with
/// m shared q_i.
fn memo_dag(shape: Shape, m: usize, bits: u32) -> Expr {
    let p = Arc::new(Expr::Integer(
        Integer::from(2).pow(bits) + Integer::from(12345),
    ));
    let qs: Vec<Arc<Expr>> = (0..m)
        .map(|i| match shape {
            Shape::ZeroTimes => Arc::new(Expr::Unary(UnaryOp::Neg, p.clone())),
            Shape::SumMinusSum => Arc::new(Expr::Binary(
                BinaryOp::Mul,
                p.clone(),
                Arc::new(Expr::from((i + 2) as i64)),
            )),
        })
        .collect();
    let zero = Arc::new(Expr::from(0));
    let sum = || -> Arc<Expr> {
        let term = |i: usize| match shape {
            Shape::ZeroTimes => Arc::new(Expr::Binary(BinaryOp::Mul, zero.clone(), qs[i].clone())),
            Shape::SumMinusSum => qs[i].clone(),
        };
        let mut s = term(0);
        for i in 1..m {
            s = Arc::new(Expr::Binary(BinaryOp::Add, s, term(i)));
        }
        s
    };
    match shape {
        Shape::ZeroTimes => Expr::Binary(BinaryOp::Add, sum(), sum()),
        Shape::SumMinusSum => Expr::Binary(BinaryOp::Sub, sum(), sum()),
    }
}

/// B2 (both profiles, 64 MiB thread): `simplify` of review 5's library DAGs with m = 4 000 and P of
/// 2^20 bits raises the peak by less than 64 MB and gives 0. HEAD: +3 MB (0.046 s and 6.9 s in
/// debug). Before the fix: +517 MB and +502 MB (debug and release alike).
#[test]
fn b2r5_simplify_does_not_keep_the_values_of_a_dag() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut failures = Vec::new();
    for shape in [Shape::ZeroTimes, Shape::SumMinusSum] {
        let r = std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(move || {
                let e = memo_dag(shape, 4000, 1 << 20);
                let before = reset_peak();
                let t = Instant::now();
                let s = e.simplify();
                let secs = t.elapsed().as_secs_f64();
                let peak = vm_hwm_bytes().unwrap_or(0);
                (peak.saturating_sub(before), secs, s == Expr::from(0))
            })
            .expect("spawn")
            .join()
            .expect("the thread panicked");
        let (growth, secs, zero) = r;
        eprintln!(
            "{shape:?}, m = 4000: simplify {secs:.3} s, the peak grew by {} MB",
            growth >> 20
        );
        if growth >= MEMO_GROWTH_BYTES {
            failures.push(format!(
                "{shape:?}: simplify raised VmHWM by {} MB",
                growth >> 20
            ));
        }
        if !zero {
            failures.push(format!("{shape:?}: simplify did not give 0"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The case a child process runs, from `S1R5_MEMO_CHILD` = "m:bits" (P = 2^bits + 12345).
#[test]
#[ignore = "run in a child process by b2r5_simplify_of_a_large_dag_does_not_abort"]
fn b2r5_memo_child() {
    let Some(spec) = std::env::var_os("S1R5_MEMO_CHILD") else {
        return;
    };
    let spec = spec.to_string_lossy().to_string();
    let (m, bits) = spec.split_once(':').expect("m:bits");
    let (m, bits): (usize, u32) = (m.parse().expect("m"), bits.parse().expect("bits"));
    let (zero, growth) = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            let e = memo_dag(Shape::ZeroTimes, m, bits);
            let before = reset_peak();
            let zero = e.simplify() == Expr::from(0);
            (zero, vm_hwm_bytes().unwrap_or(0).saturating_sub(before))
        })
        .expect("spawn")
        .join()
        .expect("the thread panicked");
    println!("S1R5 MEMO zero={zero} growth={}", growth >> 20);
}

/// B2 (both profiles): the `0 * q_i` DAG with m = 40 000 (P of 2^20 bits) and with m = 2 000 (P of
/// 2^23 bits) under a 4 GB address-space limit (`ulimit -v 4000000`, review 5's limit) gives 0,
/// does not abort, and raises the peak by less than 64 MB. HEAD: 0.45 s and 46 MB in total; 0.18 s and +3 MB. Before the fix: an abort on
/// a failed allocation (rc 134) in both profiles; +2 008 MB.
#[test]
fn b2r5_simplify_of_a_large_dag_does_not_abort() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let exe = std::env::current_exe().expect("the test binary");
    let mut failures = Vec::new();
    for spec in ["40000:1048576", "2000:8388608"] {
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg("ulimit -v 4000000 && exec \"$0\" \"$@\"")
            .arg(&exe)
            .args([
                "--exact",
                "b2r5_memo_child",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("S1R5_MEMO_CHILD", spec)
            .output()
            .expect("run the child");
        let stdout = String::from_utf8_lossy(&out.stdout).to_string();
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        let growth_mb: Option<u64> = stdout.find("S1R5 MEMO zero=true growth=").and_then(|at| {
            stdout[at + "S1R5 MEMO zero=true growth=".len()..]
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
        });
        match growth_mb {
            Some(mb) if out.status.success() && mb < MEMO_GROWTH_BYTES >> 20 => {
                eprintln!("m:bits = {spec}: 0, the peak grew by {mb} MB")
            }
            _ => failures.push(format!(
                "m:bits = {spec}: the child failed or grew too much ({:?}): {}\n{}",
                out.status.code(),
                stdout.trim(),
                stderr.lines().last().unwrap_or("")
            )),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// K = (2^254+1)^4096 followed by `n` factors 3 (review 5's `cons/sinmul3_6000.txt` and
/// `cons/sinmul3_11000.txt`).
fn sin_mul3(n: usize) -> String {
    format!("sin((2^254+1)^4096{})/(1-x)", "*3".repeat(n))
}

/// B2 through the engine (release, 8 MB thread): `sin(K*3 x n)/(1-x)` with n = 6 000 (12 025 bytes)
/// and 11 000 (22 025 bytes) at 0, order 1, through `taylor_with_budget` with 1e8 (within 20 s):
/// refused, or `simplify(R.differentiate(x))` within 20 s with the peak under 512 MB. Before the
/// fix: 3.8 s and 774 MB; 7.7 s and 1 422 MB.
#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b2r5_differentiate_then_simplify_of_a_chain_result_release() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for n in [6000usize, 11000] {
        let f = sin_mul3(n);
        assert!(f.len() <= 100_000);
        reset_peak();
        let r = std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(move || {
                let t = Instant::now();
                let r = tw(&parse(&f), &x(), &Expr::from(0), 1, REPL_TAYLOR_WORK);
                (r, t.elapsed().as_secs_f64())
            })
            .expect("spawn")
            .join()
            .expect("the thread panicked");
        let (r, secs) = r;
        assert!(secs < BAR_SECONDS, "n = {n}: taylor {secs:.1} s");
        let r = match r {
            Err(e) => {
                eprintln!("n = {n}: refused ({e:?})");
                continue;
            }
            Ok(r) => Arc::new(r),
        };
        let (tx, rx) = mpsc::channel();
        let rr = r.clone();
        std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(move || {
                let t = Instant::now();
                let d = rr.differentiate(&Symbol::new("x")).simplify();
                let _ = tx.send((format!("{d}").len(), t.elapsed().as_secs_f64()));
            })
            .expect("spawn");
        match rx.recv_timeout(Duration::from_secs_f64(BAR_SECONDS)) {
            Ok((len, t)) => eprintln!("n = {n}: simplify of the derivative {t:.3} s ({len})"),
            Err(_) => panic!("n = {n}: simplify of the derivative not done within {BAR_SECONDS} s"),
        }
        let peak = vm_hwm_bytes().unwrap_or(0);
        assert!(peak < BAR_BYTES, "n = {n}: VmHWM {} MB", peak >> 20);
    }
}
