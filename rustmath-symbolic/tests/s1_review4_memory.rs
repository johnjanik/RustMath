//! S1 implementation, fix-up round 4: review 4's blocker B2 (resources lens,
//! `fu2/s1impl/S1.review4-resources.md`), as regression tests written before the fix.
//!
//! B2: round 3 gave `simplify` a memo of shared nodes. It stored the value of every node whose
//! `Arc` had a strong count above 1 and kept it until the walk ended. The count includes owners
//! outside the walked DAG. In the REPL, `a2 = a1*3` stores a shallow clone of a1's node under a
//! new `Arc`, so every child of a1 is held twice, once by a1 and once by the clone, although it
//! has one parent in the DAG of a2. A chain a_k = a_{k-1}*3 of 6 000 such lines therefore made
//! `simplify(a_6000)` keep the 6 000 intermediate literals of about 128 KB each: VmHWM 754 MB,
//! where HEAD's recursive `simplify`, which drops each intermediate value as it combines, used
//! 9 MB. The `+ 1` chain of 4 000 levels: HEAD 7 MB, round 3 501 MB. Eight factors of 2^20 bits
//! and 600 levels: HEAD 10 MB, round 3 604 MB; at 5 600 levels round 3 aborted on a failed
//! allocation (4 GB limit), HEAD needed 14 MB. Through the series engine (release):
//! `sin(a_6000)/(1-x)` at 0, order 1, budget 1e8 was `Ok` in 0.017 s, then `simplify` of the
//! result took 1.55 s and 755 MB (with `/ 3` in place of `* 3`: 11.7 s and 756 MB).
//!
//! S1_BAR (b), a regression against HEAD on an input HEAD handles, and (c) through the engine.
//! The required property: `simplify` keeps HEAD's memory on a DAG whose nodes are also held
//! elsewhere, and keeps round 3's sharing inside the walked DAG.
//!
//! VmHWM is the peak of the whole process, so these tests are a test binary of their own, they
//! run one at a time (a lock), and each resets the peak before it measures (writing 5 to
//! `/proc/self/clear_refs`, Linux 4.0 and later; without it the absolute peak is compared, which
//! is only stricter). Both profiles: `cargo test -p rustmath-symbolic --test s1_review4_memory`;
//! the `#[ignore]` test is the release measurement through the engine:
//! `cargo test --release -p rustmath-symbolic --test s1_review4_memory -- --ignored --test-threads=1`.
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`,
//! §49.1.3-49.1.4: 1/(1-x) has every coefficient 1, here times sin of a constant); S1 design §2.10
//! (`simplify`), S1_BAR (b) and (c).

mod s1_common;

use rustmath_integers::Integer;
use s1_common::api::*;
use s1_common::check::*;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// One measurement at a time: VmHWM is the peak of the process.
static SERIAL: Mutex<()> = Mutex::new(());

/// The growth of the peak that `simplify` of a chain may cause (HEAD: 9 MB in total).
const CHAIN_GROWTH_BYTES: u64 = 64 << 20;

/// Reset the peak resident set to the current one and return it (bytes).
fn reset_peak() -> u64 {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
    vm_hwm_bytes().unwrap_or(0)
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Add1,
    Mul3,
    Div3,
}

/// a_1 = `base` (parsed, not folded) and a_k = a_{k-1} op c for k = 2..=levels, as the REPL
/// builds them: each line stores a shallow clone of the previous value under a new `Arc` (the
/// children are shared), and every a_k stays alive (the REPL's variables).
fn chain(base: &str, op: Op, levels: usize) -> Vec<Expr> {
    let mut v: Vec<Expr> = vec![parse(base)];
    for k in 1..levels {
        let p = v[k - 1].clone();
        v.push(match op {
            Op::Add1 => p + Expr::from(1),
            Op::Mul3 => p * Expr::from(3),
            Op::Div3 => p / Expr::from(3),
        });
    }
    v
}

fn k_value() -> Integer {
    (Integer::from(2).pow(254) + Integer::from(1)).pow(4096)
}

/// B2 (both profiles, 64 MiB thread): `simplify` of the last link of a chain whose links are all
/// held (the REPL's variables) raises the peak by less than 64 MB and gives the folded value.
/// gp: `(2^254+1)^4096*3^5999`, `(2^254+1)^4096 + 3999`, `prod(k=1,8,(2^254+k)^4096) + 599`.
/// Before the fix: 754 MB, 501 MB and 604 MB (review 4, debug and release alike).
#[test]
fn b2r4_simplify_of_a_chain_held_by_variables() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let k = k_value();
    let eight: String = (1..=8)
        .map(|j| format!("(2^254+{j})^4096"))
        .collect::<Vec<_>>()
        .join("*");
    let eight_value = (1..=8).fold(Integer::from(1), |acc, j| {
        acc * (Integer::from(2).pow(254) + Integer::from(j)).pow(4096)
    });
    let cases: Vec<(String, String, Op, usize, Integer)> = vec![
        (
            "K * 3, 6000 levels".into(),
            K_TEXT.into(),
            Op::Mul3,
            6000,
            k.clone() * Integer::from(3).pow(5999),
        ),
        (
            "K + 1, 4000 levels".into(),
            K_TEXT.into(),
            Op::Add1,
            4000,
            k.clone() + Integer::from(3999),
        ),
        (
            "eight factors + 1, 600 levels".into(),
            eight,
            Op::Add1,
            600,
            eight_value + Integer::from(599),
        ),
    ];
    let mut failures = Vec::new();
    for (what, base, op, levels, want) in cases {
        let r = std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(move || {
                let v = chain(&base, op, levels);
                let before = reset_peak();
                let t = Instant::now();
                let s = v.last().expect("a link").simplify();
                let secs = t.elapsed().as_secs_f64();
                let peak = vm_hwm_bytes().unwrap_or(0);
                let ok = s == Expr::Integer(want);
                drop(v);
                (peak.saturating_sub(before), secs, ok)
            })
            .expect("spawn")
            .join()
            .expect("the thread panicked");
        let (growth, secs, ok) = r;
        eprintln!(
            "{what}: simplify {secs:.3} s, the peak grew by {} MB",
            growth >> 20
        );
        if growth >= CHAIN_GROWTH_BYTES {
            failures.push(format!(
                "{what}: simplify raised VmHWM by {} MB",
                growth >> 20
            ));
        }
        if !ok {
            failures.push(format!("{what}: simplify did not give the folded value"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The text of K = (2^254+1)^4096.
const K_TEXT: &str = "(2^254+1)^4096";

/// B2 through the engine (release, 8 MB thread): `sin(a_6000)/(1-x)`, a_k the `* 3` and the
/// `/ 3` chain held by variables, through `taylor_with_budget` at 0, order 1, with 1e8 (within
/// 20 s), then refused, or `simplify` of the result within 20 s with the peak under 512 MB.
/// Before the fix: `* 3` 1.55 s and 755 MB; `/ 3` 11.7 s and 756 MB.
#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b2r4_taylor_then_simplify_of_a_held_chain_release() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for op in [Op::Mul3, Op::Div3] {
        let v = Arc::new(chain(K_TEXT, op, 6000));
        reset_peak();
        let vv = v.clone();
        let r = std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(move || {
                let a = vv.last().expect("a link").clone();
                let z = Expr::Unary(rustmath_symbolic::UnaryOp::Sin, Arc::new(a))
                    / (Expr::from(1) - Expr::symbol("x"));
                let t = Instant::now();
                let r = tw(&z, &x(), &Expr::from(0), 1, REPL_TAYLOR_WORK);
                (r, t.elapsed().as_secs_f64())
            })
            .expect("spawn")
            .join()
            .expect("the thread panicked");
        let (r, secs) = r;
        assert!(secs < BAR_SECONDS, "{op:?}: taylor {secs:.1} s");
        let r = match r {
            Err(e) => {
                eprintln!("{op:?}: refused ({e:?})");
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
                let s = rr.simplify();
                let _ = tx.send((s == *rr, t.elapsed().as_secs_f64()));
            })
            .expect("spawn");
        match rx.recv_timeout(Duration::from_secs_f64(BAR_SECONDS)) {
            Err(_) => {
                panic!("{op:?}: simplify of the result did not finish within {BAR_SECONDS} s")
            }
            Ok((_, t)) => eprintln!("{op:?}: simplify {t:.3} s"),
        }
        let peak = vm_hwm_bytes().unwrap_or(0);
        assert!(peak < BAR_BYTES, "{op:?}: VmHWM {} MB", peak >> 20);
        drop(v);
    }
}
