//! Emit the HomotopyContinuation.jl driver for the `[2,12,5]` M23 portal solve.
//!
//! Builds the parameter-homotopy job from a rational seed and renders the Julia
//! monodromy driver (`monodromy_solve(F,[z0],p0)` → track the fibre to `p*` →
//! JSON), the N-probe / candidate harvester for the pinned system.
//!
//! Usage: `cargo run -p rustmath-curves --example emit_2_12_5_homotopy -- [out.jl]`
//! (default output path: `belyi_2_12_5.jl` in the current directory).

use rustmath_curves::belyi::pipeline::{assemble_2_12_5_homotopy, HomotopySeed};
use rustmath_numerical::homotopy::render_parameter_homotopy_script;
use rustmath_rationals::Rational;

fn ri(n: i64) -> Rational {
    Rational::from_i64(n)
}

fn main() {
    // A generic rational seed z0: psi(z0)=p0 makes it an exact start solution of
    // F(z; p0)=0, so it seeds the monodromy loop about the generic fibre.
    let seed = HomotopySeed {
        a: vec![ri(1), ri(-2), ri(3), ri(0), ri(-1), ri(2), ri(1), ri(-3)],
        b: vec![ri(2), ri(1), ri(-1), ri(3), ri(0), ri(-2), ri(1), ri(1)],
        r: vec![ri(-1), ri(2), ri(1)],
        s: vec![ri(3), ri(-2), ri(1), ri(2)],
        lambda: Rational::new(3, 2).unwrap(),
    };

    let job = assemble_2_12_5_homotopy(&seed);
    let script = render_parameter_homotopy_script(&job);

    let out = std::env::args().nth(1).unwrap_or_else(|| "belyi_2_12_5.jl".to_string());
    std::fs::write(&out, &script).expect("write Julia driver");

    eprintln!(
        "wrote {out}\n  variables : {}\n  parameters: {}\n  equations : {}\n  result    : {}",
        job.variables.len(),
        job.parameters.len(),
        job.equations_julia.len(),
        job.output_json,
    );
}
