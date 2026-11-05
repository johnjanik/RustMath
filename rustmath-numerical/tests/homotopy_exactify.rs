//! Integration tests for the Wave-2 parameter-homotopy solve interface
//! (`homotopy` + `exactify`).
//!
//! These live in `tests/` (compiled as a separate crate against the built
//! library) rather than as `#[cfg(test)] mod tests` inside the modules, because
//! the crate's pre-existing `src/lib.rs` test module has an unrelated
//! `Integrable`/`Optimizable` `eval` ambiguity that fails to compile the `--lib`
//! test target. Confirmed pre-existing (clean git before this work); see the
//! agent report. Ported/adapted from `dessin_engine` tests in
//! `src/homotopy_adapter.rs` and `src/exactification.rs`.

use rustmath_numerical::exactify::{exactify, ExactifyOutcome};
use rustmath_numerical::homotopy::{
    parse_result_json, render_parameter_homotopy_script, ComplexDecimal, CoordinateReIm,
    NumericalSolution, ParameterHomotopyJob,
};
use rustmath_polynomials::poly_system::PolySystem;
use rustmath_rationals::Rational;

// --- (a) parameter-homotopy emitter + candidate importer round-trip ---------

#[test]
fn emits_parameter_homotopy_script_and_round_trips_candidate() {
    // System F(z; p): x^2 - a = 0, x - y + b = 0, with variables (x, y) and
    // parameters (a, b). Encoded as a PolySystem in 4 slots [x, y, a, b].
    let sys = PolySystem::from_terms(
        4,
        &[
            // x^2 - a
            vec![(vec![2, 0, 0, 0], 1), (vec![0, 0, 1, 0], -1)],
            // x - y + b
            vec![(vec![1, 0, 0, 0], 1), (vec![0, 1, 0, 0], -1), (vec![0, 0, 0, 1], 1)],
        ],
    );

    let variables = vec!["x".to_string(), "y".to_string()];
    let parameters = vec!["a".to_string(), "b".to_string()];
    // Free start solution z0 and start/target parameter points.
    let z0 = vec![ComplexDecimal::real("1.0"), ComplexDecimal::new("2.5", "0.5")];
    let p0 = vec![ComplexDecimal::real("1.0"), ComplexDecimal::real("0.0")];
    let pstar = vec![ComplexDecimal::real("4.0"), ComplexDecimal::real("1.0")];

    let job = ParameterHomotopyJob::from_system(
        "demo_ph",
        &sys,
        &variables,
        &parameters,
        z0,
        p0,
        pstar,
        "/tmp/ph_out.json",
    )
    .expect("job assembles");

    assert_eq!(job.variables, vec!["x", "y"]);
    assert_eq!(job.parameters, vec!["a", "b"]);
    // Equations rendered symbolically over vars + params.
    assert!(job.equations_julia[0].contains("x^2"));
    assert!(job.equations_julia[0].contains('a'));
    assert!(job.equations_julia[1].contains('b'));

    let script = render_parameter_homotopy_script(&job);

    // Separate @var blocks: variables and parameters.
    assert!(script.contains("@var x, y"), "variables block");
    assert!(script.contains("@var a, b"), "parameters block");
    // System with an explicit parameters kwarg.
    assert!(script.contains("System("));
    assert!(script.contains("parameters=pars"));
    assert!(script.contains("variables=vars"));
    // Start/target parameters and a free start solution.
    assert!(script.contains("start_parameters=p0"), "start parameters");
    assert!(script.contains("target_parameters=pstar"), "target parameters");
    assert!(script.contains("z0 = ["), "free start solution");
    assert!(script.contains("p0 = ["), "start parameter point");
    assert!(script.contains("pstar = ["), "target parameter point");
    // The monodromy-enlarged fibre variant.
    assert!(script.contains("monodromy_solve(F, [z0], p0"), "monodromy variant");
    // The free start solution's complex literal is present.
    assert!(script.contains("ComplexF64(1.0, 0.0)"));

    // Importer round-trip: a solver result document parses back into candidates.
    let json = r#"{"solutions":[
        {"coordinates_re_im_decimal":[{"re":"2.0","im":"0.0"},{"re":"3.0","im":"0.0"}],
         "residual_norm_decimal":"1e-30","path_status":"candidate"}
    ]}"#;
    let res = parse_result_json(json).expect("candidate JSON parses");
    assert_eq!(res.solutions.len(), 1);
    assert_eq!(res.solutions[0].coordinates_re_im_decimal[0].re, "2.0");
    assert_eq!(res.solutions[0].coordinates_re_im_decimal[1].re, "3.0");
    assert_eq!(res.solutions[0].path_status, "candidate");
}

// --- (b) exactify: recognize a rational solution and certify it --------------

fn coord(re: &str, im: &str) -> CoordinateReIm {
    CoordinateReIm {
        re: re.to_string(),
        im: im.to_string(),
    }
}

#[test]
fn exactify_rational_candidate_certifies() {
    // System x^2 - 4 = 0, x - y + 1 = 0; candidate (2, 3) is an exact zero.
    let sys = PolySystem::from_terms(
        2,
        &[
            vec![(vec![2, 0], 1), (vec![0, 0], -4)],
            vec![(vec![1, 0], 1), (vec![0, 1], -1), (vec![0, 0], 1)],
        ],
    );
    let sol = NumericalSolution {
        coordinates_re_im_decimal: vec![coord("2.0", "0.0"), coord("3.0", "0.0")],
        residual_norm_decimal: "1e-30".into(),
        path_status: "candidate".into(),
    };
    match exactify(&sol, &sys, 4) {
        ExactifyOutcome::CertifiedRational(pt) => {
            assert_eq!(pt, vec![Rational::from_i64(2), Rational::from_i64(3)]);
        }
        other => panic!("expected CertifiedRational, got {other:?}"),
    }
}

#[test]
fn exactify_rejects_spurious_rational() {
    // Same system, candidate (2, 2) is NOT a zero (y should be 3).
    let sys = PolySystem::from_terms(
        2,
        &[
            vec![(vec![2, 0], 1), (vec![0, 0], -4)],
            vec![(vec![1, 0], 1), (vec![0, 1], -1), (vec![0, 0], 1)],
        ],
    );
    let sol = NumericalSolution {
        coordinates_re_im_decimal: vec![coord("2.0", "0.0"), coord("2.0", "0.0")],
        residual_norm_decimal: "1e-30".into(),
        path_status: "candidate".into(),
    };
    assert_eq!(exactify(&sol, &sys, 4), ExactifyOutcome::SubstitutionFailed);
}
