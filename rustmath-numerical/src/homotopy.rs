//! Parameter-homotopy interface to an external numerical polynomial-system
//! solver (HomotopyContinuation.jl by default).
//!
//! Ported and **generalized** from `dessin_engine/src/homotopy_adapter.rs`
//! (`/home/john/inverse_galois/M23/dessin_engine`). The reference implementation
//! emitted a *generic total-degree* `solve(F)` call; this module emits a
//! **parameter homotopy** for a system `F(z; p) = 0` with a distinguished block
//! of parameters `p`, a caller-supplied *free start pair* `(z0, p0)`, and a
//! *target* parameter point `p*`. The solver tracks `p0 → p*`, and a
//! `monodromy_solve(F, [z0], p0)` variant enlarges the generic fibre before
//! tracking. This is the committed solve route (generic total-degree homotopy
//! and global Gröbner are retired — see `DESSIN_REFACTOR_PLAN.md`).
//!
//! The design keeps the reference module's discipline: numerical output is an
//! **untrusted suggestion**. This module only *emits* a job, *runs* the solver,
//! and *parses* candidates. Certifying a candidate is `crate::exactify`'s job
//! (recognize + exact back-substitution); nothing here is a theorem.
//!
//! Nothing Belyi-specific lives here — the module is general. The assembly of a
//! start pair from an affine-in-parameters system (`p0 = Ψ(z0)`) is a Belyi-layer
//! concern; here `z0`, `p0`, and `p*` are inputs.

use num_bigint::BigInt;
use num_traits::Signed;
use rustmath_integers::Integer;
use rustmath_polynomials::poly_system::PolySystem;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HomotopyError {
    #[error("external solver failed: {0}")]
    External(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("job assembly error: {0}")]
    Assembly(String),
}

/// A complex number carried as a `(re, im)` pair of decimal strings, so job
/// specifications round-trip losslessly through JSON and into Julia without an
/// intermediate binary float representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComplexDecimal {
    pub re: String,
    pub im: String,
}

impl ComplexDecimal {
    pub fn new(re: impl Into<String>, im: impl Into<String>) -> Self {
        Self {
            re: re.into(),
            im: im.into(),
        }
    }

    /// A real value with zero imaginary part.
    pub fn real(re: impl Into<String>) -> Self {
        Self {
            re: re.into(),
            im: "0.0".to_string(),
        }
    }

    /// `ComplexF64(re, im)` — Julia literal.
    fn to_julia(&self) -> String {
        format!("ComplexF64({}, {})", self.re, self.im)
    }
}

/// A parameter-homotopy job for `F(z; p) = 0`.
///
/// * `variables` — the names of the solve variables `z` (length `n`).
/// * `parameters` — the names of the parameters `p` (length `m`); a separate
///   `@var` block from the variables.
/// * `equations_julia` — the `n`-... i.e. one Julia expression per equation,
///   written over both the variable and parameter symbols.
/// * `start_solution` (`z0`) — a free start solution, one entry per variable.
/// * `start_parameters` (`p0`) — the start parameter point, one entry per
///   parameter; for a system affine in `p`, a Belyi-layer caller sets
///   `p0 = Ψ(z0)`, but that assembly is out of scope here.
/// * `target_parameters` (`p*`) — the parameter point the homotopy tracks to.
/// * `use_monodromy` — also emit a `monodromy_solve(F, [z0], p0)` pass and track
///   its discovered fibre to `p*`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterHomotopyJob {
    pub name: String,
    pub variables: Vec<String>,
    pub parameters: Vec<String>,
    pub equations_julia: Vec<String>,
    pub start_solution: Vec<ComplexDecimal>,
    pub start_parameters: Vec<ComplexDecimal>,
    pub target_parameters: Vec<ComplexDecimal>,
    pub precision_digits: usize,
    pub use_monodromy: bool,
    /// Where the Julia script writes its result JSON (a path string, embedded
    /// raw in the script).
    pub output_json: String,
}

impl ParameterHomotopyJob {
    /// Assemble a job directly from a [`PolySystem`] that encodes `F(z; p)` as a
    /// single polynomial system whose first `variables.len()` slots are the
    /// solve variables and whose remaining slots are the parameters.
    ///
    /// The system's variable count must equal `variables.len() +
    /// parameters.len()`; the parameter symbols are rendered as Julia `@var`
    /// parameters (not substituted), so an affine-in-`p` encoder gets a true
    /// parameter homotopy for free.
    #[allow(clippy::too_many_arguments)]
    pub fn from_system(
        name: &str,
        sys: &PolySystem,
        variables: &[String],
        parameters: &[String],
        start_solution: Vec<ComplexDecimal>,
        start_parameters: Vec<ComplexDecimal>,
        target_parameters: Vec<ComplexDecimal>,
        output_json: impl Into<String>,
    ) -> Result<Self, HomotopyError> {
        let n = variables.len();
        let m = parameters.len();
        if sys.num_variables() != n + m {
            return Err(HomotopyError::Assembly(format!(
                "system has {} variables but variables({}) + parameters({}) = {}",
                sys.num_variables(),
                n,
                m,
                n + m
            )));
        }
        if start_solution.len() != n {
            return Err(HomotopyError::Assembly(format!(
                "start_solution has {} entries, expected {} (one per variable)",
                start_solution.len(),
                n
            )));
        }
        if start_parameters.len() != m || target_parameters.len() != m {
            return Err(HomotopyError::Assembly(format!(
                "parameter points must have {m} entries (start={}, target={})",
                start_parameters.len(),
                target_parameters.len()
            )));
        }
        let names: Vec<String> = variables.iter().chain(parameters).cloned().collect();
        let equations_julia = render_system_julia(sys, &names);
        Ok(Self {
            name: name.to_string(),
            variables: variables.to_vec(),
            parameters: parameters.to_vec(),
            equations_julia,
            start_solution,
            start_parameters,
            target_parameters,
            precision_digits: 50,
            use_monodromy: true,
            output_json: output_json.into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Rendering exact integer-coefficient equations to Julia strings
// ---------------------------------------------------------------------------

/// Render one integer-coefficient term list as a Julia polynomial string over
/// the given symbol names (variables followed by parameters). Ported from
/// `dessin_engine/src/homotopy_adapter.rs::poly_to_julia`, adapted to
/// `rustmath_integers::Integer` coefficients.
fn poly_to_julia(terms: &[(Vec<u32>, BigInt)], names: &[String]) -> String {
    if terms.is_empty() {
        return "0".to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    for (exps, coeff) in terms {
        let mut factors: Vec<String> = Vec::new();
        let mag = coeff.abs();
        let all_zero = exps.iter().all(|&e| e == 0);
        if mag != BigInt::from(1) || all_zero {
            factors.push(mag.to_string());
        }
        for (j, &e) in exps.iter().enumerate() {
            match e {
                0 => {}
                1 => factors.push(names[j].clone()),
                _ => factors.push(format!("{}^{}", names[j], e)),
            }
        }
        let body = if factors.is_empty() {
            "1".to_string()
        } else {
            factors.join("*")
        };
        let sign = if coeff.is_negative() { "-" } else { "+" };
        parts.push(format!("{sign} {body}"));
    }
    let joined = parts.join(" ");
    joined.strip_prefix("+ ").unwrap_or(&joined).to_string()
}

/// Render every equation of a [`PolySystem`] to a Julia string over `names`
/// (dense exponent vectors of length `names.len()`).
pub fn render_system_julia(sys: &PolySystem, names: &[String]) -> Vec<String> {
    let nvars = names.len();
    sys.polynomials()
        .iter()
        .map(|poly| {
            let terms: Vec<(Vec<u32>, BigInt)> = poly
                .terms()
                .map(|(mono, coeff)| {
                    let mut exps = vec![0u32; nvars];
                    for (&var, &e) in mono.iter_exponents() {
                        if var < nvars {
                            exps[var] = e;
                        }
                    }
                    (exps, integer_to_bigint(coeff))
                })
                .collect();
            poly_to_julia(&terms, names)
        })
        .collect()
}

fn integer_to_bigint(n: &Integer) -> BigInt {
    n.as_bigint().clone()
}

// ---------------------------------------------------------------------------
// Script emission
// ---------------------------------------------------------------------------

fn vec_complex_julia(v: &[ComplexDecimal]) -> String {
    let entries: Vec<String> = v.iter().map(ComplexDecimal::to_julia).collect();
    format!("[{}]", entries.join(", "))
}

/// Render a HomotopyContinuation.jl parameter-homotopy driver script as a string
/// (separated from I/O so it is testable without Julia).
///
/// The emitted script:
/// * declares `variables` and `parameters` in **separate** `@var` blocks,
/// * builds `System(eqs; variables=vars, parameters=pars)`,
/// * tracks the free start solution `z0` from `start_parameters=p0` to
///   `target_parameters=p*`,
/// * (if `use_monodromy`) also runs `monodromy_solve(F, [z0], p0)` and tracks the
///   discovered fibre to `p*`, unioning the two solution sets,
/// * writes `{ "solutions": [...] }` as JSON, one `(re, im)` decimal pair per
///   coordinate — the shape [`parse_result_json`] imports.
pub fn render_parameter_homotopy_script(job: &ParameterHomotopyJob) -> String {
    let vars = job.variables.join(", ");
    let pars = job.parameters.join(", ");
    let var_list = job.variables.join(", ");
    let par_list = job.parameters.join(", ");
    let eqs = job.equations_julia.join(",\n    ");
    let z0 = vec_complex_julia(&job.start_solution);
    let p0 = vec_complex_julia(&job.start_parameters);
    let pstar = vec_complex_julia(&job.target_parameters);

    let monodromy_block = if job.use_monodromy {
        "\
# Enlarge the generic fibre by monodromy about p0, then track each to pstar.
try
    mon = monodromy_solve(F, [z0], p0; show_progress=false)
    monres = solve(F, solutions(mon); start_parameters=p0, target_parameters=pstar,
                   show_progress=false)
    for s in solutions(monres)
        push!(allsols, s)
    end
catch err
    @warn \"monodromy pass failed\" err
end
"
    } else {
        ""
    };

    format!(
        "using HomotopyContinuation\n\
         using JSON3\n\n\
         # Parameter homotopy for F(z; p) = 0 : track p0 -> pstar from a free start (z0, p0).\n\
         @var {vars}\n\
         @var {pars}\n\n\
         vars = [{var_list}]\n\
         pars = [{par_list}]\n\n\
         F = System([\n    {eqs}\n]; variables=vars, parameters=pars)\n\n\
         z0 = {z0}\n\
         p0 = {p0}\n\
         pstar = {pstar}\n\n\
         allsols = Vector{{Vector{{ComplexF64}}}}()\n\n\
         # Direct track of the free start solution.\n\
         res = solve(F, [z0]; start_parameters=p0, target_parameters=pstar, show_progress=false)\n\
         for s in solutions(res)\n\
         \u{20}   push!(allsols, s)\n\
         end\n\n\
         {monodromy_block}\n\
         sols = []\n\
         for sol in allsols\n\
         \u{20}   coords = [Dict(\"re\" => string(real(z)), \"im\" => string(imag(z))) for z in sol]\n\
         \u{20}   push!(sols, Dict(\n\
         \u{20}       \"coordinates_re_im_decimal\" => coords,\n\
         \u{20}       \"residual_norm_decimal\" => string(maximum(abs.(F(sol, pstar)))),\n\
         \u{20}       \"path_status\" => \"candidate\"))\n\
         end\n\n\
         open(raw\"{out}\", \"w\") do io\n\
         \u{20}   JSON3.write(io, Dict(\"solutions\" => sols))\n\
         end\n",
        vars = vars,
        pars = pars,
        var_list = var_list,
        par_list = par_list,
        eqs = eqs,
        z0 = z0,
        p0 = p0,
        pstar = pstar,
        monodromy_block = monodromy_block,
        out = job.output_json,
    )
}

/// Emit the driver script to `script_path`.
pub fn write_parameter_homotopy_script(
    job: &ParameterHomotopyJob,
    script_path: &Path,
) -> Result<(), HomotopyError> {
    std::fs::write(script_path, render_parameter_homotopy_script(job))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Candidate importer (kept from the reference implementation)
// ---------------------------------------------------------------------------

/// One numerical candidate: a `(re, im)` decimal pair per coordinate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalSolution {
    pub coordinates_re_im_decimal: Vec<CoordinateReIm>,
    pub residual_norm_decimal: String,
    pub path_status: String,
}

/// A single coordinate as re/im decimal strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinateReIm {
    pub re: String,
    pub im: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomotopyResult {
    pub solutions: Vec<NumericalSolution>,
}

/// Parse a solver result document (separated out so it is testable without
/// Julia). Ported from `dessin_engine/src/homotopy_adapter.rs::parse_result_json`.
pub fn parse_result_json(json: &str) -> Result<HomotopyResult, HomotopyError> {
    serde_json::from_str(json).map_err(|e| HomotopyError::Parse(e.to_string()))
}

/// Run the solver and parse candidates. Requires a working `julia` with
/// HomotopyContinuation.jl + JSON3; untestable without that toolchain.
pub fn run_homotopycontinuation(
    job: &ParameterHomotopyJob,
    julia_bin: &str,
    workdir: &Path,
) -> Result<HomotopyResult, HomotopyError> {
    let script_path = workdir.join(format!("{}.jl", job.name));
    write_parameter_homotopy_script(job, &script_path)?;
    let status = Command::new(julia_bin).arg(&script_path).status()?;
    if !status.success() {
        return Err(HomotopyError::External(format!("julia exited with {status}")));
    }
    parse_result_json(&std::fs::read_to_string(&job.output_json)?)
}
