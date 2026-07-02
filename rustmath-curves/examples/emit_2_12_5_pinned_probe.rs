//! Feasibility probe for a *direct* solve of the pinned `[2,12,5]` square system
//! (25 unknowns incl. `c`, 25 equations, no parameters). Renders a Julia script
//! that reports the per-equation degrees, the Bézout (total-degree) path count,
//! and attempts the polyhedral mixed-volume — WITHOUT tracking — so we learn
//! whether a global homotopy solve is even feasible.
//!
//! Usage: `cargo run -p rustmath-curves --example emit_2_12_5_pinned_probe -- [out.jl]`

use rustmath_curves::belyi::pinned::{pinned_system_2_12_5, unknown_names};
use rustmath_numerical::homotopy::render_system_julia;

fn main() {
    let sys = pinned_system_2_12_5();
    let names = unknown_names();
    let eqs = render_system_julia(&sys, &names);

    let vars = names.join(", ");
    let eqs_joined = eqs.join(",\n    ");

    let script = format!(
        "using HomotopyContinuation\n\n\
         @var {vars}\n\
         vars = [{vars}]\n\n\
         F = System([\n    {eqs_joined}\n]; variables=vars)\n\n\
         degs = degrees(F)\n\
         println(\"per-equation degrees: \", degs)\n\
         println(\"Bezout (total-degree paths): \", prod(big.(degs)))\n\n\
         println(\"computing mixed volume (polyhedral path count)...\")\n\
         try\n\
         \u{20}   mv = @time HomotopyContinuation.mixed_volume(F)\n\
         \u{20}   println(\"MIXED VOLUME (polyhedral paths): \", mv)\n\
         catch err\n\
         \u{20}   @warn \"mixed_volume failed / too large\" err\n\
         end\n",
        vars = vars,
        eqs_joined = eqs_joined,
    );

    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "belyi_2_12_5_pinned_probe.jl".to_string());
    std::fs::write(&out, &script).expect("write pinned probe");
    eprintln!("wrote {out}: {} vars, {} equations", names.len(), eqs.len());
}
