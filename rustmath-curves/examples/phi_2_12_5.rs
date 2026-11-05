use rustmath_curves::belyi::hypergeometric::{abc_params, phi_in_u};
fn main() {
    let (a, b, c) = abc_params(2, 12, 5);
    println!("(2,12,5) hypergeometric params: A={a}, B={b}, C={c}");
    let phi = phi_in_u(2, 12, 5, 24);
    println!("phi(w) as series in u = w/kappa  (phi = u^2 + c4 u^4 + ...):");
    for k in (2..24).step_by(2) {
        println!("  [u^{k:>2}]  {}", phi.coeff(k));
    }
}
