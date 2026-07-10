//! LMFDB (L-functions and Modular Forms Database) interface
//!
//! Provides access to a small **built-in, local** table of mathematical
//! objects (elliptic curves, modular forms, number fields, Dirichlet
//! characters) that mirrors a handful of well-known LMFDB entries.
//!
//! # Important: this is not a live LMFDB client
//!
//! Despite the name, [`LMFDBClient`] does **not** perform any network
//! requests by default (the crate's `online` feature does not wire up real
//! HTTP calls for this module at all — see the `client` field). Every
//! `search_*`/`lookup_*` method here is served from a tiny hard-coded table
//! (`builtin_elliptic_curves`, `builtin_modular_forms`,
//! `builtin_number_fields`, `builtin_dirichlet_character`) containing only a
//! few illustrative examples (conductors 11/37/389, levels 11/37, degrees
//! 2/3, etc.). Looking up anything not in that short list simply returns an
//! empty result (or, for [`LMFDBClient::dirichlet_character`], an error if
//! the input is mathematically invalid) — it is never a live query against
//! the real LMFDB database at <https://www.lmfdb.org>. Treat every value
//! returned from this module as demonstration/example data, not as a
//! verified database lookup.
//!
//! # Example
//!
//! ```no_run
//! use rustmath_databases::lmfdb::LMFDBClient;
//!
//! let client = LMFDBClient::new();
//!
//! // Search the small built-in table for elliptic curves
//! if let Ok(results) = client.search_elliptic_curves("11.a1") {
//!     println!("Found {} results", results.len());
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Error type for LMFDB operations
#[derive(Debug)]
pub enum LMFDBError {
    /// Network error during API request
    NetworkError(String),
    /// JSON parsing error
    ParseError(String),
    /// Object not found
    NotFound(String),
    /// Invalid query
    InvalidQuery(String),
}

impl fmt::Display for LMFDBError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LMFDBError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            LMFDBError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            LMFDBError::NotFound(msg) => write!(f, "Not found: {}", msg),
            LMFDBError::InvalidQuery(msg) => write!(f, "Invalid query: {}", msg),
        }
    }
}

impl Error for LMFDBError {}

/// Result type for LMFDB operations
pub type Result<T> = std::result::Result<T, LMFDBError>;

/// Represents an elliptic curve from LMFDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMFDBEllipticCurve {
    /// LMFDB label
    pub label: String,
    /// Conductor
    pub conductor: u64,
    /// Weierstrass coefficients [a1, a2, a3, a4, a6]
    pub ainvs: Vec<i64>,
    /// Rank
    pub rank: u32,
    /// Torsion structure
    pub torsion_structure: Vec<u32>,
    /// j-invariant (as string for exact representation)
    pub jinv: String,
}

/// Represents a modular form from LMFDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModularForm {
    /// Label
    pub label: String,
    /// Weight
    pub weight: u32,
    /// Level
    pub level: u64,
    /// Dimension
    pub dim: u32,
}

/// Represents a number field from LMFDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberField {
    /// Label
    pub label: String,
    /// Degree
    pub degree: u32,
    /// Discriminant
    pub disc: i64,
    /// Class number
    pub class_number: u64,
}

/// Represents a Dirichlet character
///
/// Produced by [`LMFDBClient::dirichlet_character`], which computes
/// `order`/`conductor`/`is_primitive` exactly via the standard Conrey
/// labeling of `(Z/modulus Z)^*` (see the `dirichlet` module for the
/// algorithm) rather than serving them from a lookup table.
#[derive(Debug, Clone)]
pub struct DirichletCharacter {
    /// Modulus `q`
    pub modulus: u64,
    /// Conrey number `n`: labels the character `chi_q(n, ·)`, `n` coprime to `q`
    pub number: u32,
    /// Order of the character, i.e. the multiplicative order of `number` in `(Z/modulus Z)^*`
    pub order: u32,
    /// Conductor: the smallest `f | modulus` such that this character is induced
    /// from a character mod `f`
    pub conductor: u64,
    /// Whether the character is primitive, i.e. `conductor == modulus`
    pub is_primitive: bool,
}

/// Client for querying a small **built-in, local** table of LMFDB-style data
///
/// All lookup methods on this type are served from a hard-coded table of a
/// few illustrative examples baked into this crate (see the module-level
/// docs); none of them contact the real LMFDB service, even when the
/// `online` feature is enabled. This type exists for demonstration and
/// testing purposes, not as a general-purpose LMFDB client.
pub struct LMFDBClient {
    #[allow(dead_code)]
    base_url: String,
    /// Real HTTP client, only present (and only usable) when the `online`
    /// feature is enabled. Currently unused since queries below are served
    /// from built-in data, but kept available for future LMFDB HTTP API use.
    #[cfg(feature = "online")]
    #[allow(dead_code)]
    client: reqwest::blocking::Client,
    /// In-memory cache of queries
    #[allow(dead_code)]
    cache: std::sync::Mutex<HashMap<String, String>>,
}

impl LMFDBClient {
    /// Create a new LMFDB client
    pub fn new() -> Self {
        LMFDBClient {
            base_url: "https://www.lmfdb.org/api".to_string(),
            #[cfg(feature = "online")]
            client: reqwest::blocking::Client::new(),
            cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Create a new LMFDB client with custom base URL
    #[allow(dead_code)]
    pub fn with_base_url(base_url: String) -> Self {
        LMFDBClient {
            base_url,
            #[cfg(feature = "online")]
            client: reqwest::blocking::Client::new(),
            cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Search the **built-in local table** for elliptic curves
    ///
    /// This does not query the live LMFDB; it matches `query` as a substring
    /// against the handful of curves baked into this crate (currently
    /// conductors 11, 37, and 389). Anything else returns an empty vector.
    ///
    /// # Arguments
    ///
    /// * `query` - Search query (e.g., conductor, label)
    ///
    /// # Returns
    ///
    /// Vector of matching elliptic curves from the built-in table
    pub fn search_elliptic_curves(&self, query: &str) -> Result<Vec<LMFDBEllipticCurve>> {
        // Served entirely from the small built-in table below; this is not a
        // real LMFDB API query.
        Ok(self.builtin_elliptic_curves(query))
    }

    /// Lookup elliptic curve by label in the **built-in local table**
    ///
    /// See [`LMFDBClient::search_elliptic_curves`]; this is not a live LMFDB
    /// lookup.
    pub fn lookup_elliptic_curve(&self, label: &str) -> Result<Option<LMFDBEllipticCurve>> {
        let curves = self.builtin_elliptic_curves(label);
        Ok(curves.into_iter().next())
    }

    /// Search the **built-in local table** for modular forms
    ///
    /// Only a couple of (weight, level) pairs are present (currently
    /// (2, 11) and (2, 37)); this is not a live LMFDB query.
    pub fn search_modular_forms(&self, weight: u32, level: u64) -> Result<Vec<ModularForm>> {
        Ok(self.builtin_modular_forms(weight, level))
    }

    /// Search the **built-in local table** for number fields
    ///
    /// Only degrees 2 and 3 have any entries; this is not a live LMFDB
    /// query.
    pub fn search_number_fields(&self, degree: u32) -> Result<Vec<NumberField>> {
        Ok(self.builtin_number_fields(degree))
    }

    /// Compute the Conrey-labeled Dirichlet character `chi_modulus(number, ·)`
    ///
    /// Unlike the other methods on this client, this is **not** served from
    /// a hard-coded table: `order`, `conductor`, and `is_primitive` are
    /// computed exactly via the standard Conrey construction (CRT
    /// decomposition of `(Z/modulus Z)^*` into prime-power components, with
    /// the `C2 x C_{2^(e-2)}` splitting for the `2^e` component when
    /// `e >= 3`, discrete logs w.r.t. a primitive root of each odd
    /// prime-power component, and the standard `-1`/`5` generators for the
    /// `2^e` component).
    ///
    /// # Errors
    ///
    /// Returns [`LMFDBError::InvalidQuery`] if `number` is not coprime to
    /// `modulus` (i.e. does not label a valid Dirichlet character mod
    /// `modulus`), or if `modulus` is `0`.
    pub fn dirichlet_character(&self, modulus: u64, number: u32) -> Result<DirichletCharacter> {
        self.builtin_dirichlet_character(modulus, number)
    }

    // Built-in data for demonstration

    fn builtin_elliptic_curves(&self, query: &str) -> Vec<LMFDBEllipticCurve> {
        let mut curves = Vec::new();

        // Add some well-known curves
        if query.contains("11") || query.contains("11.a") {
            curves.push(LMFDBEllipticCurve {
                label: "11.a1".to_string(),
                conductor: 11,
                ainvs: vec![0, -1, 1, -10, -20],
                rank: 0,
                torsion_structure: vec![5],
                jinv: "-122023936/161051".to_string(),
            });
        }

        if query.contains("37") || query.contains("37.a") {
            curves.push(LMFDBEllipticCurve {
                label: "37.a1".to_string(),
                conductor: 37,
                ainvs: vec![0, 0, 1, -1, 0],
                rank: 1,
                torsion_structure: vec![],
                jinv: "-7*13^3/37".to_string(),
            });
        }

        if query.contains("389") || query.contains("389.a") {
            curves.push(LMFDBEllipticCurve {
                label: "389.a1".to_string(),
                conductor: 389,
                ainvs: vec![0, 1, 1, -2, 0],
                rank: 2,
                torsion_structure: vec![],
                jinv: "-1159088625/9834496".to_string(),
            });
        }

        curves
    }

    fn builtin_modular_forms(&self, weight: u32, level: u64) -> Vec<ModularForm> {
        let mut forms = Vec::new();

        // Example modular forms
        if weight == 2 && level == 11 {
            forms.push(ModularForm {
                label: "11.2.a.a".to_string(),
                weight: 2,
                level: 11,
                dim: 1,
            });
        }

        if weight == 2 && level == 37 {
            forms.push(ModularForm {
                label: "37.2.a.a".to_string(),
                weight: 2,
                level: 37,
                dim: 1,
            });
        }

        forms
    }

    fn builtin_number_fields(&self, degree: u32) -> Vec<NumberField> {
        let mut fields = Vec::new();

        if degree == 2 {
            fields.push(NumberField {
                label: "2.0.5.1".to_string(),
                degree: 2,
                disc: 5,
                class_number: 1,
            });

            fields.push(NumberField {
                label: "2.0.8.1".to_string(),
                degree: 2,
                disc: 8,
                class_number: 1,
            });
        }

        if degree == 3 {
            fields.push(NumberField {
                label: "3.1.23.1".to_string(),
                degree: 3,
                disc: 23,
                class_number: 1,
            });
        }

        fields
    }

    /// Compute the exact order and conductor of the Conrey character
    /// `chi_modulus(number, ·)`.
    ///
    /// This is genuine number theory (not built-in table data): it uses the
    /// standard Conrey pairing construction. See
    /// [`dirichlet::order_and_conductor`] for the algorithm and
    /// [`dirichlet`]'s tests for independent verification against a
    /// brute-force reference character construction.
    fn builtin_dirichlet_character(&self, modulus: u64, number: u32) -> Result<DirichletCharacter> {
        if modulus == 0 {
            return Err(LMFDBError::InvalidQuery(
                "modulus must be a positive integer".to_string(),
            ));
        }
        let n = (number as u64) % modulus.max(1);
        if dirichlet::gcd(n, modulus) != 1 {
            return Err(LMFDBError::InvalidQuery(format!(
                "number={number} is not coprime to modulus={modulus}; \
                 it does not label a Dirichlet character mod {modulus}"
            )));
        }

        let (order, conductor) = dirichlet::order_and_conductor(n, modulus);
        let is_primitive = conductor == modulus;

        Ok(DirichletCharacter {
            modulus,
            number,
            order: order as u32,
            conductor,
            is_primitive,
        })
    }
}

impl Default for LMFDBClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Exact (non-database) computation of Conrey Dirichlet character invariants.
///
/// Implements the standard "Conrey labeling" of Dirichlet characters used by
/// LMFDB: `(Z/qZ)^*` is decomposed via CRT into its prime-power components;
/// each odd-prime-power component `(Z/p^e Z)^*` is cyclic and is identified
/// with `Z/dZ` (`d = phi(p^e)`) via a primitive root; the `2^e` component for
/// `e >= 3` is identified with `C2 x C_{2^(e-2)}` via the standard generators
/// `-1` and `5`. The character `chi_q(n, ·)` pairs the discrete-log
/// coordinates of `n` with those of its argument component-wise. Under this
/// pairing, the order of `chi_q(n, ·)` equals the multiplicative order of
/// `n` in `(Z/qZ)^*`, and the conductor is the product, over prime-power
/// components, of the smallest `p^j` through which that component's local
/// character factors.
///
/// This module is independently verified (see the `tests` submodule) against
/// a brute-force reference that builds the actual character function
/// `chi_q(n, a)` from the same CRT/generator data and directly (a) finds the
/// LCM of the denominators of its values to get the order, and (b) scans all
/// divisors of `q` to find the smallest one through which `chi_q(n, ·)`
/// factors, to get the conductor — with no shortcut formula assumed.
pub(crate) mod dirichlet {
    /// `gcd(a, b)`, with `gcd(0, b) = b` and `gcd(a, 0) = a`.
    pub(crate) fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }

    fn lcm(a: u64, b: u64) -> u64 {
        if a == 0 || b == 0 {
            0
        } else {
            a / gcd(a, b) * b
        }
    }

    /// `p`-adic valuation of `m` (largest `k` with `p^k | m`); `m` must be `> 0`.
    fn v_p(mut m: u64, p: u64) -> u32 {
        let mut k = 0;
        while m % p == 0 {
            m /= p;
            k += 1;
        }
        k
    }

    /// Factor `n` into `(prime, exponent)` pairs via trial division.
    ///
    /// This is only ever used on the (small, user-supplied) `modulus` of a
    /// built-in demonstration Dirichlet character, so trial division is
    /// sufficient; it is not meant for cryptographic-scale integers.
    fn factorize(mut n: u64) -> Vec<(u64, u32)> {
        let mut factors = Vec::new();
        let mut d = 2u64;
        while d.saturating_mul(d) <= n {
            if n % d == 0 {
                let mut e = 0u32;
                while n % d == 0 {
                    n /= d;
                    e += 1;
                }
                factors.push((d, e));
            }
            d += 1;
        }
        if n > 1 {
            factors.push((n, 1));
        }
        factors
    }

    /// `base^exp mod modu`, computed with `u128` intermediates to avoid overflow.
    fn mod_pow(base: u64, exp: u64, modu: u64) -> u64 {
        if modu == 1 {
            return 0;
        }
        let mut result: u128 = 1;
        let mut b: u128 = (base % modu) as u128;
        let modu = modu as u128;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = (result * b) % modu;
            }
            b = (b * b) % modu;
            e >>= 1;
        }
        result as u64
    }

    /// Divisors of `n` in increasing order, from its factorization.
    fn divisors_sorted(n: u64) -> Vec<u64> {
        if n == 0 {
            return vec![];
        }
        let mut divs = vec![1u64];
        for (p, e) in factorize(n) {
            let mut new_divs = Vec::with_capacity(divs.len() * (e as usize + 1));
            let mut pk = 1u64;
            for _ in 0..=e {
                for &d in &divs {
                    new_divs.push(d * pk);
                }
                pk *= p;
            }
            divs = new_divs;
        }
        divs.sort_unstable();
        divs
    }

    /// Multiplicative order of `n` modulo `modu` (`n` must be coprime to `modu`).
    fn multiplicative_order(n: u64, modu: u64) -> u64 {
        if modu == 1 {
            return 1;
        }
        let phi = euler_phi(modu);
        let one = 1u64 % modu;
        for d in divisors_sorted(phi) {
            if mod_pow(n, d, modu) == one {
                return d;
            }
        }
        unreachable!("order of a unit must divide phi(modu)")
    }

    /// Euler's totient of `n`, computed from its factorization.
    fn euler_phi(n: u64) -> u64 {
        let mut result = n;
        for (p, _) in factorize(n) {
            result = result / p * (p - 1);
        }
        result
    }

    /// A primitive root of the cyclic group `(Z/p^e Z)^*` for an odd prime `p`.
    fn find_primitive_root(p: u64, e: u32) -> u64 {
        let pe = p.pow(e);
        let order = euler_phi(pe);
        for g in 2..pe {
            if gcd(g, pe) == 1 && multiplicative_order(g, pe) == order {
                return g;
            }
        }
        unreachable!("(Z/p^e Z)^* is cyclic for odd p, so a primitive root exists")
    }

    /// Discrete log of `a` base `g` in a cyclic group of the given `order`,
    /// realized as residues mod `modu` (brute force; `order` is expected to
    /// be small, as this is only used for small built-in demonstration
    /// moduli).
    fn discrete_log_cyclic(a: u64, g: u64, modu: u64, order: u64) -> u64 {
        let target = a % modu;
        let mut x_val = 1u64 % modu;
        for x in 0..order {
            if x_val == target {
                return x;
            }
            x_val = (x_val * g) % modu;
        }
        unreachable!("a is assumed to lie in the cyclic group generated by g")
    }

    /// Order and conductor of the Conrey-labeled Dirichlet character
    /// `chi_modulus(number, ·)`.
    ///
    /// # Panics
    ///
    /// `number` must be coprime to `modulus` and `modulus` must be `>= 1`;
    /// callers (see [`super::LMFDBClient::dirichlet_character`]) are
    /// expected to validate this and return an error instead of calling in
    /// with invalid input.
    pub(crate) fn order_and_conductor(number: u64, modulus: u64) -> (u64, u64) {
        assert!(modulus >= 1, "modulus must be positive");
        assert_eq!(gcd(number, modulus), 1, "number must be coprime to modulus");

        let mut order = 1u64;
        let mut conductor = 1u64;

        for (p, e) in factorize(modulus) {
            let pe = p.pow(e);
            let ni = number % pe;

            if p == 2 {
                if e == 1 {
                    // (Z/2Z)^* is trivial: no contribution.
                } else if e == 2 {
                    // (Z/4Z)^* = {1, 3}, cyclic of order 2, generator 3.
                    let comp_order = 2u64;
                    let x = discrete_log_cyclic(ni, 3, pe, comp_order);
                    let m_i = if x == 0 { 1 } else { comp_order / gcd(x, comp_order) };
                    order = lcm(order, m_i);
                    if m_i > 1 {
                        conductor *= 2u64.pow(v_p(m_i, 2) + 1);
                    }
                } else {
                    // e >= 3: (Z/2^e Z)^* = <-1> x <5> = C2 x C_{2^(e-2)}.
                    let a_bit = if ni % 4 == 1 { 0u64 } else { 1u64 };
                    let base = if a_bit == 1 { (ni * (pe - 1)) % pe } else { ni };
                    let border = 1u64 << (e - 2);
                    let b = discrete_log_cyclic(base, 5, pe, border);

                    let o_a = if a_bit == 1 { 2 } else { 1 };
                    let o_b = if b == 0 { 1 } else { border / gcd(b, border) };
                    order = lcm(order, lcm(o_a, o_b));

                    let j_a = if a_bit == 1 { 2 } else { 0 };
                    let j_b = if b == 0 { 0 } else { e - v_p(b, 2) };
                    conductor *= 2u64.pow(j_a.max(j_b));
                }
            } else {
                // Odd prime power: (Z/p^e Z)^* is cyclic of order phi(p^e).
                let comp_order = euler_phi(pe);
                let g = find_primitive_root(p, e);
                let x = discrete_log_cyclic(ni, g, pe, comp_order);
                let m_i = if x == 0 { 1 } else { comp_order / gcd(x, comp_order) };
                order = lcm(order, m_i);
                if m_i > 1 {
                    conductor *= p.pow(v_p(m_i, p) + 1);
                }
            }
        }

        (order, conductor)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Brute-force reference: builds the actual Conrey character
        /// `chi_q(n, ·)` from the same CRT/generator data as
        /// `order_and_conductor`, then computes its order as the LCM of the
        /// denominators of its values (in lowest terms) and its conductor by
        /// scanning divisors of `q` for the smallest one the character
        /// factors through. This does not use the `order_and_conductor`
        /// shortcut formula at all, so agreement between the two is a real
        /// independent check.
        mod reference {
            use super::gcd;

            #[derive(Clone, Copy)]
            enum Component {
                Trivial,
                Cyclic { pe: u64, g: u64, order: u64 },
                TwoSplit { pe: u64, border: u64 },
            }

            fn factorize(n: u64) -> Vec<(u64, u32)> {
                let mut n = n;
                let mut factors = Vec::new();
                let mut d = 2u64;
                while d * d <= n {
                    if n % d == 0 {
                        let mut e = 0;
                        while n % d == 0 {
                            n /= d;
                            e += 1;
                        }
                        factors.push((d, e));
                    }
                    d += 1;
                }
                if n > 1 {
                    factors.push((n, 1));
                }
                factors
            }

            fn euler_phi_pe(p: u64, e: u32) -> u64 {
                (p - 1) * p.pow(e - 1)
            }

            /// Find a primitive root of the cyclic group `(Z/p^e Z)^*` by
            /// brute force: `g` is a primitive root iff repeatedly
            /// multiplying by `g` visits `order` distinct residues before
            /// returning to `1`.
            fn find_primitive_root(p: u64, e: u32) -> u64 {
                let pe = p.pow(e);
                let order = euler_phi_pe(p, e);
                'cand: for g in 2..pe {
                    if gcd(g, pe) != 1 {
                        continue;
                    }
                    let mut seen = std::collections::HashSet::new();
                    let mut y = 1u64 % pe;
                    for _ in 0..order {
                        if !seen.insert(y) {
                            continue 'cand;
                        }
                        y = (y * g) % pe;
                    }
                    if seen.len() as u64 == order {
                        return g;
                    }
                }
                unreachable!("primitive root must exist for odd prime power")
            }

            fn dlog(a: u64, g: u64, modu: u64, order: u64) -> u64 {
                let target = a % modu;
                let mut x_val = 1u64 % modu;
                for x in 0..order {
                    if x_val == target {
                        return x;
                    }
                    x_val = (x_val * g) % modu;
                }
                unreachable!()
            }

            /// Reference character builder for modulus `q`.
            pub(super) struct Conrey {
                q: u64,
                components: Vec<Component>,
            }

            impl Conrey {
                pub(super) fn new(q: u64) -> Self {
                    let mut components = Vec::new();
                    for (p, e) in factorize(q) {
                        let pe = p.pow(e);
                        if p == 2 {
                            if e == 1 {
                                components.push(Component::Trivial);
                            } else if e == 2 {
                                components.push(Component::Cyclic { pe, g: 3, order: 2 });
                            } else {
                                components.push(Component::TwoSplit {
                                    pe,
                                    border: 1u64 << (e - 2),
                                });
                            }
                        } else {
                            let g = find_primitive_root(p, e);
                            components.push(Component::Cyclic {
                                pe,
                                g,
                                order: euler_phi_pe(p, e),
                            });
                        }
                    }
                    Conrey { q, components }
                }

                /// coordinates of `n` in each component, as (num, denom) pairs
                /// summed as a fraction mod 1 later.
                fn coords(&self, n: u64) -> Vec<(u64, u64, u64, u64)> {
                    // returns per component: (x_num, x_denom, y_num_placeholder..)
                    // simplified: for Cyclic -> (x, order, 0, 1); for TwoSplit -> (a, 2, b, border)
                    self.components
                        .iter()
                        .map(|c| match *c {
                            Component::Trivial => (0, 1, 0, 1),
                            Component::Cyclic { pe, g, order } => {
                                let ni = n % pe;
                                let x = dlog(ni, g, pe, order);
                                (x, order, 0, 1)
                            }
                            Component::TwoSplit { pe, border } => {
                                let ni = n % pe;
                                let a_bit = if ni % 4 == 1 { 0u64 } else { 1u64 };
                                let base = if a_bit == 1 { (ni * (pe - 1)) % pe } else { ni };
                                let b = dlog(base, 5, pe, border);
                                (a_bit, 2, b, border)
                            }
                        })
                        .collect()
                }

                /// chi_q(n, a) as an exact fraction (num, denom) in [0,1).
                fn pairing(&self, n: u64, a: u64) -> (u64, u64) {
                    let nc = self.coords(n);
                    let ac = self.coords(a);
                    // accumulate sum of x*y/d fractions with a common denominator
                    let mut denom = 1u64;
                    for &(_, d1, _, d2) in &nc {
                        denom = lcm_local(denom, d1);
                        denom = lcm_local(denom, d2);
                    }
                    let mut num = 0i128;
                    for ((x, d1, xb, d2), (y, _, yb, _)) in nc.iter().zip(ac.iter()) {
                        num += (*x as i128) * (*y as i128) * (denom as i128) / (*d1 as i128);
                        num += (*xb as i128) * (*yb as i128) * (denom as i128) / (*d2 as i128);
                    }
                    let num = num.rem_euclid(denom as i128) as u64;
                    let g = gcd(num, denom).max(1);
                    (num / g, denom / g)
                }

                /// order and conductor of chi_q(number, ·) via brute force
                /// (no shortcut formula).
                pub(super) fn order_and_conductor_bruteforce(&self, number: u64) -> (u64, u64) {
                    let units: Vec<u64> = (1..self.q).filter(|&a| gcd(a, self.q) == 1).collect();
                    let mut values: Vec<((u64, u64), u64)> = Vec::new();
                    for &a in &units {
                        values.push((self.pairing(number, a), a));
                    }
                    // order = lcm of denominators (in lowest terms, already reduced)
                    let mut order = 1u64;
                    for (( _n, d), _a) in &values {
                        order = lcm_local(order, *d);
                    }
                    // conductor = smallest divisor f of q such that value depends
                    // only on a mod f
                    let mut divisors: Vec<u64> = (1..=self.q).filter(|d| self.q % d == 0).collect();
                    divisors.sort_unstable();
                    for f in divisors {
                        let mut seen: std::collections::HashMap<u64, (u64, u64)> =
                            std::collections::HashMap::new();
                        let mut ok = true;
                        for (val, a) in &values {
                            let key = a % f;
                            if let Some(prev) = seen.get(&key) {
                                if prev != val {
                                    ok = false;
                                    break;
                                }
                            } else {
                                seen.insert(key, *val);
                            }
                        }
                        if ok {
                            return (order, f);
                        }
                    }
                    (order, self.q)
                }
            }

            fn lcm_local(a: u64, b: u64) -> u64 {
                if a == 0 || b == 0 {
                    0
                } else {
                    a / gcd(a, b) * b
                }
            }
        }

        #[test]
        fn order_and_conductor_matches_bruteforce_reference() {
            let moduli = [
                3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 20, 21, 24, 25, 32, 36, 40, 45, 48, 60,
                64, 72, 100, 105, 144,
            ];
            let mut checked = 0;
            for &q in &moduli {
                let reference = reference::Conrey::new(q);
                for n in 1..q {
                    if gcd(n, q) != 1 {
                        continue;
                    }
                    let (ref_order, ref_conductor) =
                        reference.order_and_conductor_bruteforce(n);
                    let (order, conductor) = order_and_conductor(n, q);
                    assert_eq!(
                        (order, conductor),
                        (ref_order, ref_conductor),
                        "mismatch at modulus={q} number={n}"
                    );
                    checked += 1;
                }
            }
            assert!(checked > 400, "sanity: should have checked many pairs, got {checked}");
        }

        #[test]
        fn known_values() {
            // Cross-checked independently against the brute-force reference
            // above (and, for small cases, by hand / against standard
            // Conrey-labeling references).
            let cases: &[(u64, u64, u64, u64)] = &[
                // (modulus, number, expected_order, expected_conductor)
                (5, 1, 1, 1),
                (5, 2, 4, 5),
                (5, 3, 4, 5),
                (5, 4, 2, 5),
                (8, 1, 1, 1),
                (8, 3, 2, 8),
                (8, 5, 2, 8),
                (8, 7, 2, 4),
                (16, 3, 4, 16),
                (16, 7, 2, 8),
                (16, 9, 2, 8),
                (12, 1, 1, 1),
                (12, 5, 2, 3),
                (12, 7, 2, 4),
                (12, 11, 2, 12),
                (7, 3, 6, 7),
                (9, 2, 6, 9),
                (15, 2, 4, 15),
                (15, 4, 2, 5),
                (20, 3, 4, 20),
                (21, 2, 6, 21),
                (1, 1, 1, 1),
                (4, 3, 2, 4),
                (3, 2, 2, 3),
                (100, 3, 20, 100),
                (105, 2, 12, 105),
                (24, 5, 2, 24),
                (24, 7, 2, 4),
                (24, 11, 2, 24),
                (24, 13, 2, 8),
            ];
            for &(q, n, expected_order, expected_conductor) in cases {
                let (order, conductor) = order_and_conductor(n, q);
                assert_eq!(order, expected_order, "order mismatch at q={q} n={n}");
                assert_eq!(
                    conductor, expected_conductor,
                    "conductor mismatch at q={q} n={n}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let _client = LMFDBClient::new();
        let _client2 = LMFDBClient::default();
    }

    #[test]
    fn test_search_elliptic_curves() {
        let client = LMFDBClient::new();
        let curves = client.search_elliptic_curves("11").unwrap();

        assert!(!curves.is_empty());
        assert!(curves.iter().any(|c| c.label == "11.a1"));
    }

    #[test]
    fn test_lookup_elliptic_curve() {
        let client = LMFDBClient::new();
        let curve = client.lookup_elliptic_curve("37.a1").unwrap();

        assert!(curve.is_some());
        let curve = curve.unwrap();
        assert_eq!(curve.conductor, 37);
        assert_eq!(curve.rank, 1);
    }

    #[test]
    fn test_search_modular_forms() {
        let client = LMFDBClient::new();
        let forms = client.search_modular_forms(2, 11).unwrap();

        assert!(!forms.is_empty());
        assert_eq!(forms[0].weight, 2);
        assert_eq!(forms[0].level, 11);
    }

    #[test]
    fn test_search_number_fields() {
        let client = LMFDBClient::new();
        let fields = client.search_number_fields(2).unwrap();

        assert!(!fields.is_empty());
        for field in fields {
            assert_eq!(field.degree, 2);
        }
    }

    #[test]
    fn test_dirichlet_character() {
        let client = LMFDBClient::new();
        let chi = client.dirichlet_character(5, 2).unwrap();

        assert_eq!(chi.modulus, 5);
        assert_eq!(chi.number, 2);
        // chi_5(2, .) has 2 as a primitive root mod 5, so it generates the
        // full character group of order 4 = phi(5); independently verified
        // in the `dirichlet::tests` module.
        assert_eq!(chi.order, 4);
        assert_eq!(chi.conductor, 5);
        assert!(chi.is_primitive);
    }

    #[test]
    fn test_dirichlet_character_trivial() {
        let client = LMFDBClient::new();
        let chi = client.dirichlet_character(5, 1).unwrap();
        assert_eq!(chi.order, 1);
        assert_eq!(chi.conductor, 1);
        assert!(!chi.is_primitive);
    }

    #[test]
    fn test_dirichlet_character_imprimitive() {
        let client = LMFDBClient::new();
        // 8.7: the character mod 8 induced from the unique nontrivial
        // character mod 4 (order 2, conductor 4, not primitive mod 8).
        let chi = client.dirichlet_character(8, 7).unwrap();
        assert_eq!(chi.order, 2);
        assert_eq!(chi.conductor, 4);
        assert!(!chi.is_primitive);
    }

    #[test]
    fn test_dirichlet_character_rejects_non_coprime_number() {
        let client = LMFDBClient::new();
        // gcd(2, 4) = 2 != 1, so "2" is not a valid Conrey number mod 4.
        let err = client.dirichlet_character(4, 2).unwrap_err();
        assert!(matches!(err, LMFDBError::InvalidQuery(_)));
    }

    #[test]
    fn test_dirichlet_character_rejects_zero_modulus() {
        let client = LMFDBClient::new();
        let err = client.dirichlet_character(0, 1).unwrap_err();
        assert!(matches!(err, LMFDBError::InvalidQuery(_)));
    }
}
