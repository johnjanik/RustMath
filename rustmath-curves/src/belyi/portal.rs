//! General genus-0 ansatz from cycle types.
//!
//! Ported from `dessin_engine/src/portal.rs` in
//! `/home/john/inverse_galois/M23/dessin_engine`. Only the construction layer is
//! ported here (Agent C, first half): the [`Portal`] descriptor, its
//! [`genus`](Portal::genus)/[`degree`](Portal::degree), and the genus-0
//! [`ansatz`](Portal::ansatz)/[`encode`](Portal::encode) that build a Belyi system
//! from the three branch cycle types alone. The conic-outcome aggregation
//! (`PortalConic`/`UniformLaw`, which needs `rustmath-quadraticforms`) is part of
//! the later descent/decide half and is deliberately not ported here.
//!
//! A genus-0 portal is given by its three branch cycle types (partitions of `n`).
//! [`Portal::ansatz`] groups each fiber's cycles by length into `form^length`
//! factors and applies the generic PGL₂ + scaling normalization that, for genus 0
//! (`c₀+c₁+c∞ = n+2`), makes the system exactly square.

use crate::belyi::encode::{BelyiAnsatzSystem, Coeff, Encoded, FactorSpec};
use std::collections::BTreeMap;

/// A genus-0 portal by its three branch cycle types (each a partition of `n`).
#[derive(Debug, Clone)]
pub struct Portal {
    pub name: String,
    pub ct0: Vec<usize>,
    pub ct1: Vec<usize>,
    pub ctinf: Vec<usize>,
}

/// Group a cycle type by length: `length → count`, descending in length.
fn by_length(ct: &[usize]) -> Vec<(usize, usize)> {
    let mut m: BTreeMap<usize, usize> = BTreeMap::new();
    for &l in ct {
        *m.entry(l).or_insert(0) += 1;
    }
    let mut v: Vec<(usize, usize)> = m.into_iter().collect();
    v.sort_by(|a, b| b.0.cmp(&a.0)); // length descending
    v
}

impl Portal {
    pub fn degree(&self) -> usize {
        self.ct0.iter().sum()
    }

    /// Riemann–Hurwitz genus: `1 − n + ½ Σ(n − #cycles)`.
    pub fn genus(&self) -> i64 {
        let n = self.degree() as i64;
        let defect = |ct: &[usize]| n - ct.len() as i64;
        1 - n + (defect(&self.ct0) + defect(&self.ct1) + defect(&self.ctinf)) / 2
    }

    /// Build the Belyi ansatz system from the cycle types (genus-0 only).
    /// Returns `None` if not genus 0.
    pub fn ansatz(&self) -> Option<BelyiAnsatzSystem> {
        if self.genus() != 0 {
            return None;
        }
        // forms per fiber: (mult = length, degree = count, name)
        let make = |ct: &[usize], pre: &str| -> Vec<(usize, usize, String)> {
            by_length(ct)
                .into_iter()
                .map(|(len, cnt)| (len, cnt, format!("{pre}{len}")))
                .collect()
        };
        let zero = make(&self.ct0, "z");
        let inf = make(&self.ctinf, "w");
        let one = make(&self.ct1, "v");

        // designate the PGL2/scaling anchors:
        //  A1 = first zero-form (translation a0=0, scale a1=1, monic),
        //  R1 = first inf-form  (pole r_top=0, monic r_{top-1}=1),
        //  FREE = last form that is neither A1 nor R1 (its scaling is left free).
        let flat: Vec<(char, usize)> = zero
            .iter()
            .enumerate()
            .map(|(i, _)| ('z', i))
            .chain(inf.iter().enumerate().map(|(i, _)| ('w', i)))
            .chain(one.iter().enumerate().map(|(i, _)| ('v', i)))
            .collect();
        let a1 = ('z', 0);
        let r1 = ('w', 0);
        let free = *flat.iter().rev().find(|&&t| t != a1 && t != r1)?;

        let coeffs_for = |fiber: char, idx: usize, deg: usize, name: &str| -> Vec<Coeff> {
            let mut c: Vec<Coeff> = (0..=deg)
                .map(|i| Coeff::Unknown(format!("{name}_{i}")))
                .collect();
            let here = (fiber, idx);
            if here == a1 {
                c[0] = Coeff::Fixed(0); // root at [0:1]
                if deg >= 1 {
                    c[1] = Coeff::Fixed(1); // x-scale
                }
                c[deg] = Coeff::Fixed(1); // monic
            } else if here == r1 {
                c[deg] = Coeff::Fixed(0); // pole at [1:0]
                if deg >= 1 {
                    c[deg - 1] = Coeff::Fixed(1); // monic on the next coeff
                }
            } else if here == free {
                // leave fully unknown (its scaling absorbs the residual freedom)
            } else {
                c[deg] = Coeff::Fixed(1); // monic
            }
            c
        };

        let build = |forms: &[(usize, usize, String)], fiber: char| -> Vec<FactorSpec> {
            forms
                .iter()
                .enumerate()
                .map(|(idx, (mult, deg, name))| FactorSpec {
                    mult: *mult as u32,
                    coeffs: coeffs_for(fiber, idx, *deg, name),
                })
                .collect()
        };

        Some(BelyiAnsatzSystem {
            zero: build(&zero, 'z'),
            inf: build(&inf, 'w'),
            one: build(&one, 'v'),
            c: Coeff::Unknown("c".into()),
        })
    }

    /// Encode the portal's system (genus-0 only).
    pub fn encode(&self) -> Option<Encoded> {
        Some(self.ansatz()?.encode())
    }
}

/// The `[2,12,5]` portal (cycle types `2⁸1⁸ / 12² / 5⁴1⁴`).
pub fn portal_2_12_5() -> Portal {
    let mut ct0 = vec![2usize; 8];
    ct0.extend(vec![1usize; 8]);
    let mut ctinf = vec![5usize; 4];
    ctinf.extend(vec![1usize; 4]);
    Portal {
        name: "[2,12,5]".into(),
        ct0,
        ct1: vec![12, 12],
        ctinf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portal_2_12_5_is_genus0_and_square() {
        let p = portal_2_12_5();
        assert_eq!(p.degree(), 24);
        assert_eq!(p.genus(), 0);
        let enc = p.encode().unwrap();
        // generic normalization yields the same square 25x25 system as the ansatz
        assert_eq!(enc.degree, 24);
        assert_eq!(enc.system.num_variables(), 25);
        assert_eq!(enc.system.num_equations(), 25);
    }

    #[test]
    fn another_genus0_portal_is_square() {
        // A valid genus-0 portal: 2^8 1^8 / 6^4 / 4^6, all summing to 24;
        // #cycles 16,4,6 -> defects 8,20,18 = 46 -> g = 1-24+23 = 0.
        let mut ct0 = vec![2usize; 8];
        ct0.extend(vec![1usize; 8]);
        let p = Portal {
            name: "[2,4,6]".into(),
            ct0,
            ct1: vec![6, 6, 6, 6],
            ctinf: vec![4, 4, 4, 4, 4, 4],
        };
        assert_eq!(p.genus(), 0);
        let enc = p.encode().unwrap();
        assert_eq!(enc.system.num_variables(), enc.system.num_equations());
    }

    #[test]
    fn non_genus0_portal_has_no_ansatz() {
        // n=4: 4/4/2^2: #cycles 1,1,2, defects 3,3,2 = 8, g = 1-4+4 = 1.
        let p1 = Portal {
            name: "g1".into(),
            ct0: vec![4],
            ct1: vec![4],
            ctinf: vec![2, 2],
        };
        assert_eq!(p1.genus(), 1);
        assert!(p1.ansatz().is_none());
    }
}
