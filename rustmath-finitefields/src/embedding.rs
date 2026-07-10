//! Field embeddings `GF(p^m) -> GF(p^n)` for `m | n` (MAGMA Handbook ch. 21).
//!
//! MAGMA source: Chapter 21 §21.4 (`Embed(E, F)`, automatic coercion between
//! compatibly-embedded finite fields).
//!
//! For fields defined by **Conway polynomials** (see [`crate::conway`]) the
//! embedding is the canonical norm-compatible one: the generator of
//! `GF(p^m)` (a root of `C_{p,m}`) is sent to `g^{(p^n-1)/(p^m-1)}` where `g`
//! is the generator of `GF(p^n)` (a root of `C_{p,n}`). By the defining
//! property of Conway polynomials this power *is* a root of `C_{p,m}`, and
//! the resulting maps **commute**: for `k | m | n`,
//! `GF(p^k) -> GF(p^m) -> GF(p^n)` equals `GF(p^k) -> GF(p^n)`.
//!
//! For fields *not* defined by Conway polynomials an embedding still exists
//! whenever `m | n` (any root of the domain's defining polynomial works); we
//! pick the root with the lexicographically smallest coefficient vector so
//! the choice is at least deterministic, and flag the embedding as
//! [`FieldEmbedding::is_conway_compatible`]` == false`: no promise is made
//! that such embeddings commute with other embeddings.

use std::fmt;

use rustmath_core::{MathError, Result, Ring};
use rustmath_integers::Integer;

use crate::finite_field::{FiniteField, FiniteFieldElement};
use crate::poly_factor::{roots, FqPoly};

/// An embedding of finite fields `GF(p^m) -> GF(p^n)` with `m | n`.
///
/// The map is the unique ring homomorphism sending the domain's generator to
/// [`FieldEmbedding::image_of_generator`]; elements are mapped by evaluating
/// their coefficient polynomial at that image ([`FieldEmbedding::apply`]).
#[derive(Clone)]
pub struct FieldEmbedding {
    domain: FiniteField,
    codomain: FiniteField,
    image_of_gen: FiniteFieldElement,
    conway_compatible: bool,
}

impl fmt::Debug for FieldEmbedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FieldEmbedding({} -> {}, gen -> {}{})",
            self.domain,
            self.codomain,
            self.image_of_gen,
            if self.conway_compatible {
                ", Conway-compatible"
            } else {
                ""
            }
        )
    }
}

impl FieldEmbedding {
    /// Construct the embedding `domain -> codomain`.
    ///
    /// Requires equal characteristics and `domain.degree() | codomain.degree()`.
    /// If both fields are defined by Conway polynomials the canonical
    /// norm-compatible embedding is returned (and embeddings between Conway
    /// fields commute); otherwise a deterministic but non-canonical embedding
    /// is returned with `is_conway_compatible() == false`.
    pub fn new(domain: &FiniteField, codomain: &FiniteField) -> Result<Self> {
        if domain.characteristic() != codomain.characteristic() {
            return Err(MathError::InvalidArgument(format!(
                "cannot embed {domain} into {codomain}: different characteristics"
            )));
        }
        let m = domain.degree();
        let n = codomain.degree();
        if n % m != 0 {
            return Err(MathError::InvalidArgument(format!(
                "cannot embed {domain} into {codomain}: {m} does not divide {n}"
            )));
        }
        // Identity on the same field.
        if domain.same_field(codomain) {
            return Ok(FieldEmbedding {
                domain: domain.clone(),
                codomain: codomain.clone(),
                image_of_gen: codomain.generator(),
                conway_compatible: true,
            });
        }
        // Prime subfield: k mod p -> k mod p, canonical for any codomain.
        // (The "generator" of GF(p) = F_p[x]/(x) is the class of x, i.e. 0.)
        if domain.is_prime_field() {
            return Ok(FieldEmbedding {
                domain: domain.clone(),
                codomain: codomain.clone(),
                image_of_gen: codomain.zero(),
                conway_compatible: true,
            });
        }
        // Conway-compatible path: g_m -> g_n^{(p^n-1)/(p^m-1)}.
        if domain.is_conway() && codomain.is_conway() {
            let e = (codomain.order() - Integer::one()) / (domain.order() - Integer::one());
            let image = codomain.generator().pow_int(&e);
            let emb = FieldEmbedding {
                domain: domain.clone(),
                codomain: codomain.clone(),
                image_of_gen: image,
                conway_compatible: true,
            };
            // The Conway property guarantees this; verify anyway (cheap).
            if !emb.image_is_root_of_domain_modulus() {
                return Err(MathError::NumericalError(
                    "Conway compatibility violated: table entry inconsistent".into(),
                ));
            }
            return Ok(emb);
        }
        // General path: send the generator to a root of the domain's defining
        // polynomial in the codomain (exists since m | n). Deterministic
        // choice: lexicographically smallest coefficient vector.
        let coeffs: Vec<FiniteFieldElement> = domain
            .defining_polynomial()
            .iter()
            .map(|c| codomain.from_int(c.clone()))
            .collect();
        let f = FqPoly::new(codomain.clone(), coeffs);
        let mut rs = roots(&f);
        if rs.is_empty() {
            // Cannot happen for an irreducible modulus of degree m | n.
            return Err(MathError::NumericalError(format!(
                "no root of the defining polynomial of {domain} found in {codomain}"
            )));
        }
        rs.sort_by(|a, b| a.eltseq().cmp(b.eltseq()));
        Ok(FieldEmbedding {
            domain: domain.clone(),
            codomain: codomain.clone(),
            image_of_gen: rs.swap_remove(0),
            conway_compatible: false,
        })
    }

    /// The domain `GF(p^m)`.
    pub fn domain(&self) -> &FiniteField {
        &self.domain
    }

    /// The codomain `GF(p^n)`.
    pub fn codomain(&self) -> &FiniteField {
        &self.codomain
    }

    /// The image of the domain's generator in the codomain.
    pub fn image_of_generator(&self) -> &FiniteFieldElement {
        &self.image_of_gen
    }

    /// Whether this is the canonical norm-compatible (Conway) embedding.
    /// Only Conway-compatible embeddings are guaranteed to commute with each
    /// other; non-Conway embeddings are deterministic but arbitrary.
    pub fn is_conway_compatible(&self) -> bool {
        self.conway_compatible
    }

    /// Apply the embedding to `elt` (must lie in the domain).
    pub fn apply(&self, elt: &FiniteFieldElement) -> Result<FiniteFieldElement> {
        if !elt.field().same_field(&self.domain) {
            return Err(MathError::InvalidArgument(
                "element does not lie in the domain of this embedding".into(),
            ));
        }
        // Horner evaluation of the coefficient polynomial at image_of_gen.
        let mut acc = self.codomain.zero();
        for c in elt.eltseq().iter().rev() {
            acc = acc * self.image_of_gen.clone() + self.codomain.from_int(c.clone());
        }
        Ok(acc)
    }

    /// Compose with a second embedding: `self.compose(&g) = g ∘ self`,
    /// an embedding `self.domain() -> g.codomain()`. Requires
    /// `self.codomain() == g.domain()`.
    pub fn compose(&self, second: &FieldEmbedding) -> Result<FieldEmbedding> {
        if !self.codomain.same_field(&second.domain) {
            return Err(MathError::InvalidArgument(
                "cannot compose: codomain of the first embedding differs from the domain of the second".into(),
            ));
        }
        Ok(FieldEmbedding {
            domain: self.domain.clone(),
            codomain: second.codomain.clone(),
            image_of_gen: second.apply(&self.image_of_gen)?,
            conway_compatible: self.conway_compatible && second.conway_compatible,
        })
    }

    /// Check `f(image_of_gen) == 0` for the domain's defining polynomial `f`
    /// (i.e. that the map defined by `image_of_gen` really is a ring
    /// homomorphism).
    fn image_is_root_of_domain_modulus(&self) -> bool {
        let mut acc = self.codomain.zero();
        for c in self.domain.defining_polynomial().iter().rev() {
            acc = acc * self.image_of_gen.clone() + self.codomain.from_int(c.clone());
        }
        acc.is_zero()
    }
}

impl FiniteField {
    /// The embedding of this field into `codomain` (MAGMA `Embed`); see
    /// [`FieldEmbedding::new`].
    pub fn embedding_into(&self, codomain: &FiniteField) -> Result<FieldEmbedding> {
        FieldEmbedding::new(self, codomain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gf(p: i64, n: usize) -> FiniteField {
        FiniteField::new(Integer::from(p), n).unwrap()
    }

    /// All p^n elements of the field (little-endian base-p counter).
    fn all_elements(field: &FiniteField) -> Vec<FiniteFieldElement> {
        let p = field
            .characteristic()
            .clone();
        let n = field.degree();
        let mut out = Vec::new();
        let mut idx = vec![Integer::zero(); n];
        'outer: loop {
            out.push(field.element(idx.clone()));
            let mut i = 0;
            loop {
                if i == n {
                    break 'outer;
                }
                idx[i] = idx[i].clone() + Integer::one();
                if idx[i] < p {
                    break;
                }
                idx[i] = Integer::zero();
                i += 1;
            }
        }
        out
    }

    #[test]
    fn conway_image_is_norm_compatible_power() {
        // GF(4) -> GF(16): generator maps to g16^5, an element of order 3,
        // whose minimal polynomial is the GF(4) Conway polynomial x^2+x+1.
        let f4 = gf(2, 2);
        let f16 = gf(2, 4);
        let emb = FieldEmbedding::new(&f4, &f16).unwrap();
        assert!(emb.is_conway_compatible());
        let expected = f16.generator().pow_int(&Integer::from(5));
        assert_eq!(*emb.image_of_generator(), expected);
        assert_eq!(
            emb.image_of_generator().multiplicative_order(),
            Some(Integer::from(3))
        );
        assert_eq!(
            emb.image_of_generator().minimal_polynomial(),
            f4.defining_polynomial().to_vec()
        );
    }

    #[test]
    fn embedding_is_ring_homomorphism() {
        // GF(9) -> GF(729): check + and * are preserved on ALL 81 pairs.
        let f9 = gf(3, 2);
        let f729 = gf(3, 6);
        let emb = FieldEmbedding::new(&f9, &f729).unwrap();
        assert!(emb.is_conway_compatible());
        assert!(emb.apply(&f9.one()).unwrap().is_one());
        let elts = all_elements(&f9);
        assert_eq!(elts.len(), 9);
        for a in &elts {
            for b in &elts {
                let sum = emb.apply(&(a.clone() + b.clone())).unwrap();
                assert_eq!(
                    sum,
                    emb.apply(a).unwrap() + emb.apply(b).unwrap(),
                    "embedding does not preserve + on {a}, {b}"
                );
                let prod = emb.apply(&(a.clone() * b.clone())).unwrap();
                assert_eq!(
                    prod,
                    emb.apply(a).unwrap() * emb.apply(b).unwrap(),
                    "embedding does not preserve * on {a}, {b}"
                );
            }
        }
        // Injectivity on this small field.
        for (i, a) in elts.iter().enumerate() {
            for b in elts.iter().skip(i + 1) {
                assert_ne!(emb.apply(a).unwrap(), emb.apply(b).unwrap());
            }
        }
    }

    #[test]
    fn tower_gf4_gf16_gf256_commutes() {
        // GF(2^2) -> GF(2^4) -> GF(2^8) must equal GF(2^2) -> GF(2^8).
        let f4 = gf(2, 2);
        let f16 = gf(2, 4);
        let f256 = gf(2, 8);
        let e_4_16 = FieldEmbedding::new(&f4, &f16).unwrap();
        let e_16_256 = FieldEmbedding::new(&f16, &f256).unwrap();
        let direct = FieldEmbedding::new(&f4, &f256).unwrap();
        assert!(e_4_16.is_conway_compatible());
        assert!(e_16_256.is_conway_compatible());
        assert!(direct.is_conway_compatible());

        let composed = e_4_16.compose(&e_16_256).unwrap();
        assert!(composed.is_conway_compatible());
        assert_eq!(composed.image_of_generator(), direct.image_of_generator());

        // Element-wise, on all 4 elements of GF(4).
        for a in all_elements(&f4) {
            let via_tower = e_16_256.apply(&e_4_16.apply(&a).unwrap()).unwrap();
            let via_direct = direct.apply(&a).unwrap();
            assert_eq!(via_tower, via_direct, "tower disagrees on {a}");
            assert_eq!(composed.apply(&a).unwrap(), via_direct);
        }
    }

    #[test]
    fn tower_odd_characteristic_commutes() {
        // GF(3) -> GF(3^2) -> GF(3^6) equals GF(3) -> GF(3^6),
        // and GF(3^2) -> GF(3^6) is Conway-compatible.
        let f3 = gf(3, 1);
        let f9 = gf(3, 2);
        let f729 = gf(3, 6);
        let e1 = FieldEmbedding::new(&f3, &f9).unwrap();
        let e2 = FieldEmbedding::new(&f9, &f729).unwrap();
        let direct = FieldEmbedding::new(&f3, &f729).unwrap();
        for a in all_elements(&f3) {
            assert_eq!(
                e2.apply(&e1.apply(&a).unwrap()).unwrap(),
                direct.apply(&a).unwrap()
            );
        }
        // Prime-subfield embedding is the obvious coercion.
        for k in 0..3i64 {
            let a = f3.from_int(Integer::from(k));
            assert_eq!(
                direct.apply(&a).unwrap(),
                f729.from_int(Integer::from(k))
            );
        }
    }

    #[test]
    fn non_conway_codomain_falls_back_with_honest_flag() {
        // GF(2^4) defined by x^4 + x^3 + 1 (irreducible, but not the Conway
        // polynomial x^4 + x + 1). Embedding GF(4) into it still exists but
        // must be flagged as not Conway-compatible.
        let f4 = gf(2, 2);
        let f16_alt = FiniteField::with_modulus(
            Integer::from(2),
            vec![
                Integer::from(1),
                Integer::from(0),
                Integer::from(0),
                Integer::from(1),
                Integer::from(1),
            ],
        )
        .unwrap();
        assert!(!f16_alt.is_conway());
        let emb = FieldEmbedding::new(&f4, &f16_alt).unwrap();
        assert!(!emb.is_conway_compatible());
        // Still a ring homomorphism: image of the generator is a root of
        // x^2 + x + 1, and + / * are preserved.
        let img = emb.image_of_generator().clone();
        let val = img.clone() * img.clone() + img.clone() + f16_alt.one();
        assert!(val.is_zero());
        for a in all_elements(&f4) {
            for b in all_elements(&f4) {
                assert_eq!(
                    emb.apply(&(a.clone() * b.clone())).unwrap(),
                    emb.apply(&a).unwrap() * emb.apply(&b).unwrap()
                );
                assert_eq!(
                    emb.apply(&(a.clone() + b.clone())).unwrap(),
                    emb.apply(&a).unwrap() + emb.apply(&b).unwrap()
                );
            }
        }
    }

    #[test]
    fn identity_embedding() {
        let f8 = gf(2, 3);
        let emb = f8.embedding_into(&f8).unwrap();
        for a in all_elements(&f8) {
            assert_eq!(emb.apply(&a).unwrap(), a);
        }
    }

    #[test]
    fn rejects_bad_embeddings() {
        // 2 does not divide 3.
        assert!(FieldEmbedding::new(&gf(2, 2), &gf(2, 3)).is_err());
        // Different characteristics.
        assert!(FieldEmbedding::new(&gf(2, 1), &gf(3, 2)).is_err());
        // Element of the wrong field.
        let emb = FieldEmbedding::new(&gf(2, 2), &gf(2, 4)).unwrap();
        assert!(emb.apply(&gf(2, 4).generator()).is_err());
        // Composition mismatch.
        let other = FieldEmbedding::new(&gf(2, 3), &gf(2, 6)).unwrap();
        assert!(emb.compose(&other).is_err());
    }
}
