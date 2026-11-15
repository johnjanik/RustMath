//! Differential forms on manifolds
//!
//! This module implements differential p-forms, which are totally antisymmetric
//! covariant tensor fields. Includes:
//! - Exterior derivative
//! - Wedge product
//! - Interior product
//! - Lie derivative (Cartan's formula)
//! - An exact three-valued decision whether a form vanishes ([`DiffForm::vanishes`])
//!
//! # References
//!
//! * J. M. Lee, *Introduction to Smooth Manifolds*, 2nd ed., GTM 218, Springer 2013, ch. 14:
//!   "The Algebra of Alternating Tensors", "The Wedge Product", "The Exterior Derivative",
//!   "Interior Multiplication", "Lie Derivatives of Differential Forms".
//! * M. Spivak, *Calculus on Manifolds*, Benjamin 1965, ch. 4.
//! * MAGMA Handbook: Chapter 82 §82.3 "Exterior Algebras" (the algebra of alternating forms),
//!   Chapter 42 §42.15 "Differentials" (`Differential(a)`), Chapter 111 "Differential Rings"
//!   (derivations). Digests in `rustmath-lava/docs/`. MAGMA has no smooth-manifold chapter, and
//!   the handbook has no examples of forms on manifolds to port; the tests use PARI/GP and
//!   SageMath reference values.
//! * SageMath, `sage.manifolds.differentiable.diff_form`: the API reference and the convention.
//!
//! # Convention
//!
//! The determinant convention (Lee 2013, ch. 14; also SageMath's): a p-form is
//! ω = Σ_{I increasing} ω_I dx^I, and its components are those of the alternating tensor,
//! ω_{i_1...i_p} = ω(∂_{i_1}, ..., ∂_{i_p}). So ω_{σ(I)} = sgn(σ) ω_I, and a component with a
//! repeated index is 0. Then
//!
//! * (dω)_I = Σ_k (-1)^k ∂_{i_k} ω_{I \ i_k} for I = (i_0 < ... < i_p),
//! * (ω ∧ η)_I = Σ_{A ⊂ I, |A| = p} sgn(A, I \ A) ω_A η_{I \ A},
//! * (i_X ω)_J = Σ_k X^k ω_{kJ}.
//!
//! # Storage and invariants
//!
//! Storage is dense: per chart a vector of n^p slots in the layout of
//! [`crate::tensor_field::MultiIndex::to_flat`] (first index most significant).
//!
//! * (I1) In every chart, the slot of an increasing tuple I holds an expression v_I; the slot
//!   σ(I) holds an expression equal in value to sgn(σ) v_I; a slot with a repeated index holds
//!   the literal `Expr::Integer(0)`. Every constructor establishes it (antisymmetry is checked
//!   with the exact zero test of [`crate::expr_zero`], and the vector is rebuilt from its
//!   independent components); every operation preserves it. Structural equality of forms
//!   (`PartialEq`) is not part of the invariant.
//! * (I2) A form of degree > n is zero. Operations return it as [`DiffForm::new`] (no
//!   components, nothing allocated).
//!
//! Every result component is simplified and put through the zero test, so d∘d of a polynomial
//! form and α∧α of a 1-form are literal zeros. Components must be finite (no `oo`, no NaN or
//! infinite `Real`): the constructors refuse others.
//!
//! # Preconditions of the exterior derivative
//!
//! * rustmath-symbolic's `differentiate` returns 0 for named functions, `mod`, gamma,
//!   factorial, zeta and a `root_sum` whose polynomial contains the variable, and an
//!   approximate coefficient for erf. d refuses such a component with `Err(NotImplemented)`
//!   (see [`crate::expr_zero`]) instead of returning a fabricated derivative.
//! * Components are stored by chart name. d checks the chart's dimension and repeated
//!   coordinates, and compares the coordinates with the manifold's registered chart of that
//!   name (the first one, if several share the name). A chart that is not registered is
//!   trusted: it must be the chart the components were written in.

use crate::chart::Chart;
use crate::differentiable::DifferentiableManifold;
use crate::errors::{ManifoldError, Result};
use crate::expr_zero::{check_derivative, is_finite_expr, normalize, sum_signed, zero_test, ZeroTest};
use crate::tensor_field::{component_count, flat_index, zeroed_components, ChartId, TensorField};
use crate::vector_field::VectorField;
use rustmath_symbolic::Expr;
use std::collections::{HashMap, HashSet};
use std::ops::{Add, Neg};
use std::sync::Arc;

/// A differential p-form
///
/// A p-form is a totally antisymmetric covariant tensor of rank p.
/// In local coordinates: ω = Σ_{i₁<...<iₚ} ω_{i₁...iₚ} dx^{i₁} ∧ ... ∧ dx^{iₚ}
#[derive(Clone, Debug)]
pub struct DiffForm {
    /// Base tensor field (totally antisymmetric covariant tensor)
    tensor: TensorField,
    /// Degree (p) of the form
    degree: usize,
}

// ---------------------------------------------------------------------------------------------
// Combinatorial helpers
// ---------------------------------------------------------------------------------------------

/// The first k-subset of 0..n in lexicographic order, `None` when k > n.
fn first_comb(k: usize, n: usize) -> Option<Vec<usize>> {
    if k > n {
        None
    } else {
        Some((0..k).collect())
    }
}

/// Advances `c` to the next k-subset of 0..n in lexicographic order; false after the last.
fn next_comb(c: &mut [usize], n: usize) -> bool {
    let k = c.len();
    let mut i = k;
    while i > 0 {
        i -= 1;
        if c[i] < n - k + i {
            c[i] += 1;
            for j in i + 1..k {
                c[j] = c[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// Calls `f(perm, even)` for every permutation of `t` (Heap's algorithm), the identity first;
/// `even` is the parity of the permutation.
fn for_each_perm(t: &[usize], mut f: impl FnMut(&[usize], bool)) {
    let mut a = t.to_vec();
    let k = a.len();
    let mut c = vec![0usize; k];
    let mut even = true;
    f(&a, even);
    let mut i = 1;
    while i < k {
        if c[i] < i {
            if i % 2 == 0 {
                a.swap(0, i);
            } else {
                a.swap(c[i], i);
            }
            even = !even;
            f(&a, even);
            c[i] += 1;
            i = 1;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
}

/// The simplified negation of `e`.
fn neg(e: &Expr) -> Expr {
    (-e.clone()).simplify()
}

/// Writes `v` at every even and `neg(v)` at every odd permutation of the increasing tuple `t`.
/// Nothing for a literal-zero `v` (the slots are zero already).
fn scatter(out: &mut [Expr], t: &[usize], n: usize, v: &Expr) {
    if v.is_zero() {
        return;
    }
    let nv = neg(v);
    for_each_perm(t, |p, even| {
        out[flat_index(p, n)] = if even { v.clone() } else { nv.clone() };
    });
}

/// The digits of `o` in base `n` (first most significant) into `digits`.
fn decode(mut o: usize, n: usize, digits: &mut [usize]) {
    for d in digits.iter_mut().rev() {
        *d = o % n;
        o /= n;
    }
}

/// C(n, k), for k <= n with n^k storable (so every intermediate fits in u128).
fn binomial(n: usize, k: usize) -> Option<usize> {
    let mut c: u128 = 1;
    for i in 0..k {
        c = c.checked_mul((n - i) as u128)? / (i as u128 + 1);
    }
    usize::try_from(c).ok()
}

/// Checks the chart used for a derivative in coordinates: its dimension is dim(M), its
/// coordinate names are distinct, and when the manifold has a registered chart of the same name
/// (the first one), the coordinate names agree. Components are stored by chart name, so a chart
/// that is not registered cannot be checked further. `Err(InvalidChart)`.
pub(crate) fn check_coordinate_chart(manifold: &DifferentiableManifold, chart: &Chart) -> Result<()> {
    let n = manifold.dimension();
    if chart.dimension() != n || chart.coordinate_names().len() != n {
        return Err(ManifoldError::InvalidChart(format!(
            "the chart '{}' has dimension {}, but the manifold has dimension {}",
            chart.name(),
            chart.dimension(),
            n
        )));
    }
    let names = chart.coordinate_names();
    let mut seen = HashSet::with_capacity(names.len());
    for name in names {
        if !seen.insert(name.as_str()) {
            return Err(ManifoldError::InvalidChart(format!(
                "the chart '{}' repeats the coordinate {}",
                chart.name(),
                name
            )));
        }
    }
    if let Ok(registered) = manifold.chart(chart.name()) {
        if registered.coordinate_names() != names {
            return Err(ManifoldError::InvalidChart(format!(
                "the chart '{}' has coordinates ({}), but the manifold's chart of that name has \
                 coordinates ({}); components are stored by chart name",
                chart.name(),
                names.join(", "),
                registered.coordinate_names().join(", ")
            )));
        }
    }
    Ok(())
}

/// Validates dense p-form components in place (invariant I1): every slot finite; for p >= 2,
/// every slot with a repeated index decided zero and every permuted slot decided equal to the
/// signed increasing slot; then rebuilds the permuted slots from the increasing ones.
fn validate_components(comps: &mut [Expr], n: usize, p: usize) -> Result<()> {
    let mut digits = vec![0usize; p];
    for (o, c) in comps.iter().enumerate() {
        if !is_finite_expr(c) {
            if n > 0 {
                decode(o, n, &mut digits);
            }
            return Err(ManifoldError::ValidationError(format!(
                "the component {:?} is {}, which is not finite",
                digits, c
            )));
        }
    }
    if p < 2 || n == 0 {
        return Ok(());
    }
    // slots with a repeated index
    let mut sorted = vec![0usize; p];
    for (o, c) in comps.iter_mut().enumerate() {
        decode(o, n, &mut digits);
        sorted.copy_from_slice(&digits);
        sorted.sort_unstable();
        if sorted.windows(2).all(|w| w[0] != w[1]) {
            continue;
        }
        if !c.is_zero() {
            match zero_test(c) {
                ZeroTest::Zero => {}
                ZeroTest::NonZero => {
                    return Err(ManifoldError::ValidationError(format!(
                        "the components are not antisymmetric: {:?} has a repeated index and is {}",
                        digits, c
                    )))
                }
                ZeroTest::Unresolved => {
                    return Err(ManifoldError::Undecided(format!(
                        "cannot verify that the component {:?}, which has a repeated index, is zero",
                        digits
                    )))
                }
            }
        }
        *c = Expr::from(0);
    }
    // permutations of the increasing tuples
    let Some(mut t) = first_comb(p, n) else {
        return Ok(());
    };
    loop {
        let v = comps[flat_index(&t, n)].clone();
        let mut nv: Option<Expr> = None;
        let mut failure: Option<ManifoldError> = None;
        for_each_perm(&t, |perm, even| {
            if failure.is_some() {
                return;
            }
            let c = &comps[flat_index(perm, n)];
            if even {
                if *c == v {
                    return;
                }
            } else if *c == *nv.get_or_insert_with(|| neg(&v)) {
                return;
            }
            let diff = if even { c.clone() - v.clone() } else { c.clone() + v.clone() };
            match zero_test(&diff) {
                ZeroTest::Zero => {}
                ZeroTest::NonZero => {
                    failure = Some(ManifoldError::ValidationError(format!(
                        "the components are not antisymmetric: {:?} is {} and {:?} is {}",
                        t, v, perm, c
                    )))
                }
                ZeroTest::Unresolved => {
                    failure = Some(ManifoldError::Undecided(format!(
                        "cannot verify the antisymmetry of {:?} and {:?}",
                        t, perm
                    )))
                }
            }
        });
        if let Some(e) = failure {
            return Err(e);
        }
        // rebuild (I1): v at the even, neg(v) at the odd permutations, literal zeros for zero
        if v.is_zero() {
            for_each_perm(&t, |perm, _| comps[flat_index(perm, n)] = Expr::from(0));
        } else {
            let nv = nv.unwrap_or_else(|| neg(&v));
            for_each_perm(&t, |perm, even| {
                comps[flat_index(perm, n)] = if even { v.clone() } else { nv.clone() };
            });
        }
        if !next_comb(&mut t, n) {
            break;
        }
    }
    Ok(())
}

impl DiffForm {
    /// Create a new differential form
    ///
    /// It has no components. By invariant I2 it is the zero form of degree > n; for degree
    /// <= n the operations that need components return `Err(NoComponentsInChart)`.
    pub fn new(manifold: Arc<DifferentiableManifold>, degree: usize) -> Self {
        let tensor = TensorField::new(manifold.clone(), 0, degree);
        Self { tensor, degree }
    }

    /// Create from a covariant tensor field of rank `degree`.
    ///
    /// In every chart the tensor has components in, each component must be finite, and for
    /// degree >= 2 the components must be antisymmetric: a slot with a repeated index decided
    /// zero, and every permuted slot decided equal to the signed increasing slot by the exact
    /// zero test. Then the vector is rebuilt from the increasing slots (invariant I1).
    ///
    /// # Errors
    ///
    /// `InvalidTensorRank`, `DimensionMismatch` for the type; `ValidationError` for a
    /// component that is not finite or not antisymmetric; `Undecided` when the antisymmetry
    /// cannot be verified.
    pub fn from_tensor(tensor: TensorField, degree: usize) -> Result<Self> {
        if tensor.contravariant_rank() != 0 {
            return Err(ManifoldError::InvalidTensorRank);
        }
        if tensor.covariant_rank() != degree {
            return Err(ManifoldError::DimensionMismatch {
                expected: degree,
                actual: tensor.covariant_rank(),
            });
        }
        let mut tensor = tensor;
        let n = tensor.manifold().dimension();
        for comps in tensor.chart_components_mut().values_mut() {
            validate_components(comps, n, degree)?;
        }
        Ok(Self { tensor, degree })
    }

    /// Create a differential form from components
    ///
    /// # Arguments
    ///
    /// * `manifold` - The differentiable manifold
    /// * `chart` - The chart for the components
    /// * `degree` - The degree (p) of the form
    /// * `components` - The n^p dense components of the alternating tensor in the given chart
    ///
    /// # Errors
    ///
    /// The errors of [`DiffForm::from_tensor`], and those of `TensorField::from_components`
    /// for the length.
    pub fn from_components(
        manifold: Arc<DifferentiableManifold>,
        chart: &Chart,
        degree: usize,
        components: Vec<Expr>,
    ) -> Result<Self> {
        let tensor = TensorField::from_components(manifold, 0, degree, chart, components)?;
        Self::from_tensor(tensor, degree)
    }

    /// A p-form from its independent components: `values[r]` is ω_I for the r-th strictly
    /// increasing p-tuple I in lexicographic order.
    ///
    /// # Errors
    ///
    /// `DimensionMismatch` unless `values.len()` is C(n, p) (for p > n, `values` must be empty
    /// and the result is [`DiffForm::new`]); `ValidationError` for a component that is not
    /// finite; the errors of [`component_count`] and of the allocation.
    pub fn from_independent_components(
        manifold: Arc<DifferentiableManifold>,
        chart: &Chart,
        degree: usize,
        values: Vec<Expr>,
    ) -> Result<Self> {
        let n = manifold.dimension();
        if degree > n {
            if !values.is_empty() {
                return Err(ManifoldError::DimensionMismatch { expected: 0, actual: values.len() });
            }
            return Ok(DiffForm::new(manifold, degree));
        }
        let count = component_count(n, degree)?;
        let expected = binomial(n, degree).ok_or(ManifoldError::ComponentCountOverflow {
            dimension: n,
            rank: degree,
        })?;
        if values.len() != expected {
            return Err(ManifoldError::DimensionMismatch { expected, actual: values.len() });
        }
        if let Some((r, v)) = values.iter().enumerate().find(|(_, v)| !is_finite_expr(v)) {
            return Err(ManifoldError::ValidationError(format!(
                "the independent component {} is {}, which is not finite",
                r, v
            )));
        }
        let mut out = zeroed_components(count)?;
        if let Some(mut t) = first_comb(degree, n) {
            let mut it = values.iter();
            loop {
                if let Some(v) = it.next() {
                    scatter(&mut out, &t, n, v);
                }
                if !next_comb(&mut t, n) {
                    break;
                }
            }
        }
        Ok(DiffForm {
            tensor: TensorField::from_dense_unchecked(manifold, 0, degree, chart, out),
            degree,
        })
    }

    /// The independent components in `chart`: (I, ω_I) for the strictly increasing p-tuples I,
    /// in lexicographic order. Empty for p > n. `Err(NoComponentsInChart)`.
    pub fn independent_components(&self, chart: &Chart) -> Result<Vec<(Vec<usize>, Expr)>> {
        let n = self.manifold().dimension();
        if self.degree > n {
            return Ok(Vec::new());
        }
        let w = self.tensor.components_ref(chart)?;
        let mut out = Vec::new();
        if let Some(mut t) = first_comb(self.degree, n) {
            loop {
                out.push((t.clone(), w[flat_index(&t, n)].clone()));
                if !next_comb(&mut t, n) {
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Get the degree of the form
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        self.tensor.manifold()
    }

    /// Get the underlying tensor field
    pub fn tensor(&self) -> &TensorField {
        &self.tensor
    }

    /// Create a coordinate 1-form dx^i
    pub fn coordinate_form(
        manifold: Arc<DifferentiableManifold>,
        chart: &Chart,
        index: usize,
    ) -> Result<Self> {
        let n = manifold.dimension();
        if index >= n {
            return Err(ManifoldError::InvalidIndex(
                format!("Coordinate index {} out of range for dimension {}", index, n)
            ));
        }

        let mut components = zeroed_components(n)?;
        components[index] = Expr::from(1);

        let tensor = TensorField::from_dense_unchecked(manifold, 0, 1, chart, components);
        Ok(Self { tensor, degree: 1 })
    }

    /// Exterior derivative d: Ωᵖ(M) → Ωᵖ⁺¹(M)
    ///
    /// (dω)_I = Σ_{k=0}^{p} (-1)^k ∂_{i_k} ω_{I \ i_k} for I = (i_0 < ... < i_p) (Lee 2013,
    /// ch. 14), computed on the C(n, p+1) increasing tuples, each component simplified and put
    /// through the exact zero test, then written to all its permutations. For p >= n the result
    /// is the zero form of degree p + 1 (no components).
    ///
    /// Precondition: `chart` is the chart the components were written in (they are stored by
    /// chart name; see the module documentation).
    ///
    /// # Errors
    ///
    /// * `DegreeOverflow` when p + 1 overflows.
    /// * `NoComponentsInChart`.
    /// * `InvalidChart` when the chart's dimension is not dim(M), when it repeats a coordinate,
    ///   or when the manifold's registered chart of the same name has other coordinates.
    /// * `NotImplemented` when a component has a derivative that rustmath-symbolic's
    ///   `differentiate` does not implement (a named function, `mod`, gamma, factorial, zeta or
    ///   erf of the coordinate, a `root_sum` whose polynomial contains it).
    /// * `ValidationError` for a component that is not finite.
    /// * `ComponentCountOverflow`, `ComponentAllocationFailed` for the storage.
    pub fn exterior_derivative(&self, chart: &Chart) -> Result<DiffForm> {
        let n = self.manifold().dimension();
        let p = self.degree;
        let k = p.checked_add(1).ok_or(ManifoldError::DegreeOverflow { left: p, right: 1 })?;

        if p >= n {
            // I2: a form of degree > n is zero
            return Ok(DiffForm::new(self.manifold().clone(), k));
        }

        let w = self.tensor.components_ref(chart)?;
        check_coordinate_chart(self.manifold(), chart)?;
        let symbols = chart.coordinate_symbols();
        let count = component_count(n, k)?;
        let mut out = zeroed_components(count)?;

        // p < n, so k <= n and the first tuple exists
        let mut i: Vec<usize> = (0..k).collect();
        let mut rest = vec![0usize; p];
        loop {
            let mut terms: Vec<(Expr, bool)> = Vec::new();
            for kk in 0..k {
                // rest = i without its kk-th entry (still increasing)
                let mut r = 0;
                for (a, &v) in i.iter().enumerate() {
                    if a != kk {
                        rest[r] = v;
                        r += 1;
                    }
                }
                let wj = &w[flat_index(&rest, n)];
                if wj.is_constant() {
                    continue;
                }
                let var = &symbols[i[kk]];
                check_derivative(wj, var)?;
                let t = wj.differentiate(var).simplify();
                if t.is_zero() {
                    continue;
                }
                terms.push((t, kk % 2 == 1));
            }
            if let Some(s) = sum_signed(terms) {
                let v = normalize(s)?;
                scatter(&mut out, &i, n, &v);
            }
            if !next_comb(&mut i, n) {
                break;
            }
        }

        Ok(DiffForm {
            tensor: TensorField::from_dense_unchecked(self.manifold().clone(), 0, k, chart, out),
            degree: k,
        })
    }

    /// Wedge product: ωᵖ ∧ ηᵍ → ω∧η ∈ Ωᵖ⁺ᵍ(M)
    ///
    /// (ω ∧ η)_I = Σ_{A ⊂ I, |A| = p} sgn(A, I \ A) ω_A η_{I \ A} (Lee 2013, ch. 14, the
    /// determinant convention), computed on the increasing tuples I, each component simplified
    /// and put through the exact zero test. For p + q > n the result is the zero form of degree
    /// p + q (no components).
    ///
    /// # Errors
    ///
    /// `DifferentManifolds`, `DegreeOverflow`, `NoComponentsInChart` (self first),
    /// `ComponentCountOverflow`, `ComponentAllocationFailed`.
    pub fn wedge(&self, other: &DiffForm, chart: &Chart) -> Result<DiffForm> {
        if !Arc::ptr_eq(self.manifold(), other.manifold()) {
            return Err(ManifoldError::DifferentManifolds);
        }

        let p = self.degree;
        let q = other.degree;
        let n = self.manifold().dimension();
        let k = p.checked_add(q).ok_or(ManifoldError::DegreeOverflow { left: p, right: q })?;

        if k > n {
            // I2: the wedge product vanishes
            return Ok(DiffForm::new(self.manifold().clone(), k));
        }

        let a = self.tensor.components_ref(chart)?;
        let b = other.tensor.components_ref(chart)?;
        let count = component_count(n, k)?;
        let mut out = zeroed_components(count)?;

        let mut i: Vec<usize> = (0..k).collect();
        let mut ia = vec![0usize; p];
        let mut ib = vec![0usize; q];
        loop {
            let mut terms: Vec<(Expr, bool)> = Vec::new();
            // the p-subsets S of the positions 0..k, lexicographic
            let mut s: Vec<usize> = (0..p).collect();
            loop {
                let (mut x, mut y) = (0, 0);
                // inversions of the shuffle: pairs (a in S, b not in S) with b before a
                let mut inv = 0usize;
                for (pos, &ip) in i.iter().enumerate() {
                    if x < p && s[x] == pos {
                        ia[x] = ip;
                        x += 1;
                    } else {
                        ib[y] = ip;
                        y += 1;
                        inv += p - x;
                    }
                }
                let av = &a[flat_index(&ia, n)];
                let bv = &b[flat_index(&ib, n)];
                if !av.is_zero() && !bv.is_zero() {
                    let t = (av.clone() * bv.clone()).simplify();
                    if !t.is_zero() {
                        terms.push((t, inv % 2 == 1));
                    }
                }
                if !next_comb(&mut s, k) {
                    break;
                }
            }
            if let Some(sum) = sum_signed(terms) {
                let v = normalize(sum)?;
                scatter(&mut out, &i, n, &v);
            }
            if !next_comb(&mut i, n) {
                break;
            }
        }

        Ok(DiffForm {
            tensor: TensorField::from_dense_unchecked(self.manifold().clone(), 0, k, chart, out),
            degree: k,
        })
    }

    /// Interior product (contraction) with a vector field: i_X ω
    ///
    /// (i_X ω)_J = Σ_k X^k ω_{kJ} (Lee 2013, ch. 14, "Interior Multiplication"; SageMath
    /// contracts the first index the same way). For p > n + 1 the result is the zero form of
    /// degree p - 1 (no components); for p = n + 1 it is the zero n-form, stored as zeros in
    /// `chart`.
    ///
    /// The vector field's components are not validated (`VectorField::from_components` checks
    /// the length only). A non-finite X^k gives `Err(ValidationError)` when its term is kept.
    ///
    /// # Errors
    ///
    /// `DifferentManifolds`; `InvalidOperation` for a 0-form; `NoComponentsInChart` for the
    /// vector field or the form; `ValidationError`; the storage errors.
    pub fn interior_product(&self, vector: &VectorField, chart: &Chart) -> Result<DiffForm> {
        if !Arc::ptr_eq(self.manifold(), vector.manifold()) {
            return Err(ManifoldError::DifferentManifolds);
        }

        if self.degree == 0 {
            return Err(ManifoldError::InvalidOperation(
                "Cannot contract 0-form".to_string()
            ));
        }

        let n = self.manifold().dimension();
        let p = self.degree;
        let k = p - 1;
        if k > n {
            return Ok(DiffForm::new(self.manifold().clone(), k));
        }
        if p > n {
            // p = n + 1: the zero n-form, with components
            let out = zeroed_components(component_count(n, k)?)?;
            return Ok(DiffForm {
                tensor: TensorField::from_dense_unchecked(self.manifold().clone(), 0, k, chart, out),
                degree: k,
            });
        }

        let x = vector.components(chart)?;
        let w = self.tensor.components_ref(chart)?;
        let count = component_count(n, k)?;
        let mut out = zeroed_components(count)?;

        // k < n here, so the first tuple exists
        let mut j: Vec<usize> = (0..k).collect();
        let mut full = vec![0usize; p];
        loop {
            let mut terms: Vec<(Expr, bool)> = Vec::new();
            for (kk, xk) in x.iter().enumerate() {
                if j.contains(&kk) {
                    continue;
                }
                full[0] = kk;
                full[1..].copy_from_slice(&j);
                let wv = &w[flat_index(&full, n)];
                if wv.is_zero() || xk.is_zero() {
                    continue;
                }
                let t = (xk.clone() * wv.clone()).simplify();
                if !t.is_zero() {
                    terms.push((t, false));
                }
            }
            if let Some(s) = sum_signed(terms) {
                let v = normalize(s)?;
                scatter(&mut out, &j, n, &v);
            }
            if !next_comb(&mut j, n) {
                break;
            }
        }

        Ok(DiffForm {
            tensor: TensorField::from_dense_unchecked(self.manifold().clone(), 0, k, chart, out),
            degree: k,
        })
    }

    /// Lie derivative along a vector field: ℒ_X ω
    ///
    /// Cartan's formula ℒ_X ω = i_X(dω) + d(i_X ω) (Lee 2013, ch. 14). For a 0-form this is
    /// i_X(dω) = X(f); for p = n it is d(i_X ω), since dω has degree n + 1; for p > n the
    /// result is the zero form of degree p. On a 0-dimensional manifold a 0-form gives the
    /// 0-form 0.
    ///
    /// Precondition and errors: those of [`DiffForm::exterior_derivative`] and
    /// [`DiffForm::interior_product`]; `chart` must be the chart the components were written in.
    pub fn lie_derivative(&self, vector: &VectorField, chart: &Chart) -> Result<DiffForm> {
        if !Arc::ptr_eq(self.manifold(), vector.manifold()) {
            return Err(ManifoldError::DifferentManifolds);
        }
        let n = self.manifold().dimension();
        let p = self.degree;
        if p > n {
            return Ok(DiffForm::new(self.manifold().clone(), p));
        }
        self.tensor.components_ref(chart)?;
        if n == 0 {
            return Ok(DiffForm {
                tensor: TensorField::from_dense_unchecked(self.manifold().clone(), 0, 0, chart, vec![Expr::from(0)]),
                degree: 0,
            });
        }

        if p == 0 {
            return self.exterior_derivative(chart)?.interior_product(vector, chart);
        }
        let d_ix_omega = self.interior_product(vector, chart)?.exterior_derivative(chart)?;
        if p == n {
            return Ok(d_ix_omega);
        }
        let ix_d_omega = self.exterior_derivative(chart)?.interior_product(vector, chart)?;
        ix_d_omega + d_ix_omega
    }

    /// factor * ω in every chart: each independent component is multiplied, normalized and
    /// scattered. A literal 0 gives the zero form in the same charts, a literal 1 a clone.
    ///
    /// # Errors
    ///
    /// `ValidationError` for a factor that is not finite (checked first); otherwise only the
    /// allocation errors.
    pub fn scale(&self, factor: &Expr) -> Result<DiffForm> {
        if !is_finite_expr(factor) {
            return Err(ManifoldError::ValidationError(format!(
                "the factor {} is not finite",
                factor
            )));
        }
        let is_one = match factor {
            Expr::Integer(c) => c.is_one(),
            Expr::Rational(r) => r.numerator() == r.denominator(),
            _ => false,
        };
        if is_one {
            return Ok(self.clone());
        }
        let n = self.manifold().dimension();
        let p = self.degree;
        let mut map: HashMap<ChartId, Vec<Expr>> = HashMap::new();
        for (id, w) in self.tensor.chart_components() {
            let mut out = zeroed_components(w.len())?;
            if !factor.is_zero() {
                if let Some(mut t) = first_comb(p, n) {
                    loop {
                        let v = &w[flat_index(&t, n)];
                        if !v.is_zero() {
                            let s = normalize(factor.clone() * v.clone())?;
                            scatter(&mut out, &t, n, &s);
                        }
                        if !next_comb(&mut t, n) {
                            break;
                        }
                    }
                }
            }
            map.insert(id.clone(), out);
        }
        Ok(DiffForm {
            tensor: TensorField::from_chart_components(self.manifold().clone(), 0, p, map),
            degree: p,
        })
    }

    /// Decides whether ω vanishes identically in `chart`. Degree > n: `Zero` (I2). Otherwise
    /// the exact zero test over the independent components: `NonZero` as soon as one is
    /// `NonZero`, else `Unresolved` if one is `Unresolved`, else `Zero`.
    ///
    /// # Errors
    ///
    /// `NoComponentsInChart`.
    pub fn vanishes(&self, chart: &Chart) -> Result<ZeroTest> {
        let n = self.manifold().dimension();
        if self.degree > n {
            return Ok(ZeroTest::Zero);
        }
        let w = self.tensor.components_ref(chart)?;
        let mut answer = ZeroTest::Zero;
        if let Some(mut t) = first_comb(self.degree, n) {
            loop {
                let v = &w[flat_index(&t, n)];
                if !v.is_zero() {
                    match zero_test(v) {
                        ZeroTest::Zero => {}
                        ZeroTest::NonZero => return Ok(ZeroTest::NonZero),
                        ZeroTest::Unresolved => answer = ZeroTest::Unresolved,
                    }
                }
                if !next_comb(&mut t, n) {
                    break;
                }
            }
        }
        Ok(answer)
    }

    /// Check if this is the zero form, structurally: every stored component is the literal 0
    /// (a form without components counts as zero, the convention of [`DiffForm::new`] and
    /// [`DiffForm::zero`]). `false` does not prove the form nonzero; use
    /// [`DiffForm::vanishes`] for a decision.
    pub fn is_zero(&self) -> bool {
        self.tensor.is_zero()
    }

    /// Whether dω vanishes: [`DiffForm::vanishes`] of [`DiffForm::exterior_derivative`].
    ///
    /// Precondition: `chart` is the chart the components were written in. The errors of d
    /// propagate (`NotImplemented` from the derivative guard, `InvalidChart`,
    /// `NoComponentsInChart`); a fabricated derivative never reaches the zero test.
    pub fn closedness(&self, chart: &Chart) -> Result<ZeroTest> {
        self.exterior_derivative(chart)?.vanishes(chart)
    }

    /// Check if this form is closed (i.e., d(form) = 0)
    ///
    /// A differential form ω is closed if its exterior derivative is zero: dω = 0.
    /// This is a key concept in de Rham cohomology.
    ///
    /// # Arguments
    ///
    /// * `chart` - The chart in which to compute the exterior derivative; it must be the chart
    ///   the components were written in
    ///
    /// # Returns
    ///
    /// `Ok(true)` when dω is decided zero, `Ok(false)` when it is decided nonzero.
    ///
    /// # Errors
    ///
    /// `Undecided` when the exact zero test cannot decide a component of dω, and the errors
    /// of [`DiffForm::closedness`].
    pub fn is_closed(&self, chart: &Chart) -> Result<bool> {
        match self.closedness(chart)? {
            ZeroTest::Zero => Ok(true),
            ZeroTest::NonZero => Ok(false),
            ZeroTest::Unresolved => Err(ManifoldError::Undecided(
                "d(ω) has components the zero test cannot decide".to_string(),
            )),
        }
    }

    /// Create a zero p-form
    pub fn zero(manifold: Arc<DifferentiableManifold>, degree: usize) -> Self {
        Self::new(manifold, degree)
    }
}

// PartialEq implementation
impl PartialEq for DiffForm {
    fn eq(&self, other: &Self) -> bool {
        // Two differential forms are equal if they have the same degree and tensor components
        // Note: This is a structural equality, not mathematical equality
        self.degree == other.degree && self.tensor == other.tensor
    }
}

// Arithmetic operations

/// Componentwise sum in every chart where both summands have components. The sums are
/// computed on the independent components (simplified and zero-tested) and scattered.
///
/// # Errors
///
/// `DifferentManifolds`; `DimensionMismatch` for different degrees; `NoComponentsInChart` when
/// no chart is shared. A form from [`DiffForm::new`] or [`DiffForm::zero`] of degree <= n has no
/// components, so adding it errs. For degree > n the sum is [`DiffForm::new`] (I2).
impl Add for DiffForm {
    type Output = Result<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        if !Arc::ptr_eq(self.manifold(), rhs.manifold()) {
            return Err(ManifoldError::DifferentManifolds);
        }
        if self.degree != rhs.degree {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.degree,
                actual: rhs.degree,
            });
        }
        let n = self.manifold().dimension();
        let p = self.degree;
        if p > n {
            return Ok(DiffForm::new(self.manifold().clone(), p));
        }
        let rhs_map = rhs.tensor.chart_components();
        let mut map: HashMap<ChartId, Vec<Expr>> = HashMap::new();
        for (id, a) in self.tensor.chart_components() {
            let Some(b) = rhs_map.get(id) else {
                continue;
            };
            let mut out = zeroed_components(a.len())?;
            if let Some(mut t) = first_comb(p, n) {
                loop {
                    let f = flat_index(&t, n);
                    let (x, y) = (&a[f], &b[f]);
                    let v = if x.is_zero() {
                        y.clone()
                    } else if y.is_zero() {
                        x.clone()
                    } else {
                        normalize(x.clone() + y.clone())?
                    };
                    scatter(&mut out, &t, n, &v);
                    if !next_comb(&mut t, n) {
                        break;
                    }
                }
            }
            map.insert(id.clone(), out);
        }
        if map.is_empty() {
            return Err(ManifoldError::NoComponentsInChart);
        }
        Ok(DiffForm {
            tensor: TensorField::from_chart_components(self.manifold().clone(), 0, p, map),
            degree: p,
        })
    }
}

/// Negates every component in every chart, in place: for each increasing tuple with value v,
/// the even permutations get neg(v) and the odd ones the old v, which is equal in value to
/// neg(neg(v)) (invariant I1 holds up to value).
impl Neg for DiffForm {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut out = self;
        let n = out.manifold().dimension();
        let p = out.degree;
        if p > n {
            return out;
        }
        for w in out.tensor.chart_components_mut().values_mut() {
            if let Some(mut t) = first_comb(p, n) {
                loop {
                    let v = w[flat_index(&t, n)].clone();
                    if !v.is_zero() {
                        let nv = neg(&v);
                        for_each_perm(&t, |perm, even| {
                            w[flat_index(perm, n)] = if even { nv.clone() } else { v.clone() };
                        });
                    }
                    if !next_comb(&mut t, n) {
                        break;
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::EuclideanSpace;
    use rustmath_symbolic::expression::BinaryOp;
    use rustmath_rationals::Rational;
    use rustmath_symbolic::{parse, Symbol};
    use std::time::{Duration, Instant};

    // Reference values: PARI/GP 2.15, script
    // fu2/design/MF1-manifolds/refs.gp (run `gp -q refs.gp`), rev2/refs2.gp and rev3/refs4.gp
    // (`gp -q refs4.gp < /dev/null`); the gp lines are quoted next to each test. SageMath values
    // are quoted from the documentation of sage.manifolds.differentiable.diff_form.

    fn s(name: &str) -> Expr {
        Expr::symbol(name)
    }

    fn int(n: i64) -> Expr {
        Expr::from(n)
    }

    fn p(src: &str) -> Expr {
        parse(src).unwrap()
    }

    /// The 3-manifold R3 with the registered chart c = (x, y, z).
    fn r3c() -> (Arc<DifferentiableManifold>, Chart) {
        let mut m = DifferentiableManifold::new("R3", 3);
        let c = Chart::new("c", 3, vec!["x", "y", "z"]).unwrap();
        m.add_chart(c.clone()).unwrap();
        (Arc::new(m), c)
    }

    /// R^2 with its default chart (x, y).
    fn r2() -> (Arc<DifferentiableManifold>, Chart) {
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let c = m.default_chart().unwrap().clone();
        (m, c)
    }

    /// An n-manifold with the registered chart c of the given coordinate names.
    fn manifold(names: &[String]) -> (Arc<DifferentiableManifold>, Chart) {
        let mut m = DifferentiableManifold::new("M", names.len());
        let c = Chart::new("c", names.len(), names.to_vec()).unwrap();
        m.add_chart(c.clone()).unwrap();
        (Arc::new(m), c)
    }

    fn assert_value(ours: &Expr, reference: &Expr) {
        assert_eq!(
            zero_test(&(ours.clone() - reference.clone())),
            ZeroTest::Zero,
            "{} is not {}",
            ours,
            reference
        );
    }

    fn dense(form: &DiffForm, c: &Chart) -> Vec<Expr> {
        form.tensor().components(c).unwrap()
    }

    fn values(form: &DiffForm, c: &Chart) -> Vec<Expr> {
        form.independent_components(c).unwrap().into_iter().map(|(_, v)| v).collect()
    }

    fn literal_zero(e: &Expr) -> bool {
        matches!(e, Expr::Integer(n) if n.is_zero())
    }

    fn one_form(m: &Arc<DifferentiableManifold>, c: &Chart, comps: Vec<Expr>) -> DiffForm {
        DiffForm::from_components(m.clone(), c, 1, comps).unwrap()
    }

    /// A = x*y*z dx - z*x dy + y*z dz on R3 (the Sage example).
    fn form_a(m: &Arc<DifferentiableManifold>, c: &Chart) -> DiffForm {
        one_form(m, c, vec![p("x*y*z"), p("-z*x"), p("y*z")])
    }

    fn rs(f: Expr, g: Expr) -> Expr {
        Expr::Function("root_sum".into(), vec![Arc::new(f), Arc::new(g)])
    }

    #[test]
    fn test_diff_form_creation() {
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let omega = DiffForm::new(m.clone(), 1);
        assert_eq!(omega.degree(), 1);
    }

    #[test]
    fn test_coordinate_form() {
        let m = Arc::new(EuclideanSpace::new(3).manifold().clone());
        let chart = m.default_chart().unwrap();

        let dx = DiffForm::coordinate_form(m.clone(), chart, 0).unwrap();
        assert_eq!(dx.degree(), 1);
    }

    #[test]
    fn test_zero_form() {
        let m = Arc::new(EuclideanSpace::new(2).manifold().clone());
        let zero = DiffForm::zero(m, 1);
        assert!(zero.is_zero());
    }

    #[test]
    fn test_exterior_derivative_degree() {
        let m = Arc::new(EuclideanSpace::new(3).manifold().clone());
        let chart = m.default_chart().unwrap();
        // A 1-form needs components populated in the chart (n = 3) for the
        // exterior derivative to be computable.
        let omega = DiffForm::from_components(m.clone(), chart, 1, vec![Expr::from(0); 3]).unwrap();

        let d_omega = omega.exterior_derivative(chart).unwrap();
        assert_eq!(d_omega.degree(), 2);
    }

    #[test]
    fn test_wedge_product_degree() {
        let m = Arc::new(EuclideanSpace::new(3).manifold().clone());
        let chart = m.default_chart().unwrap();

        let omega1 = DiffForm::from_components(m.clone(), chart, 1, vec![Expr::from(0); 3]).unwrap();
        let omega2 = DiffForm::from_components(m.clone(), chart, 1, vec![Expr::from(0); 3]).unwrap();

        let wedge = omega1.wedge(&omega2, chart).unwrap();
        assert_eq!(wedge.degree(), 2);
    }

    #[test]
    fn test_d_one_form_r3() {
        // gp: dA = [D(A[2],1) - D(A[1],2), ...] prints [-z*x - z, -y*x, x + z]
        // Sage: dA = -(x + 1)*z dx∧dy - x*y dx∧dz + (x + z) dy∧dz
        let (m, c) = r3c();
        let da = form_a(&m, &c).exterior_derivative(&c).unwrap();
        assert_eq!(da.degree(), 2);
        let v = values(&da, &c);
        assert_value(&v[0], &p("-z*x - z"));
        assert_value(&v[1], &p("-y*x"));
        assert_value(&v[2], &p("x + z"));
        let w = dense(&da, &c);
        assert_eq!(w[3], (-w[1].clone()).simplify());
        assert!(literal_zero(&w[0]));
        assert!(literal_zero(&w[4]));
    }

    #[test]
    fn test_dd_is_literal_zero() {
        // gp: ddA_{xyz} = 0
        let (m, c) = r3c();
        let dda = form_a(&m, &c).exterior_derivative(&c).unwrap().exterior_derivative(&c).unwrap();
        assert_eq!(dda.degree(), 3);
        assert!(dense(&dda, &c).iter().all(literal_zero));
        let f = DiffForm::from_components(m.clone(), &c, 0, vec![p("x^2*y*z^3 + x*y")]).unwrap();
        let ddf = f.exterior_derivative(&c).unwrap().exterior_derivative(&c).unwrap();
        assert!(dense(&ddf, &c).iter().all(literal_zero));
    }

    #[test]
    fn test_d_of_coordinate_form_is_zero() {
        // at HEAD the first n slots were trees of n zeros
        let names: Vec<String> = (0..300).map(|i| format!("x{}", i)).collect();
        let (m, c) = manifold(&names);
        let dx0 = DiffForm::coordinate_form(m.clone(), &c, 0).unwrap();
        let d = dx0.exterior_derivative(&c).unwrap();
        let w = dense(&d, &c);
        assert_eq!(w.len(), 90000);
        assert!(w.iter().all(literal_zero));

        let names: Vec<String> = (0..30).map(|i| format!("x{}", i)).collect();
        let (m, c) = manifold(&names);
        let dx0 = DiffForm::coordinate_form(m.clone(), &c, 0).unwrap();
        let dd = dx0.exterior_derivative(&c).unwrap().exterior_derivative(&c).unwrap();
        let w = dense(&dd, &c);
        assert_eq!(w.len(), 27000);
        assert!(w.iter().all(literal_zero));
    }

    #[test]
    fn test_d_trig_sage() {
        // Sage: dB = cos(x) dx∧dy + sin(z) dx∧dz - sin(y) dy∧dz for B = (cos z, sin x, cos y)
        let (m, c) = r3c();
        let b = one_form(&m, &c, vec![s("z").cos(), s("x").sin(), s("y").cos()]);
        let v = values(&b.exterior_derivative(&c).unwrap(), &c);
        assert_value(&v[0], &s("x").cos());
        assert_value(&v[1], &s("z").sin());
        assert_value(&v[2], &(-s("y").sin()));
    }

    #[test]
    fn test_d_two_dim() {
        // gp: deriv(x^2,x) - deriv(x*y,y) = x; deriv(x^2,x) - deriv(-y^2,y) = 2*x + 2*y
        // Sage: da = (2*x + 2*y) dx∧dy; HEAD gave [(y + x), (2 * x), 0, 0]
        let (m, c) = r2();
        let w = dense(&one_form(&m, &c, vec![p("x*y"), p("x^2")]).exterior_derivative(&c).unwrap(), &c);
        assert!(literal_zero(&w[0]) && literal_zero(&w[3]));
        assert_value(&w[1], &s("x"));
        assert_value(&w[2], &(-s("x")));
        let v = values(&one_form(&m, &c, vec![p("-y^2"), p("x^2")]).exterior_derivative(&c).unwrap(), &c);
        assert_value(&v[0], &p("2*x + 2*y"));
    }

    #[test]
    fn test_d_zero_form_gradient() {
        // gp: [deriv(x*y^2,x), deriv(x*y^2,y)] = [y^2, 2*y*x]
        let (m, c) = r2();
        let f = DiffForm::from_components(m.clone(), &c, 0, vec![p("x*y^2")]).unwrap();
        let w = dense(&f.exterior_derivative(&c).unwrap(), &c);
        assert_value(&w[0], &p("y^2"));
        assert_value(&w[1], &p("2*x*y"));
    }

    #[test]
    fn test_wedge_sage_r3() {
        // Sage: A∧B = (x*y*z*sin(x) + x*z*cos(z)) dx∧dy + (x*y*z*cos(y) - y*z*cos(z)) dx∧dz
        //       - (x*cos(y) + y*sin(x))*z dy∧dz
        let (m, c) = r3c();
        let b = one_form(&m, &c, vec![s("z").cos(), s("x").sin(), s("y").cos()]);
        let v = values(&form_a(&m, &c).wedge(&b, &c).unwrap(), &c);
        assert_value(&v[0], &p("x*y*z*sin(x) + x*z*cos(z)"));
        assert_value(&v[1], &p("x*y*z*cos(y) - y*z*cos(z)"));
        assert_value(&v[2], &p("-(x*cos(y) + y*sin(x))*z"));
    }

    #[test]
    fn test_wedge_one_two() {
        // gp: a[1]*b2[3] - a[2]*b2[2] + a[3]*b2[1] = -x^2 + (-z - 1)*x + (z*y^3 + (2*z^2 - z))
        // Sage: a∧b = (-x^2 + (y^3 - x - 1)*z + 2*z^2 - x) dx∧dy∧dz
        let (m, c) = r3c();
        let a = one_form(&m, &c, vec![int(2), p("1 + x"), p("y*z")]);
        let b = DiffForm::from_independent_components(m.clone(), &c, 2, vec![p("y^2"), p("z + x"), p("z^2")]).unwrap();
        let ab = a.wedge(&b, &c).unwrap();
        let reference = p("-x^2 + (-z - 1)*x + (z*y^3 + (2*z^2 - z))");
        assert_value(&values(&ab, &c)[0], &reference);
        let ba = b.wedge(&a, &c).unwrap();
        assert_value(&values(&ba, &c)[0], &reference);
    }

    #[test]
    fn test_wedge_two_dim() {
        // gp: a2[1]*bb[2] - a2[2]*bb[1] = -x^3 - y^2*x + y^2; Sage: (-x^3 - (x - 1)*y^2) dx∧dy
        let (m, c) = r2();
        let a = one_form(&m, &c, vec![s("y"), s("x")]);
        let b = one_form(&m, &c, vec![p("x^2 + y^2"), s("y")]);
        assert_value(&values(&a.wedge(&b, &c).unwrap(), &c)[0], &p("-x^3 - y^2*x + y^2"));
    }

    #[test]
    fn test_wedge_antisymmetry() {
        let (m, c) = r2();
        let dx = DiffForm::coordinate_form(m.clone(), &c, 0).unwrap();
        let dy = DiffForm::coordinate_form(m.clone(), &c, 1).unwrap();
        assert!(dense(&dx.wedge(&dx, &c).unwrap(), &c).iter().all(literal_zero));
        assert_eq!(dense(&dx.wedge(&dy, &c).unwrap(), &c), vec![int(0), int(1), int(-1), int(0)]);
        assert_eq!(dense(&dy.wedge(&dx, &c).unwrap(), &c), vec![int(0), int(-1), int(1), int(0)]);
        let (m3, c3) = r3c();
        let a = form_a(&m3, &c3);
        assert!(dense(&a.wedge(&a, &c3).unwrap(), &c3).iter().all(literal_zero));
    }

    /// ω = Σ dq_i∧dp_i on R^{2k}, coordinates (q_1..q_k, p_1..p_k).
    fn symplectic_omega(k: usize) -> (Arc<DifferentiableManifold>, Chart, DiffForm) {
        let names: Vec<String> = (1..=k).map(|i| format!("q{}", i)).chain((1..=k).map(|i| format!("p{}", i))).collect();
        let (m, c) = manifold(&names);
        let n = 2 * k;
        let mut vals = Vec::new();
        let mut t = first_comb(2, n).unwrap();
        loop {
            vals.push(if t[1] == t[0] + k { int(1) } else { int(0) });
            if !next_comb(&mut t, n) {
                break;
            }
        }
        let w = DiffForm::from_independent_components(m.clone(), &c, 2, vals).unwrap();
        (m, c, w)
    }

    #[test]
    fn test_wedge_powers() {
        // gp: k! * (-1)^(k*(k-1)/2) prints 1, -2, -6 for k = 1, 2, 3
        for (k, top) in [(1usize, 1i64), (2, -2), (3, -6)] {
            let (_m, c, w) = symplectic_omega(k);
            let mut power = w.clone();
            for _ in 1..k {
                power = power.wedge(&w, &c).unwrap();
            }
            let v = values(&power, &c);
            assert_eq!(v.len(), 1);
            assert_value(&v[0], &int(top));
        }
    }

    #[test]
    fn test_leibniz() {
        // Lee 2013, ch. 14: d(A∧P) = dA∧P - A∧dP for the 1-form A
        let (m, c) = r3c();
        let a = form_a(&m, &c);
        let pf = one_form(&m, &c, vec![p("y^2"), p("z + x"), p("x*y")]);
        let lhs = a.wedge(&pf, &c).unwrap().exterior_derivative(&c).unwrap();
        let rhs1 = a.exterior_derivative(&c).unwrap().wedge(&pf, &c).unwrap();
        let rhs2 = a.wedge(&pf.exterior_derivative(&c).unwrap(), &c).unwrap();
        let rhs = (rhs1 + (-rhs2)).unwrap();
        assert_value(&values(&lhs, &c)[0], &values(&rhs, &c)[0]);
        assert_eq!((lhs.clone() + (-rhs)).unwrap().vanishes(&c).unwrap(), ZeroTest::Zero);
    }

    #[test]
    fn test_d_budget() {
        // review 1 B2: 26.8 s on the first prototype; every zero test is budgeted
        let (m, c) = r2();
        let x = s("x");
        let sum = (0..50).map(|k| x.clone().pow(int(k))).fold(int(0), |a, b| a + b);
        let big = sum.pow(int(64));
        let t = Instant::now();
        let w = one_form(&m, &c, vec![int(0), big.clone()]);
        let d = w.exterior_derivative(&c).unwrap();
        assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
        let slot = dense(&d, &c)[1].clone();
        assert!(!slot.is_zero());
        assert_eq!(slot, big.differentiate(&Symbol::new("x")).simplify());
        assert_eq!(w.closedness(&c).unwrap(), ZeroTest::Unresolved);
        assert!(matches!(w.is_closed(&c), Err(ManifoldError::Undecided(_))));
    }

    #[test]
    fn test_d_reserved_constants() {
        // I is opaque, pi alone is transcendental, pi and e together are not decided
        let (m, c) = r2();
        let x = s("x");
        let (i, pi, e) = (s("I"), s("pi"), s("e"));
        let cases = vec![
            (i.clone().pow(int(2)) + int(1), ZeroTest::Unresolved),
            (i.clone(), ZeroTest::Unresolved),
            (pi.clone(), ZeroTest::NonZero),
            (pi.clone() - e.clone(), ZeroTest::Unresolved),
        ];
        for (k, closed) in cases {
            let w = one_form(&m, &c, vec![int(0), k.clone() * x.clone()]);
            let slot = dense(&w.exterior_derivative(&c).unwrap(), &c)[1].clone();
            assert!(!slot.is_zero(), "{}", slot);
            assert_value(&slot, &k);
            assert_eq!(w.closedness(&c).unwrap(), closed, "{}", k);
        }
    }

    #[test]
    fn test_interior_product_long_sum() {
        // sums are balanced: a left-deep sum of 20000 terms overflows a 2 MB stack
        std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(|| {
                let n = 20000;
                let names: Vec<String> = (0..n).map(|i| format!("x{}", i)).collect();
                let (m, c) = manifold(&names);
                let w: Vec<Expr> = (0..n).map(|i| s(&names[(i + 1) % n])).collect();
                let x: Vec<Expr> = names.iter().map(|nm| s(nm)).collect();
                let form = one_form(&m, &c, w);
                let field = VectorField::from_components(m.clone(), &c, x).unwrap();
                let r = form.interior_product(&field, &c).unwrap();
                assert_eq!(r.degree(), 0);
                let v = dense(&r, &c);
                assert_eq!(v.len(), 1);
                assert!(!v[0].is_zero());
                assert_eq!(zero_test(&v[0]), ZeroTest::Unresolved);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// The error of d on f dy (R^2), and that closedness and is_closed return the same.
    fn refused(f: Expr, u: &Expr) {
        let (m, c) = r2();
        let w = one_form(&m, &c, vec![int(0), f.clone()]);
        let err = w.exterior_derivative(&c).unwrap_err();
        match &err {
            ManifoldError::NotImplemented(msg) => {
                assert!(msg.contains("the derivative of"), "{}", msg);
                assert!(msg.contains(&format!("{}", u)), "{} does not name {}", msg, u);
            }
            other => panic!("d({} dy): {:?}", f, other),
        }
        assert_eq!(w.closedness(&c).unwrap_err(), err);
        assert_eq!(w.is_closed(&c).unwrap_err(), err);
    }

    #[test]
    fn test_d_refuses_unsupported_derivatives() {
        // The true derivatives are nonzero: f'(x), Γ(x)ψ(x), ζ'(x), 1 off 3Z, 2/sqrt(π)
        // exp(-x^2) (gp: 500/443 = 1.12867, 2/sqrt(Pi) = 1.12838), ...
        let (x, y, t) = (s("x"), s("y"), s("t"));
        let f = |a: Expr| Expr::Function("f".into(), vec![Arc::new(a)]);
        let modulo = |a: Expr| Expr::Binary(BinaryOp::Mod, Arc::new(a), Arc::new(int(3)));
        let t2 = t.clone().pow(int(2));
        let cases: Vec<(Expr, Expr)> = vec![
            (f(x.clone()), f(x.clone())),
            (Expr::bessel_j(int(0), x.clone()), Expr::bessel_j(int(0), x.clone())),
            (x.clone().gamma(), x.clone().gamma()),
            (x.clone().zeta(), x.clone().zeta()),
            (x.clone().factorial(), x.clone().factorial()),
            (modulo(x.clone()), modulo(x.clone())),
            (x.clone().erf(), x.clone().erf()),
            (f(x.clone().pow(int(2)) + y.clone()).sin(), f(x.clone().pow(int(2)) + y.clone())),
            (y.clone() * (int(1) + x.clone()).gamma(), (int(1) + x.clone()).gamma()),
            (rs(t2.clone() - int(2), t.clone() * f(x.clone())), f(x.clone())),
            (p("gamma(x)"), x.clone().gamma()),
            (p("f(x)"), f(x.clone())),
        ];
        for (form, u) in cases {
            refused(form, &u);
        }
        // the 0-form gamma(x)
        let (m, c) = r2();
        let g = DiffForm::from_components(m.clone(), &c, 0, vec![x.clone().gamma()]).unwrap();
        assert!(matches!(g.exterior_derivative(&c), Err(ManifoldError::NotImplemented(_))));

        // accepted: no unsupported node contains the variable it is differentiated in
        let accepted_zero: Vec<Vec<Expr>> = vec![
            vec![int(0), f(y.clone())],
            vec![f(x.clone()), int(0)],
            vec![int(0), int(3).gamma()],
            vec![int(0), y.clone().gamma()],
            vec![int(0), modulo(y.clone())],
            vec![int(0), x.clone().sign()],
        ];
        for comps in accepted_zero {
            let w = one_form(&m, &c, comps.clone());
            let d = w.exterior_derivative(&c).unwrap();
            assert!(literal_zero(&dense(&d, &c)[1]), "{:?}", comps);
            assert_eq!(w.closedness(&c).unwrap(), ZeroTest::Zero);
        }
        // accepted with a value: d/dx |x| = sign(x) off 0, d/dx x^y = y x^(y-1) for x > 0,
        // Σ_{c^2 = 2} c = 0 and Σ_{c^2 = y} c = 0 (gp: polsym(t^2 - 2, 1)[2] = 0; the zero test
        // cannot see it, so Unresolved is honest)
        let accepted: Vec<(Expr, Expr)> = vec![
            (x.clone().abs(), x.clone().sign()),
            (x.clone().pow(y.clone()), x.clone().pow(y.clone()) * (y.clone() / x.clone())),
            (rs(t2.clone() - int(2), t.clone() * x.clone()), rs(t2.clone() - int(2), t.clone())),
            (rs(t2.clone() - y.clone(), t.clone() * x.clone()), rs(t2.clone() - y.clone(), t.clone())),
        ];
        for (comp, slot) in accepted {
            let w = one_form(&m, &c, vec![int(0), comp.clone()]);
            let d = w.exterior_derivative(&c).unwrap();
            assert_eq!(dense(&d, &c)[1], slot, "d({} dy)", comp);
            assert_eq!(w.closedness(&c).unwrap(), ZeroTest::Unresolved);
        }

        // root_sum with the variable in its polynomial (review 3 B1). gp (refs4.gp): the true
        // (dω)_xy are 2, 2, 4x, 1, 2 cos(2x), 0 (refused anyway), 2Γ(2x)ψ(2x) (0.8456 at x = 1)
        // and 4.
        let s2 = s("s").pow(int(2));
        let r_x = rs(t2.clone() - x.clone(), t2.clone());
        let parsed = p("root_sum(t^2 - x, t^2)");
        let inner = rs(s2.clone() - x.clone(), s2.clone());
        let cases: Vec<(Expr, Expr)> = vec![
            (r_x.clone(), r_x.clone()),
            (parsed.clone(), parsed.clone()),
            (rs(t2.clone() - x.clone(), t.clone().pow(int(4))), rs(t2.clone() - x.clone(), t.clone().pow(int(4)))),
            (rs(t.clone() - x.clone(), t.clone()), rs(t.clone() - x.clone(), t.clone())),
            (r_x.clone().sin(), r_x.clone()),
            (rs(x.clone().pow(int(2)) - int(2), x.clone() * y.clone()), rs(x.clone().pow(int(2)) - int(2), x.clone() * y.clone())),
            (r_x.clone().gamma(), r_x.clone().gamma()),
            (rs(t2.clone() - int(2), inner.clone()), inner.clone()),
        ];
        for (form, u) in cases {
            refused(form, &u);
        }
        // root_sum(t^2 - x, t) dx: (dω)_xy = -∂_y ω_x, and the polynomial has no y
        let w = one_form(&m, &c, vec![rs(t2.clone() - x.clone(), t.clone()), int(0)]);
        let slot = dense(&w.exterior_derivative(&c).unwrap(), &c)[1].clone();
        assert!(!slot.is_zero());
        assert_eq!(slot, -rs(t2.clone() - x.clone(), int(0)));
        assert_eq!(w.closedness(&c).unwrap(), ZeroTest::Unresolved);
    }

    #[test]
    fn test_degree_overflow() {
        let (m, c) = r2();
        assert_eq!(
            DiffForm::new(m.clone(), usize::MAX).exterior_derivative(&c).unwrap_err(),
            ManifoldError::DegreeOverflow { left: usize::MAX, right: 1 }
        );
        assert_eq!(
            DiffForm::new(m.clone(), usize::MAX).wedge(&DiffForm::new(m.clone(), 1), &c).unwrap_err(),
            ManifoldError::DegreeOverflow { left: usize::MAX, right: 1 }
        );
    }

    fn invalid_chart_message(r: Result<impl std::fmt::Debug>) -> String {
        match r {
            Err(ManifoldError::InvalidChart(m)) => m,
            other => panic!("expected InvalidChart, got {:?}", other),
        }
    }

    #[test]
    fn test_d_chart_dimension() {
        // components are keyed by the chart name, so the lookup finds them for any chart named c
        let (m, _) = r2();
        let c2 = Chart::new("c", 2, vec!["x", "y"]).unwrap();
        let w = one_form(&m, &c2, vec![p("x*y"), p("x^2")]);
        let c0 = Chart::new("c", 0, Vec::<String>::new()).unwrap();
        let msg = invalid_chart_message(w.exterior_derivative(&c0));
        assert!(msg.contains("the chart 'c' has dimension 0, but the manifold has dimension 2"), "{}", msg);
        let c1 = Chart::new("c", 1, vec!["x"]).unwrap();
        let msg = invalid_chart_message(w.exterior_derivative(&c1));
        assert!(msg.contains("has dimension 1, but the manifold has dimension 2"), "{}", msg);
        let d = w.exterior_derivative(&c2).unwrap();
        assert_value(&values(&d, &c2)[0], &s("x"));
    }

    #[test]
    fn test_d_chart_identity() {
        // review 3: at HEAD d differentiated x dy in u and v
        let mut m = DifferentiableManifold::new("R2", 2);
        let c = Chart::new("c", 2, vec!["x", "y"]).unwrap();
        m.add_chart(c.clone()).unwrap();
        let m = Arc::new(m);
        let w = one_form(&m, &c, vec![int(0), s("x")]);
        let u = Chart::new("c", 2, vec!["u", "v"]).unwrap();
        let expected = "has coordinates (u, v), but the manifold's chart of that name has coordinates (x, y)";
        assert!(invalid_chart_message(w.exterior_derivative(&u)).contains(expected));
        assert!(invalid_chart_message(w.closedness(&u)).contains(expected));
        assert!(invalid_chart_message(w.is_closed(&u)).contains(expected));
        assert!(!w.is_closed(&c).unwrap());
        // review 4: a chart that repeats a coordinate is not a chart
        let r = Chart::new("r", 2, vec!["x", "x"]).unwrap();
        let wr = one_form(&m, &r, vec![s("x"), int(0)]);
        let msg = invalid_chart_message(wr.exterior_derivative(&r));
        assert!(msg.contains("the chart 'r' repeats the coordinate x"), "{}", msg);
    }

    #[test]
    fn test_zero_by_dimension() {
        let (m, c) = r2();
        let top = DiffForm::from_independent_components(m.clone(), &c, 2, vec![s("x")]).unwrap();
        let d = top.exterior_derivative(&c).unwrap();
        assert_eq!(d.degree(), 3);
        assert_eq!(d.vanishes(&c).unwrap(), ZeroTest::Zero);
        let dx = DiffForm::coordinate_form(m.clone(), &c, 0).unwrap();
        let dy = DiffForm::coordinate_form(m.clone(), &c, 1).unwrap();
        let w = dx.wedge(&dx.wedge(&dy, &c).unwrap(), &c).unwrap();
        assert_eq!(w.degree(), 3);
        assert_eq!(w.vanishes(&c).unwrap(), ZeroTest::Zero);
    }

    fn field_x(m: &Arc<DifferentiableManifold>, c: &Chart) -> VectorField {
        VectorField::from_components(m.clone(), c, vec![p("y*z"), p("-x*z"), p("x*y")]).unwrap()
    }

    #[test]
    fn test_interior_product_gp() {
        // gp: i_X A = z^2*x^2 + (z^2 + z)*y^2*x; i_X dA = [(y^2 - z^2)*x^2 - z^2*x,
        //     -y*x^2 + (-z^2 - z)*y*x - z^2*y, -z*x^2 + (-z*y^2 - z^2)*x]
        // (HEAD: interior_product always erred)
        let (m, c) = r3c();
        let a = form_a(&m, &c);
        let x = field_x(&m, &c);
        let ia = a.interior_product(&x, &c).unwrap();
        assert_eq!(ia.degree(), 0);
        assert_value(&dense(&ia, &c)[0], &p("z^2*x^2 + (z^2 + z)*y^2*x"));
        let ida = a.exterior_derivative(&c).unwrap().interior_product(&x, &c).unwrap();
        let v = dense(&ida, &c);
        assert_value(&v[0], &p("(y^2 - z^2)*x^2 - z^2*x"));
        assert_value(&v[1], &p("-y*x^2 + (-z^2 - z)*y*x - z^2*y"));
        assert_value(&v[2], &p("-z*x^2 + (-z*y^2 - z^2)*x"));
        let dx = DiffForm::coordinate_form(m.clone(), &c, 0).unwrap();
        let dy = DiffForm::coordinate_form(m.clone(), &c, 1).unwrap();
        let v = dense(&dx.wedge(&dy, &c).unwrap().interior_product(&x, &c).unwrap(), &c);
        assert_value(&v[0], &p("x*z"));
        assert_value(&v[1], &p("y*z"));
        assert!(literal_zero(&v[2]));
        let f = DiffForm::from_components(m.clone(), &c, 0, vec![s("x")]).unwrap();
        assert!(matches!(f.interior_product(&x, &c), Err(ManifoldError::InvalidOperation(_))));
    }

    #[test]
    fn test_lie_derivative_cartan_gp() {
        // gp: (L_X A)_i = X^j D(A_i, j) + A_j D(X^j, i), independent of Cartan's formula:
        // L_X A = [(y^2 - z^2)*x^2 + z^2*x + (z^2 + z)*y^2, -y*x^2 + (z^2 + z)*y*x - z^2*y,
        //          z*x^2 + ((z + 1)*y^2 - z^2)*x]
        // d(L_X A) = L_X dA = [-2*y*x^2 - 2*y*x + (-z^2 - z)*y, 2*z*x^2 + (-z*y^2 - z^2), y*x + 2*z*y]
        // X(f) = -z*x^3 + 2*z*y^2*x for f = x^2*y. (HEAD: L_X of a 0-form returned the form.)
        let (m, c) = r3c();
        let a = form_a(&m, &c);
        let x = field_x(&m, &c);
        let la = dense(&a.lie_derivative(&x, &c).unwrap(), &c);
        assert_value(&la[0], &p("(y^2 - z^2)*x^2 + z^2*x + (z^2 + z)*y^2"));
        assert_value(&la[1], &p("-y*x^2 + (z^2 + z)*y*x - z^2*y"));
        assert_value(&la[2], &p("z*x^2 + ((z + 1)*y^2 - z^2)*x"));
        let lda = a.exterior_derivative(&c).unwrap().lie_derivative(&x, &c).unwrap();
        assert_eq!(lda.degree(), 2);
        let v = values(&lda, &c);
        assert_value(&v[0], &p("-2*y*x^2 - 2*y*x + (-z^2 - z)*y"));
        assert_value(&v[1], &p("2*z*x^2 + (-z*y^2 - z^2)"));
        assert_value(&v[2], &p("y*x + 2*z*y"));
        let f = DiffForm::from_components(m.clone(), &c, 0, vec![p("x^2*y")]).unwrap();
        let lf = f.lie_derivative(&x, &c).unwrap();
        assert_eq!(lf.degree(), 0);
        assert_value(&dense(&lf, &c)[0], &p("-z*x^3 + 2*z*y^2*x"));
        // the top form: d(i_X ω) only
        let top = DiffForm::from_independent_components(m.clone(), &c, 3, vec![int(1)]).unwrap();
        let lt = top.lie_derivative(&x, &c).unwrap();
        assert_eq!(lt.degree(), 3);
        // div X = 0 for this X
        assert_eq!(lt.vanishes(&c).unwrap(), ZeroTest::Zero);
    }

    #[test]
    fn test_add_neg_scale() {
        // HEAD: Add returned the left operand and Neg returned its input
        let (m, c) = r2();
        let xdx = one_form(&m, &c, vec![s("x"), int(0)]);
        let three_dy = one_form(&m, &c, vec![int(0), int(3)]);
        let sum = (xdx.clone() + three_dy).unwrap();
        assert_eq!(dense(&sum, &c), vec![s("x"), int(3)]);
        let negated = dense(&(-xdx.clone()), &c);
        assert_value(&negated[0], &(-s("x")));
        assert!(literal_zero(&negated[1]));
        let dxdy = DiffForm::from_independent_components(m.clone(), &c, 2, vec![int(1)]).unwrap();
        assert!(matches!(xdx.clone() + dxdy.clone(), Err(ManifoldError::DimensionMismatch { .. })));
        let other = Chart::new("other", 2, vec!["x", "y"]).unwrap();
        let elsewhere = one_form(&m, &other, vec![int(1), int(0)]);
        assert_eq!((xdx.clone() + elsewhere).unwrap_err(), ManifoldError::NoComponentsInChart);
        let half = Expr::Rational(Rational::new(1, 2).unwrap());
        let scaled = dense(&dxdy.scale(&half).unwrap(), &c);
        assert!(literal_zero(&scaled[0]) && literal_zero(&scaled[3]));
        assert_value(&scaled[1], &half);
        assert_value(&scaled[2], &Expr::Rational(Rational::new(-1, 2).unwrap()));
        assert!(dense(&dxdy.scale(&int(0)).unwrap(), &c).iter().all(literal_zero));
        assert_eq!(dxdy.scale(&int(1)).unwrap(), dxdy);
        // -(-ω) has the value of ω
        let a = DiffForm::from_independent_components(m.clone(), &c, 2, vec![p("x*y + 1")]).unwrap();
        let back = (-(-a.clone()) + (-a)).unwrap();
        assert_eq!(back.vanishes(&c).unwrap(), ZeroTest::Zero);
    }

    #[test]
    fn test_from_components_antisymmetry() {
        // HEAD accepted [5, 1, 7, 0] as a 2-form
        let (m, c) = r2();
        let two = |v: Vec<Expr>| DiffForm::from_components(m.clone(), &c, 2, v);
        assert!(matches!(two(vec![int(5), int(1), int(7), int(0)]), Err(ManifoldError::ValidationError(_))));
        assert!(matches!(two(vec![int(0), int(1), int(2), int(0)]), Err(ManifoldError::ValidationError(_))));
        assert!(two(vec![int(0), int(1), int(-1), int(0)]).is_ok());
        let sin2 = s("x").sin().pow(int(2));
        let cos2m1 = s("x").cos().pow(int(2)) - int(1);
        assert!(matches!(two(vec![int(0), sin2, cos2m1, int(0)]), Err(ManifoldError::Undecided(_))));
        assert!(DiffForm::from_components(m.clone(), &c, 1, vec![int(5), int(7)]).is_ok());
        // equal in value, not structurally: accepted and rebuilt from the increasing slot
        let w = two(vec![int(0) * s("y"), p("x*y"), p("-(y*x)"), int(0)]).unwrap();
        let v = dense(&w, &c);
        assert!(literal_zero(&v[0]));
        assert_eq!(v[2], (-v[1].clone()).simplify());
    }

    #[test]
    fn test_non_finite_components_refused() {
        // simplify(0 * oo) is 0 at HEAD, so infinities are refused before any simplification
        let (m, c) = r2();
        let oo = s("oo");
        let bad: Vec<Vec<Expr>> = vec![
            vec![oo.clone(), int(0)],
            vec![int(0) * oo.clone(), s("x")],
            vec![Expr::Real(f64::NAN), int(0)],
            vec![oo.clone().sin(), int(0)],
        ];
        for comps in bad {
            assert!(matches!(
                DiffForm::from_components(m.clone(), &c, 1, comps),
                Err(ManifoldError::ValidationError(_))
            ));
        }
        assert!(matches!(
            DiffForm::from_components(m.clone(), &c, 2, vec![int(0), oo.clone(), -oo.clone(), int(0)]),
            Err(ManifoldError::ValidationError(_))
        ));
        assert!(matches!(
            DiffForm::from_independent_components(m.clone(), &c, 2, vec![Expr::Real(f64::INFINITY)]),
            Err(ManifoldError::ValidationError(_))
        ));
        let dxdy = DiffForm::from_independent_components(m.clone(), &c, 2, vec![int(1)]).unwrap();
        assert!(matches!(dxdy.scale(&oo), Err(ManifoldError::ValidationError(_))));
        let dx = DiffForm::coordinate_form(m.clone(), &c, 0).unwrap();
        let x = VectorField::from_components(m.clone(), &c, vec![oo.clone(), int(0)]).unwrap();
        assert!(matches!(dx.interior_product(&x, &c), Err(ManifoldError::ValidationError(_))));
    }

    #[test]
    fn test_from_independent_components() {
        let (m, c) = r3c();
        let w = DiffForm::from_independent_components(m.clone(), &c, 2, vec![p("y^2"), p("z + x"), p("z^2")]).unwrap();
        let v = dense(&w, &c);
        let (a, b, cc) = (p("y^2"), p("z + x"), p("z^2"));
        assert_eq!(v[1], a);
        assert_eq!(v[2], b);
        assert_eq!(v[5], cc);
        assert_eq!(v[3], (-a).simplify());
        assert_eq!(v[6], (-b).simplify());
        assert_eq!(v[7], (-cc).simplify());
        for k in [0, 4, 8] {
            assert!(literal_zero(&v[k]));
        }
        assert!(matches!(
            DiffForm::from_independent_components(m.clone(), &c, 2, vec![int(1), int(2)]),
            Err(ManifoldError::DimensionMismatch { expected: 3, actual: 2 })
        ));
        assert_eq!(
            w.independent_components(&c).unwrap().into_iter().map(|(t, _)| t).collect::<Vec<_>>(),
            vec![vec![0, 1], vec![0, 2], vec![1, 2]]
        );
        let high = DiffForm::from_independent_components(m.clone(), &c, 4, vec![]).unwrap();
        assert_eq!(high.degree(), 4);
    }

    #[test]
    fn test_is_closed_three_valued() {
        // HEAD: is_closed was the structural zero of the facade d, Ok(false) for closed forms
        let (m, c) = r3c();
        let a = form_a(&m, &c);
        assert!(!a.is_closed(&c).unwrap());
        assert_eq!(a.closedness(&c).unwrap(), ZeroTest::NonZero);
        assert!(a.exterior_derivative(&c).unwrap().is_closed(&c).unwrap());
        let f = DiffForm::from_components(m.clone(), &c, 0, vec![p("x^2*y*z^3 + x*y")]).unwrap();
        assert!(f.exterior_derivative(&c).unwrap().is_closed(&c).unwrap());
        // x1 dx1∧dx2 + dx3∧dx4 on R^4: d(x1 dx1∧dx2) = dx1∧dx1∧dx2 = 0
        let m4 = Arc::new(EuclideanSpace::new(4).manifold().clone());
        let c4 = m4.default_chart().unwrap().clone();
        let w = DiffForm::from_independent_components(
            m4.clone(), &c4, 2,
            vec![s("x1"), int(0), int(0), int(0), int(0), int(1)],
        ).unwrap();
        assert!(w.is_closed(&c4).unwrap());
        // closed, but not decidable here: d = -(sin^2 + cos^2 - 1) dx∧dy
        let (m2, c2) = r2();
        let sc = s("x").sin().pow(int(2)) + s("x").cos().pow(int(2));
        let g = sc.clone() + s("y") * (sc - int(1));
        let w = one_form(&m2, &c2, vec![g, int(0)]);
        assert!(matches!(w.is_closed(&c2), Err(ManifoldError::Undecided(_))));
        assert_eq!(w.closedness(&c2).unwrap(), ZeroTest::Unresolved);
    }
}
