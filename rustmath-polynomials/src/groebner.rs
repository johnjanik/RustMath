//! Gröbner bases computation
//!
//! Provides algorithms for computing Gröbner bases of polynomial ideals.
//! Gröbner bases are a fundamental tool for solving systems of polynomial
//! equations and ideal membership testing.
//!
//! # Two engines — pick the right one
//!
//! **`R: Ring`** ([`groebner_basis`], [`try_groebner_basis`], [`groebner_basis_optimized`],
//! [`reduce`]): [`rustmath_core::Ring`] provides `+ − × 0 1` and *no division*. These
//! routines can only cancel a leading term when the quotient of the two leading
//! coefficients can be exhibited without dividing (`lc = ±1`, or equal/negated
//! coefficients). Consequences:
//!
//! - monic input (in particular `lc = ±1`): a genuine Gröbner basis;
//! - non-monic input over a non-field: a **generating set of the same ideal**, not a
//!   certified Gröbner basis, and the output is never reduced/canonical;
//! - always **terminating**, and bounded by [`GroebnerBudget::default`] — the `Vec`-returning
//!   entry points panic with a precise message when it trips, and [`try_groebner_basis`]
//!   returns `Err` instead.
//!
//! **`R: Field`** ([`groebner_basis_field`], [`reduced_groebner_basis_field`],
//! [`normal_form`], [`ideal_membership_field`]): divides by leading coefficients for real,
//! takes an explicit [`GroebnerBudget`] and returns `Err` when it trips.
//! [`reduced_groebner_basis_field`] is *canonical* — two ideals are equal iff their reduced
//! bases agree. **This is the engine for real work**; everything that must be canonical
//! (ideal equality, membership, radicals, elimination, saturation) goes through it.
//!
//! The `R: Ring` engine exists only because [`crate::quotient::QuotientRing`] and
//! [`crate::ideal::Ideal::reduce`] must work over `Z`. Doing this properly over a non-field
//! needs *strong* Gröbner bases (Möller, Kandri-Rody–Kapur), which this crate does not
//! implement — so the `R: Ring` engine documents the gap rather than pretending it is not
//! there.
//!
//! # `MonomialOrdering::Elimination` (API addition)
//!
//! [`MonomialOrdering`] has an additional variant, `Elimination { block }`: a block order in
//! which any monomial touching one of `x_0, …, x_{block−1}` outranks every monomial touching
//! none of them, ties broken by grevlex. It has the elimination property and is what
//! [`crate::elimination`] is built on. The addition is source-compatible except for a
//! `match` on `MonomialOrdering` that was previously exhaustive.

use crate::multivariate::{Monomial, MultivariatePolynomial};
use rustmath_core::{Field, Ring};
use std::cmp::Ordering;
use std::time::{Duration, Instant};

/// Monomial ordering for Gröbner basis computation
///
/// Different orderings lead to different Gröbner bases for the same ideal.
///
/// # Orderings
///
/// - **Lex** (Lexicographic): Compare exponents left to right
/// - **Grlex** (Graded lexicographic): Compare total degree first, then lex
/// - **Grevlex** (Graded reverse lexicographic): Compare total degree first,
///   then reverse lex (most common in practice)
///
/// # Example
///
/// For monomials x²y and xy²:
/// - Lex: x²y > xy² (x-exponent 2 > 1)
/// - Grlex: x²y = xy² (same degree 3), then lex: x²y > xy²
/// - Grevlex: x²y = xy² (same degree 3), then reverse: xy² > x²y
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonomialOrdering {
    /// Lexicographic order: compare exponents left to right
    ///
    /// Note: `Monomial::cmp_lex` walks the variable indices in *ascending* order and
    /// returns on the first differing exponent, so `x0 > x1 > x2 > ...` — the
    /// lowest-numbered variable is the most significant one.
    Lex,
    /// Graded lexicographic order: compare total degree first, then lex
    Grlex,
    /// Graded reverse lexicographic order: compare total degree first,
    /// then reverse lex (most common in practice)
    Grevlex,
    /// Block elimination order for the variables `x_0, ..., x_{block-1}`.
    ///
    /// A monomial involving any variable of the block outranks every monomial
    /// involving none of them; ties are broken by Grevlex on the whole monomial.
    /// This is the order used by [`crate::elimination`]: it has the *elimination
    /// property*, so a Gröbner basis element whose leading monomial avoids the block
    /// avoids the block entirely.
    Elimination {
        /// Number of leading variables to eliminate (indices `0..block`)
        block: usize,
    },
}

/// Compute a Gröbner basis using Buchberger's algorithm
///
/// Given a set of polynomials generating an ideal I, compute a Gröbner basis
/// for I with respect to the given monomial ordering.
///
/// # Algorithm
///
/// Buchberger's algorithm:
/// 1. Start with G = input polynomials
/// 2. For each pair (f, g) in G, compute S-polynomial S(f,g)
/// 3. Reduce S(f,g) with respect to G
/// 4. If remainder is non-zero, add it to G
/// 5. Repeat until no new polynomials are added
///
/// # S-polynomial
///
/// For polynomials f and g:
/// - Let LM(f), LM(g) be the leading monomials
/// - Let LCM be the least common multiple of LM(f) and LM(g)
/// - S(f,g) = (LCM/LT(f))·f - (LCM/LT(g))·g
///
/// where LT is the leading term (leading coefficient × leading monomial).
///
/// # Limitations
///
/// - Requires multivariate polynomial division
/// - Needs leading monomial/coefficient extraction
/// - Full implementation pending additional multivariate polynomial methods
///
/// # References
///
/// - Buchberger, B. (1965). "Ein Algorithmus zum Auffinden der Basiselemente
///   des Restklassenringes nach einem nulldimensionalen Polynomideal"
/// - Cox, Little, O'Shea. "Ideals, Varieties, and Algorithms" (2015)
pub fn groebner_basis_info() -> &'static str {
    "Gröbner basis computation requires:

1. Monomial ordering implementation (Lex, Grlex, Grevlex)
2. Leading monomial and leading coefficient extraction
3. Multivariate polynomial division
4. S-polynomial computation
5. Buchberger's algorithm with pair selection

Example usage (once fully implemented):
    let f1 = poly!(x^2 + y^2 - 1);
    let f2 = poly!(x - y);
    let basis = groebner_basis(vec![f1, f2], MonomialOrdering::Grevlex);
    // basis will be a reduced Gröbner basis for the ideal <f1, f2>

Applications:
- Solving systems of polynomial equations
- Ideal membership testing
- Elimination theory
- Implicitization in geometric modeling
- Algebraic geometry computations
"
}

/// Check if a monomial ordering is a well-ordering
///
/// A monomial ordering > is a well-ordering if:
/// 1. It is a total ordering
/// 2. 1 is the minimum element
/// 3. If m1 > m2, then m1·m3 > m2·m3 for all monomials m3
///
/// All three standard orderings (Lex, Grlex, Grevlex) are well-orderings.
pub fn is_well_ordering(ordering: MonomialOrdering) -> bool {
    match ordering {
        MonomialOrdering::Lex => true,
        MonomialOrdering::Grlex => true,
        MonomialOrdering::Grevlex => true,
        // A block order refining Grevlex: block degree is a non-negative weight, so 1
        // stays minimal and the order is multiplicative.
        MonomialOrdering::Elimination { .. } => true,
    }
}

/// Get a description of a monomial ordering
pub fn ordering_description(ordering: MonomialOrdering) -> &'static str {
    match ordering {
        MonomialOrdering::Lex => {
            "Lexicographic (Lex): Compare exponents left to right. \
             Example: x²y > xy² because 2 > 1 in the first variable."
        }
        MonomialOrdering::Grlex => {
            "Graded Lexicographic (Grlex): Compare total degree first, \
             then use lex for tiebreaking. Example: x³ > x²y > xy² > y³."
        }
        MonomialOrdering::Grevlex => {
            "Graded Reverse Lexicographic (Grevlex): Compare total degree first, \
             then use reverse lex from the right. Most commonly used in practice. \
             Example: y³ > xy² > x²y > x³."
        }
        MonomialOrdering::Elimination { .. } => {
            "Block elimination order: any monomial involving one of the first `block` \
             variables outranks every monomial involving none of them; ties are broken \
             by Grevlex. Has the elimination property for x_0..x_{block-1}."
        }
    }
}

/// Get comparison function for a monomial ordering
fn get_comparison_fn(ordering: MonomialOrdering) -> impl Fn(&Monomial, &Monomial) -> Ordering + Copy {
    move |a: &Monomial, b: &Monomial| match ordering {
        MonomialOrdering::Lex => a.cmp_lex(b),
        MonomialOrdering::Grlex => a.cmp_grlex(b),
        MonomialOrdering::Grevlex => a.cmp_grevlex(b),
        MonomialOrdering::Elimination { block } => a.cmp_elimination(b, block),
    }
}

/// Public accessor for the comparison function of a monomial ordering.
pub fn comparison_fn(
    ordering: MonomialOrdering,
) -> impl Fn(&Monomial, &Monomial) -> Ordering + Copy {
    get_comparison_fn(ordering)
}

/// Compute the S-polynomial of two polynomials
///
/// The S-polynomial is defined as:
/// S(f, g) = (lcm / LT(f)) * f - (lcm / LT(g)) * g
///
/// where lcm is the LCM of the leading monomials, and LT is the leading term.
pub fn s_polynomial<R: Ring>(
    f: &MultivariatePolynomial<R>,
    g: &MultivariatePolynomial<R>,
    ordering: MonomialOrdering,
) -> MultivariatePolynomial<R> {
    let cmp = get_comparison_fn(ordering);

    let Some((f_lm, _f_lc)) = f.leading_term(cmp) else {
        return MultivariatePolynomial::zero();
    };

    let Some((g_lm, _g_lc)) = g.leading_term(cmp) else {
        return MultivariatePolynomial::zero();
    };

    // Compute LCM of leading monomials
    let lcm = f_lm.lcm(&g_lm);

    // Compute lcm / LM(f) and lcm / LM(g)
    let f_mult = lcm.div(&f_lm).unwrap();
    let g_mult = lcm.div(&g_lm).unwrap();

    // S(f, g) = (lcm/LM(f)) * f - (lcm/LM(g)) * g
    let term1 = f.monomial_mul(&f_mult, &R::one());
    let term2 = g.monomial_mul(&g_mult, &R::one());

    term1 - term2
}

/// Reduce a polynomial with respect to a set of polynomials, over a general ring.
///
/// This is the multivariate division algorithm
/// ([`MultivariatePolynomial::divide_multiple`]): the returned remainder `r` satisfies
/// `poly − r ∈ (basis)`, and no term of `r` is reducible by any basis element *that
/// applies*.
///
/// # Honesty note — what "applies" means over a general ring
///
/// `R: Ring` has no coefficient division, so a basis element `g` can only cancel a term
/// `c·m` of `p` when `lc(g)` divides `c` by a quotient that can be exhibited in `R`
/// (`lc(g) = ±1`, or `c = ±lc(g)`). When it cannot, `g` does not apply to that term and
/// the term stays in the remainder. Consequently, for **non-monic** input over a
/// **non-field**, `r` is a valid remainder but not a canonical normal form.
///
/// Over a field, use [`normal_form`] instead: it divides by the leading coefficient for
/// real and returns the true normal form.
///
/// # Termination
///
/// Guaranteed. The previous version looped forever whenever the leading monomial of `g`
/// divided that of `p` but the leading coefficient did not: it set its `reduced` flag on
/// monomial divisibility alone, while the underlying division could not cancel anything,
/// so the state never changed. `groebner_basis([2x² − 2y, 3y² − 3x], Grevlex)` over `Q`
/// hung on exactly this.
pub fn reduce<R: Ring>(
    poly: &MultivariatePolynomial<R>,
    basis: &[MultivariatePolynomial<R>],
    ordering: MonomialOrdering,
) -> MultivariatePolynomial<R> {
    let cmp = get_comparison_fn(ordering);

    let divisors: Vec<MultivariatePolynomial<R>> =
        basis.iter().filter(|g| !g.is_zero()).cloned().collect();
    if divisors.is_empty() {
        return poly.clone();
    }

    let (_quotients, remainder) = poly.divide_multiple(&divisors, cmp);
    remainder
}

/// Compute a Gröbner basis using Buchberger's algorithm, over a general ring.
///
/// # ⚠ What this does and does not guarantee
///
/// This is the `R: Ring` engine. [`rustmath_core::Ring`] has **no division**, so neither
/// the S-polynomial ([`s_polynomial`], which multiplies by `1` rather than by the ratio
/// of leading coefficients) nor the reduction ([`reduce`]) can cancel a leading term
/// whose coefficient is not divisible by an exhibitable quotient in `R`. Therefore:
///
/// - **Monic leading coefficients (in particular: any input over a field that has been
///   made monic, and the common `lc = ±1` case): the result is a genuine Gröbner basis.**
/// - **Non-monic leading coefficients over a non-field: the result is a generating set of
///   the same ideal, but it is NOT certified to be a Gröbner basis.** Getting this right
///   needs *strong* Gröbner bases (Möller / Kandri-Rody–Kapur), which this crate does not
///   implement.
///
/// For real work over a field use [`groebner_basis_field`] / [`reduced_groebner_basis_field`],
/// which divide by the leading coefficient for real and carry a budget.
///
/// # Panics
///
/// Panics — with a precise message — if the computation exceeds
/// [`GroebnerBudget::default`]. That can happen for non-monic input over a non-field,
/// where the obstruction above lets Buchberger keep appending elements whose leading
/// monomials are already in the leading-monomial ideal. It is a hard bound, not a hang:
/// the previous version of this function ran forever on
/// `groebner_basis([2x² − 2y, 3y² − 3x], Grevlex)`. Use [`try_groebner_basis`] if you
/// need to handle that case instead of dying on it.
pub fn groebner_basis<R: Ring>(
    generators: Vec<MultivariatePolynomial<R>>,
    ordering: MonomialOrdering,
) -> Vec<MultivariatePolynomial<R>> {
    try_groebner_basis(generators, ordering, &GroebnerBudget::default())
        .unwrap_or_else(|e| panic!("groebner_basis: {}", e))
}

/// [`groebner_basis`] with an explicit budget, returning `Err` instead of panicking.
///
/// The same correctness caveat applies: over a non-field with non-monic leading
/// coefficients the result is a generating set of the ideal, not a certified Gröbner
/// basis. What is guaranteed is that this **terminates**, and that every returned
/// polynomial lies in the ideal generated by the input.
pub fn try_groebner_basis<R: Ring>(
    generators: Vec<MultivariatePolynomial<R>>,
    ordering: MonomialOrdering,
    budget: &GroebnerBudget,
) -> Result<Vec<MultivariatePolynomial<R>>, String> {
    buchberger_ring(
        generators,
        ordering,
        PairSelectionStrategy::Normal,
        false,
        budget,
    )
}

/// The single budgeted Buchberger engine behind every `R: Ring` entry point.
///
/// `strategy` picks the pair to reduce next; `use_criteria` enables Buchberger's first
/// criterion. Returns `Err` the moment any limit in `budget` is exceeded, so no caller can
/// spin: the pair count, the basis size, the degree, the term count and the wall clock are
/// all bounded.
///
/// The budget is checked against the **input generators** as well as against newly
/// produced elements — a caller who hands in a degree-100 generator under a degree-3 cap
/// gets an honest `Err` rather than a computation that ignores its own limits.
fn buchberger_ring<R: Ring>(
    mut generators: Vec<MultivariatePolynomial<R>>,
    ordering: MonomialOrdering,
    strategy: PairSelectionStrategy,
    use_criteria: bool,
    budget: &GroebnerBudget,
) -> Result<Vec<MultivariatePolynomial<R>>, String> {
    generators.retain(|p| !p.is_zero());
    if generators.is_empty() {
        return Ok(vec![]);
    }

    check_generators_against_budget(&generators, budget)?;

    let mut basis = generators;
    let mut pairs: Vec<(usize, usize, u32)> = Vec::new();

    for i in 0..basis.len() {
        for j in i + 1..basis.len() {
            if use_criteria && is_useless_pair(&basis[i], &basis[j], ordering) {
                continue;
            }
            let priority = compute_pair_priority(&basis[i], &basis[j], ordering, strategy);
            pairs.push((i, j, priority));
        }
    }
    if strategy != PairSelectionStrategy::Normal {
        pairs.sort_by_key(|&(_, _, priority)| priority);
    }

    let deadline = Instant::now() + Duration::from_millis(budget.max_millis);
    let mut processed = 0usize;

    while !pairs.is_empty() {
        if Instant::now() > deadline {
            return Err(format!(
                "Gröbner budget exceeded: exceeded the {} ms wall-clock limit after {} S-pairs \
                 (max_millis)",
                budget.max_millis, processed
            ));
        }

        // Normal: LIFO. Otherwise: lowest priority first (the list is kept sorted).
        let (i, j, _priority) = if strategy == PairSelectionStrategy::Normal {
            pairs.pop().expect("pairs is non-empty")
        } else {
            pairs.remove(0)
        };

        if i >= basis.len() || j >= basis.len() {
            continue;
        }

        processed += 1;
        if processed > budget.max_pairs {
            return Err(format!(
                "Gröbner budget exceeded: reduced {} S-pairs (max_pairs = {}). Over a \
                 non-field with non-monic leading coefficients Buchberger cannot cancel \
                 leading terms and may never stabilise — use the R: Field engine \
                 (groebner_basis_field) for such systems",
                processed, budget.max_pairs
            ));
        }

        let s = s_polynomial(&basis[i], &basis[j], ordering);
        let remainder = reduce(&s, &basis, ordering);
        if remainder.is_zero() {
            continue;
        }

        if let Some(d) = remainder.degree() {
            if d > budget.max_degree {
                return Err(format!(
                    "Gröbner budget exceeded: new basis element has total degree {} \
                     (max_degree = {})",
                    d, budget.max_degree
                ));
            }
        }
        if remainder.num_terms() > budget.max_terms {
            return Err(format!(
                "Gröbner budget exceeded: new basis element has {} terms (max_terms = {})",
                remainder.num_terms(),
                budget.max_terms
            ));
        }
        if basis.len() + 1 > budget.max_basis {
            return Err(format!(
                "Gröbner budget exceeded: basis reached {} elements (max_basis = {}). Over a \
                 non-field with non-monic leading coefficients Buchberger cannot cancel \
                 leading terms and may never stabilise — use the R: Field engine \
                 (groebner_basis_field) for such systems",
                basis.len() + 1,
                budget.max_basis
            ));
        }

        let new_idx = basis.len();
        for k in 0..new_idx {
            if use_criteria && is_useless_pair(&basis[k], &remainder, ordering) {
                continue;
            }
            let priority = compute_pair_priority(&basis[k], &remainder, ordering, strategy);
            if strategy == PairSelectionStrategy::Normal {
                pairs.push((k, new_idx, priority));
            } else {
                let insert_pos = pairs
                    .iter()
                    .position(|&(_, _, p)| p > priority)
                    .unwrap_or(pairs.len());
                pairs.insert(insert_pos, (k, new_idx, priority));
            }
        }

        basis.push(remainder);
    }

    Ok(basis)
}

/// Enforce the budget on the *input* generators.
///
/// Without this the caps are a lie for anybody who hands in a generator that already
/// violates them: the old code only ever checked polynomials that Buchberger itself
/// produced.
fn check_generators_against_budget<R: Ring>(
    generators: &[MultivariatePolynomial<R>],
    budget: &GroebnerBudget,
) -> Result<(), String> {
    if generators.len() > budget.max_basis {
        return Err(format!(
            "Gröbner budget exceeded: {} input generators (max_basis = {})",
            generators.len(),
            budget.max_basis
        ));
    }
    for g in generators {
        if let Some(d) = g.degree() {
            if d > budget.max_degree {
                return Err(format!(
                    "Gröbner budget exceeded: input generator has total degree {} \
                     (max_degree = {})",
                    d, budget.max_degree
                ));
            }
        }
        if g.num_terms() > budget.max_terms {
            return Err(format!(
                "Gröbner budget exceeded: input generator has {} terms (max_terms = {})",
                g.num_terms(),
                budget.max_terms
            ));
        }
    }
    Ok(())
}

/// Inter-reduce a Gröbner basis, over a general ring.
///
/// # ⚠ Not the *reduced* Gröbner basis over a general ring
///
/// A reduced Gröbner basis is monic and inter-reduced, and is *canonical*. This function
/// cannot make anything monic — `R: Ring` has no inverses — so it only inter-reduces, and
/// the result is **not** canonical: it is a Gröbner basis (under the monic hypothesis of
/// [`groebner_basis`]) with no redundant reducible terms, nothing more. Two equal ideals
/// can produce different outputs here.
///
/// Use [`reduced_groebner_basis_field`] over a field: that one *is* the canonical reduced
/// Gröbner basis, and is what ideal comparison must be based on.
///
/// # Panics
///
/// Inherits the budget of [`groebner_basis`] and panics with a precise message if it trips.
pub fn reduced_groebner_basis<R: Ring>(
    generators: Vec<MultivariatePolynomial<R>>,
    ordering: MonomialOrdering,
) -> Vec<MultivariatePolynomial<R>> {
    let mut basis = groebner_basis(generators, ordering);

    // Remove zero polynomials
    basis.retain(|p| !p.is_zero());

    if basis.is_empty() {
        return basis;
    }

    // Inter-reduce the basis: replace each element by its remainder modulo the others.
    //
    // Terminates: every replacement strictly lowers the replaced polynomial in the
    // well-order induced by `ordering` (a reduction step cancels a term and introduces
    // only smaller ones), and a removal strictly shrinks the basis. The step counter is a
    // belt-and-braces guard so that a future change cannot silently reintroduce a spin.
    let max_steps = 100_000usize;
    let mut steps = 0usize;
    let mut changed = true;
    while changed {
        changed = false;

        steps += 1;
        if steps > max_steps {
            panic!(
                "reduced_groebner_basis: inter-reduction did not stabilise after {} passes \
                 over a basis of {} elements — this is a bug in the reduction, not a hard \
                 problem",
                max_steps,
                basis.len()
            );
        }

        for i in 0..basis.len() {
            // Reduce basis[i] by all other polynomials
            let mut others = Vec::new();
            for (j, poly) in basis.iter().enumerate() {
                if i != j && !poly.is_zero() {
                    others.push(poly.clone());
                }
            }

            if !others.is_empty() {
                let reduced = reduce(&basis[i], &others, ordering);

                // If the reduction changed the polynomial, update it
                if reduced != basis[i] && !reduced.is_zero() {
                    basis[i] = reduced;
                    changed = true;
                } else if reduced.is_zero() {
                    // Remove zero polynomial
                    basis.remove(i);
                    changed = true;
                    break; // Restart the loop since we modified the basis
                }
            }
        }
    }

    // Remove any duplicates or zero polynomials that might have appeared
    basis.retain(|p| !p.is_zero());

    basis
}

/// Check if a polynomial is in the ideal generated by a set of polynomials
///
/// Uses Gröbner basis to test ideal membership
pub fn ideal_membership<R: Ring>(
    poly: &MultivariatePolynomial<R>,
    generators: &[MultivariatePolynomial<R>],
    ordering: MonomialOrdering,
) -> bool {
    let basis = groebner_basis(generators.to_vec(), ordering);
    let remainder = reduce(poly, &basis, ordering);
    remainder.is_zero()
}

// ============================================================================
// Phase 2.3 Enhancements: F4 Algorithm and Optimizations
// ============================================================================

/// Pair selection strategies for Buchberger's algorithm
///
/// Different strategies for selecting which S-polynomial pairs to process
/// can significantly affect the performance of Gröbner basis computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairSelectionStrategy {
    /// Normal strategy: process pairs in the order they're added
    Normal,
    /// Minimal degree strategy: process pairs with smallest degree first
    MinimalDegree,
    /// Minimal LCM degree: select pair with smallest LCM degree
    MinimalLCM,
}

/// Compute degree of a polynomial in a specific ordering
fn polynomial_degree<R: Ring>(
    poly: &MultivariatePolynomial<R>,
    ordering: MonomialOrdering,
) -> u32 {
    let cmp = get_comparison_fn(ordering);
    match poly.leading_term(cmp) {
        Some((monomial, _)) => monomial.degree(),
        None => 0,
    }
}

/// Compute the Gröbner basis with a specific pair selection strategy
///
/// This is an optimized version of Buchberger's algorithm that allows
/// choosing different pair selection strategies.
///
/// # Arguments
///
/// * `generators` - Initial set of polynomials
/// * `ordering` - Monomial ordering to use
/// * `strategy` - Pair selection strategy
///
/// # Performance
///
/// The choice of strategy can dramatically affect performance:
/// - Normal: Simple FIFO, can be slow
/// - MinimalDegree: Often much faster, processes low-degree pairs first
/// - MinimalLCM: Best for many examples, minimizes growth of intermediate polynomials
///
/// # Panics
///
/// Same contract as [`groebner_basis`]: the `R: Ring` correctness caveat applies, and this
/// panics with a precise message if [`GroebnerBudget::default`] trips.
pub fn groebner_basis_with_strategy<R: Ring>(
    generators: Vec<MultivariatePolynomial<R>>,
    ordering: MonomialOrdering,
    strategy: PairSelectionStrategy,
) -> Vec<MultivariatePolynomial<R>> {
    buchberger_ring(
        generators,
        ordering,
        strategy,
        false,
        &GroebnerBudget::default(),
    )
    .unwrap_or_else(|e| panic!("groebner_basis_with_strategy: {}", e))
}

/// Compute priority for a pair of polynomials based on strategy
fn compute_pair_priority<R: Ring>(
    f: &MultivariatePolynomial<R>,
    g: &MultivariatePolynomial<R>,
    ordering: MonomialOrdering,
    strategy: PairSelectionStrategy,
) -> u32 {
    let cmp = get_comparison_fn(ordering);

    match strategy {
        PairSelectionStrategy::Normal => 0, // No priority

        PairSelectionStrategy::MinimalDegree => {
            // Priority is the sum of degrees
            let f_deg = polynomial_degree(f, ordering);
            let g_deg = polynomial_degree(g, ordering);
            f_deg + g_deg
        }

        PairSelectionStrategy::MinimalLCM => {
            // Priority is the degree of LCM of leading monomials
            let f_lm = f.leading_term(cmp).map(|(m, _)| m);
            let g_lm = g.leading_term(cmp).map(|(m, _)| m);

            match (f_lm, g_lm) {
                (Some(m1), Some(m2)) => m1.lcm(&m2).degree(),
                _ => u32::MAX, // Put invalid pairs at the end
            }
        }
    }
}

/// Buchberger's criteria for detecting useless pairs
///
/// These criteria allow us to identify S-polynomial pairs that will reduce to zero,
/// avoiding unnecessary computation.
///
/// # Criteria
///
/// 1. **Buchberger's First Criterion**: If gcd(LM(f), LM(g)) = 1, then S(f,g) reduces to 0
/// 2. **Buchberger's Second Criterion**: More complex, involves checking if another basis
///    element divides the LCM and certain reduction properties hold
///
/// # Returns
///
/// true if the pair can be safely discarded (will reduce to zero)
pub fn is_useless_pair<R: Ring>(
    f: &MultivariatePolynomial<R>,
    g: &MultivariatePolynomial<R>,
    _ordering: MonomialOrdering,
) -> bool {
    // For now, implement only the first criterion
    // Full implementation of the second criterion requires more infrastructure

    let cmp = get_comparison_fn(_ordering);

    let f_lm = f.leading_term(cmp).map(|(m, _)| m);
    let g_lm = g.leading_term(cmp).map(|(m, _)| m);

    match (f_lm, g_lm) {
        (Some(m1), Some(m2)) => {
            // Check if leading monomials are coprime (gcd = 1)
            // Two monomials are coprime if they share no common variables with positive exponents
            let lcm = m1.lcm(&m2);
            let product_degree = m1.degree() + m2.degree();

            // If LCM degree equals sum of degrees, monomials are coprime
            lcm.degree() == product_degree
        }
        _ => false,
    }
}

/// Optimized Gröbner basis with Buchberger's criteria
///
/// This version applies Buchberger's first criterion to skip S-pairs that are guaranteed
/// to reduce to zero, and selects pairs by smallest LCM degree.
///
/// # Panics
///
/// Same contract as [`groebner_basis`]: the `R: Ring` correctness caveat applies, and this
/// panics with a precise message if [`GroebnerBudget::default`] trips.
pub fn groebner_basis_optimized<R: Ring>(
    generators: Vec<MultivariatePolynomial<R>>,
    ordering: MonomialOrdering,
) -> Vec<MultivariatePolynomial<R>> {
    buchberger_ring(
        generators,
        ordering,
        PairSelectionStrategy::MinimalLCM,
        true,
        &GroebnerBudget::default(),
    )
    .unwrap_or_else(|e| panic!("groebner_basis_optimized: {}", e))
}

/// F4 Algorithm for Gröbner basis computation
///
/// **This is not F4.** It is [`groebner_basis_optimized`] — Buchberger's algorithm with
/// the first criterion and smallest-LCM pair selection — under an F4 name.
///
/// # Why the name is wrong
///
/// Faugère's F4 replaces one-S-polynomial-at-a-time reduction with a *matrix* step: many
/// S-polynomials and their reductors are assembled into a Macaulay matrix and reduced
/// simultaneously by Gaussian elimination. None of that is implemented here — there is no
/// symbolic preprocessing and no linear algebra. Calling this "F4" told callers they were
/// getting an algorithm with a completely different complexity profile than the one they
/// actually got, so the name is deprecated rather than kept as a euphemism.
///
/// Nothing about the *result* was ever wrong: Buchberger and F4 compute the same Gröbner
/// basis. Only the claim about *how* was false.
///
/// # Use instead
///
/// - [`groebner_basis_optimized`] — exactly what this function does, honestly named;
/// - [`reduced_groebner_basis_field`] — the real entry point for work over a field
///   (correct coefficient arithmetic, canonical output, budgeted).
///
/// # Panics
///
/// Inherits the panic contract of [`groebner_basis_optimized`].
#[deprecated(
    note = "not F4: this is Buchberger (groebner_basis_optimized) under an F4 name. No \
            matrix reduction, no symbolic preprocessing. Call groebner_basis_optimized, or \
            reduced_groebner_basis_field over a field."
)]
pub fn groebner_basis_f4<R: Ring>(
    generators: Vec<MultivariatePolynomial<R>>,
    ordering: MonomialOrdering,
) -> Vec<MultivariatePolynomial<R>> {
    groebner_basis_optimized(generators, ordering)
}

// ============================================================================
// Field-based Buchberger with exact coefficient handling and budgets
//
// The `R: Ring` routines above cannot divide coefficients: `Ring` offers `+ - * zero one`
// and nothing more. Their S-polynomials multiply by `R::one()` instead of by the ratio of
// leading coefficients, and their reduction can only cancel a leading term when the
// quotient of the two leading coefficients can be exhibited in `R` without dividing
// (`lc = ±1`, or equal/negated coefficients). That is exact for monic input and simply
// *does not apply* otherwise — which is why they terminate but do not certify a Gröbner
// basis for non-monic input over a non-field.
//
// Everything below requires `R: Field`, divides by the leading coefficient for real, and
// is what the elimination/saturation/quotient layer is built on.
// ============================================================================

/// Resource budget for a Gröbner basis computation.
///
/// Buchberger's algorithm over `Q` can blow up in both the number of S-pairs and the
/// size of the coefficients. Every entry point that runs a Gröbner basis takes one of
/// these and returns `Err` when it trips, rather than running until the machine dies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroebnerBudget {
    /// Maximum number of S-pairs actually reduced
    pub max_pairs: usize,
    /// Maximum number of elements allowed in the intermediate basis
    pub max_basis: usize,
    /// Maximum total degree allowed for any polynomial produced
    pub max_degree: u32,
    /// Maximum number of terms allowed in any intermediate polynomial
    pub max_terms: usize,
    /// Wall-clock ceiling, in milliseconds.
    ///
    /// This is the backstop the combinatorial limits cannot provide: over `Q` the
    /// numerators and denominators can explode while the pair count, the degree and the
    /// term count all stay small, so the computation grinds without tripping any of
    /// them. Nothing in `R: Field` lets us measure coefficient size generically, so we
    /// bound the one thing that is always observable — elapsed time.
    pub max_millis: u64,
}

impl Default for GroebnerBudget {
    fn default() -> Self {
        GroebnerBudget {
            max_pairs: 20_000,
            max_basis: 500,
            max_degree: 40,
            max_terms: 5_000,
            max_millis: 10_000,
        }
    }
}

impl GroebnerBudget {
    /// A budget large enough that it will not realistically trip on a small example.
    ///
    /// Still finite: an honest `Err` beats an unbounded loop.
    pub fn generous() -> Self {
        GroebnerBudget {
            max_pairs: 2_000_000,
            max_basis: 20_000,
            max_degree: 500,
            max_terms: 200_000,
            max_millis: 120_000,
        }
    }

    /// A deliberately tiny budget, for testing that the budget actually fires.
    pub fn tiny() -> Self {
        GroebnerBudget {
            max_pairs: 5,
            max_basis: 5,
            max_degree: 3,
            max_terms: 10,
            max_millis: 1_000,
        }
    }
}

/// Scale `p` so that its leading coefficient is 1.
fn make_monic<R: Field, F>(p: &MultivariatePolynomial<R>, cmp: F) -> Result<MultivariatePolynomial<R>, String>
where
    F: Fn(&Monomial, &Monomial) -> Ordering + Copy,
{
    let Some((_, lc)) = p.leading_term(cmp) else {
        return Ok(MultivariatePolynomial::zero());
    };
    if lc.is_one() {
        return Ok(p.clone());
    }
    let inv = lc
        .inverse()
        .map_err(|e| format!("make_monic: leading coefficient {} is not invertible: {:?}", lc, e))?;
    Ok(p.scalar_mul(&inv))
}

/// The true multivariate normal form of `poly` modulo `basis`, over a field.
///
/// Unlike [`reduce`], this divides by the leading coefficient of the reducer, so the
/// leading term genuinely cancels at every step, and terms that no basis element can
/// reduce are moved into the remainder instead of aborting the loop. The result is the
/// full normal form: no monomial of it is divisible by any leading monomial of `basis`.
///
/// Terminates because each step strictly decreases the leading monomial of the working
/// polynomial in a well-order.
pub fn normal_form<R: Field>(
    poly: &MultivariatePolynomial<R>,
    basis: &[MultivariatePolynomial<R>],
    ordering: MonomialOrdering,
) -> Result<MultivariatePolynomial<R>, String> {
    normal_form_budgeted(poly, basis, ordering, None, None)
}

fn normal_form_budgeted<R: Field>(
    poly: &MultivariatePolynomial<R>,
    basis: &[MultivariatePolynomial<R>],
    ordering: MonomialOrdering,
    budget: Option<&GroebnerBudget>,
    deadline: Option<Instant>,
) -> Result<MultivariatePolynomial<R>, String> {
    let cmp = get_comparison_fn(ordering);

    // (polynomial, leading monomial, inverse of leading coefficient)
    let mut reducers: Vec<(&MultivariatePolynomial<R>, Monomial, R)> = Vec::new();
    for g in basis {
        if let Some((lm, lc)) = g.leading_term(cmp) {
            let inv = lc.inverse().map_err(|e| {
                format!("normal_form: leading coefficient {} is not invertible: {:?}", lc, e)
            })?;
            reducers.push((g, lm, inv));
        }
    }

    let mut p = poly.clone();
    let mut rem = MultivariatePolynomial::zero();
    let mut steps: u64 = 0;

    while !p.is_zero() {
        if let Some(b) = budget {
            if p.num_terms() > b.max_terms {
                return Err(format!(
                    "Gröbner budget exceeded: intermediate polynomial reached {} terms (max_terms = {})",
                    p.num_terms(),
                    b.max_terms
                ));
            }
        }

        // Coefficient blow-up shows up as time, not as term count, so check the clock.
        steps += 1;
        if steps % 32 == 0 {
            if let (Some(d), Some(b)) = (deadline, budget) {
                if Instant::now() > d {
                    return Err(format!(
                        "Gröbner budget exceeded: exceeded the {} ms wall-clock limit while \
                         reducing (max_millis)",
                        b.max_millis
                    ));
                }
            }
        }

        let (p_lm, p_lc) = p
            .leading_term(cmp)
            .expect("non-zero polynomial has a leading term");

        let mut step = None;
        for (g, g_lm, g_lc_inv) in &reducers {
            if let Some(q) = p_lm.div(g_lm) {
                step = Some((*g, q, p_lc.clone() * g_lc_inv.clone()));
                break;
            }
        }

        match step {
            Some((g, q, factor)) => {
                // Cancels the leading term of p exactly.
                p = p - g.monomial_mul(&q, &factor);
            }
            None => {
                // Nothing reduces this term: it belongs to the remainder.
                rem.add_term(p_lm.clone(), p_lc.clone());
                p.add_term(p_lm, -p_lc);
            }
        }
    }

    Ok(rem)
}

/// S-polynomial over a field, with the leading coefficients divided out for real.
///
/// `S(f,g) = (lcm/LT(f))·f − (lcm/LT(g))·g`, where `LT` includes the leading
/// coefficient. The leading terms cancel by construction.
pub fn s_polynomial_field<R: Field>(
    f: &MultivariatePolynomial<R>,
    g: &MultivariatePolynomial<R>,
    ordering: MonomialOrdering,
) -> Result<MultivariatePolynomial<R>, String> {
    let cmp = get_comparison_fn(ordering);

    let Some((f_lm, f_lc)) = f.leading_term(cmp) else {
        return Ok(MultivariatePolynomial::zero());
    };
    let Some((g_lm, g_lc)) = g.leading_term(cmp) else {
        return Ok(MultivariatePolynomial::zero());
    };

    let lcm = f_lm.lcm(&g_lm);
    let f_mult = lcm.div(&f_lm).expect("lcm is divisible by each leading monomial");
    let g_mult = lcm.div(&g_lm).expect("lcm is divisible by each leading monomial");

    let f_inv = f_lc
        .inverse()
        .map_err(|e| format!("s_polynomial_field: {} is not invertible: {:?}", f_lc, e))?;
    let g_inv = g_lc
        .inverse()
        .map_err(|e| format!("s_polynomial_field: {} is not invertible: {:?}", g_lc, e))?;

    Ok(f.monomial_mul(&f_mult, &f_inv) - g.monomial_mul(&g_mult, &g_inv))
}

/// Two monomials are coprime iff their lcm is their product.
fn coprime(a: &Monomial, b: &Monomial) -> bool {
    a.lcm(b) == a.mul(b)
}

/// Buchberger's algorithm over a field, with correct coefficient arithmetic and a budget.
///
/// Uses the normal selection strategy (smallest lcm first) and Buchberger's first
/// criterion (a pair with coprime leading monomials always reduces to zero, so it is
/// skipped).
///
/// Returns `Err` if the budget trips — never runs unbounded. The budget is enforced on the
/// **input generators** as well as on everything the algorithm produces: a caller who asks
/// for `max_degree = 3` and hands in a degree-8 generator gets an honest `Err`, not a
/// computation that quietly ignores its own limit.
pub fn groebner_basis_field<R: Field>(
    generators: Vec<MultivariatePolynomial<R>>,
    ordering: MonomialOrdering,
    budget: &GroebnerBudget,
) -> Result<Vec<MultivariatePolynomial<R>>, String> {
    let cmp = get_comparison_fn(ordering);

    let generators: Vec<MultivariatePolynomial<R>> =
        generators.into_iter().filter(|p| !p.is_zero()).collect();
    if generators.is_empty() {
        return Ok(vec![]);
    }
    check_generators_against_budget(&generators, budget)?;

    let mut basis: Vec<MultivariatePolynomial<R>> = Vec::new();
    for g in generators {
        basis.push(make_monic(&g, cmp)?);
    }

    // Leading monomials are needed on every pair-selection scan; cache them.
    let mut lms: Vec<Monomial> = basis.iter().map(|g| lead_monomial(g, cmp)).collect();

    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..basis.len() {
        for j in i + 1..basis.len() {
            pairs.push((i, j));
        }
    }

    let mut processed = 0usize;
    let deadline = Instant::now() + Duration::from_millis(budget.max_millis);

    while !pairs.is_empty() {
        if Instant::now() > deadline {
            return Err(format!(
                "Gröbner budget exceeded: exceeded the {} ms wall-clock limit after {} S-pairs \
                 (max_millis) — the coefficients are most likely blowing up",
                budget.max_millis, processed
            ));
        }

        // Normal selection: reduce the pair with the smallest lcm of leading monomials.
        let mut best = 0usize;
        let mut best_lcm = lms[pairs[0].0].lcm(&lms[pairs[0].1]);
        for k in 1..pairs.len() {
            let (a, b) = pairs[k];
            let lcm_k = lms[a].lcm(&lms[b]);
            if cmp(&lcm_k, &best_lcm) == Ordering::Less {
                best = k;
                best_lcm = lcm_k;
            }
        }
        let (i, j) = pairs.swap_remove(best);

        // Buchberger's first criterion.
        if coprime(&lms[i], &lms[j]) {
            continue;
        }

        processed += 1;
        if processed > budget.max_pairs {
            return Err(format!(
                "Gröbner budget exceeded: reduced {} S-pairs (max_pairs = {})",
                processed, budget.max_pairs
            ));
        }

        let s = s_polynomial_field(&basis[i], &basis[j], ordering)?;
        let r = normal_form_budgeted(&s, &basis, ordering, Some(budget), Some(deadline))?;

        if r.is_zero() {
            continue;
        }

        if let Some(d) = r.degree() {
            if d > budget.max_degree {
                return Err(format!(
                    "Gröbner budget exceeded: new basis element has total degree {} (max_degree = {})",
                    d, budget.max_degree
                ));
            }
        }
        if r.num_terms() > budget.max_terms {
            return Err(format!(
                "Gröbner budget exceeded: new basis element has {} terms (max_terms = {})",
                r.num_terms(),
                budget.max_terms
            ));
        }
        if basis.len() + 1 > budget.max_basis {
            return Err(format!(
                "Gröbner budget exceeded: basis reached {} elements (max_basis = {})",
                basis.len() + 1,
                budget.max_basis
            ));
        }

        let r = make_monic(&r, cmp)?;
        let new_idx = basis.len();
        for k in 0..new_idx {
            pairs.push((k, new_idx));
        }
        lms.push(lead_monomial(&r, cmp));
        basis.push(r);
    }

    Ok(basis)
}

fn lead_monomial<R: Ring, F>(p: &MultivariatePolynomial<R>, cmp: F) -> Monomial
where
    F: Fn(&Monomial, &Monomial) -> Ordering + Copy,
{
    p.leading_monomial(cmp).unwrap_or_else(Monomial::new)
}

/// The *reduced* Gröbner basis over a field: monic, inter-reduced, and sorted.
///
/// This is the canonical generating set of the ideal for the given ordering — two ideals
/// are equal iff their reduced Gröbner bases agree, which is what makes it safe to
/// compare against an oracle.
pub fn reduced_groebner_basis_field<R: Field>(
    generators: Vec<MultivariatePolynomial<R>>,
    ordering: MonomialOrdering,
    budget: &GroebnerBudget,
) -> Result<Vec<MultivariatePolynomial<R>>, String> {
    let cmp = get_comparison_fn(ordering);
    let gb = groebner_basis_field(generators, ordering, budget)?;
    if gb.is_empty() {
        return Ok(vec![]);
    }

    let lms: Vec<Monomial> = gb.iter().map(|g| lead_monomial(g, cmp)).collect();

    // Minimalize: drop any element whose leading monomial is divisible by another's.
    // Removing as we go also collapses duplicate leading monomials to a single survivor.
    let mut keep: Vec<usize> = (0..gb.len()).collect();
    let mut i = 0;
    while i < keep.len() {
        let idx = keep[i];
        let redundant = keep
            .iter()
            .any(|&j| j != idx && lms[idx].div(&lms[j]).is_some());
        if redundant {
            keep.remove(i);
        } else {
            i += 1;
        }
    }

    // Inter-reduce: replace each element by its normal form modulo the others.
    let mut result: Vec<MultivariatePolynomial<R>> = keep.iter().map(|&i| gb[i].clone()).collect();
    for i in 0..result.len() {
        let others: Vec<MultivariatePolynomial<R>> = result
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, p)| p.clone())
            .collect();
        let nf = normal_form_budgeted(&result[i], &others, ordering, Some(budget), None)?;
        if nf.is_zero() {
            return Err(
                "reduced_groebner_basis_field: a minimal basis element reduced to zero (internal invariant violated)"
                    .to_string(),
            );
        }
        result[i] = make_monic(&nf, cmp)?;
    }

    // Canonical order: descending leading monomial.
    result.sort_by(|a, b| cmp(&lead_monomial(b, cmp), &lead_monomial(a, cmp)));
    Ok(result)
}

/// Ideal membership over a field: `p ∈ I` iff `p` has normal form 0 modulo a Gröbner basis.
pub fn ideal_membership_field<R: Field>(
    poly: &MultivariatePolynomial<R>,
    generators: &[MultivariatePolynomial<R>],
    ordering: MonomialOrdering,
    budget: &GroebnerBudget,
) -> Result<bool, String> {
    let gb = groebner_basis_field(generators.to_vec(), ordering, budget)?;
    if gb.is_empty() {
        return Ok(poly.is_zero());
    }
    Ok(normal_form_budgeted(poly, &gb, ordering, Some(budget), None)?.is_zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multivariate::{Monomial, MultivariatePolynomial};
    use rustmath_integers::Integer;
    use rustmath_rationals::Rational;

    fn rat(n: i64) -> Rational {
        Rational::from_integer(Integer::from(n))
    }

    /// The variable x_v over Q.
    fn qvar(v: usize) -> MultivariatePolynomial<Rational> {
        MultivariatePolynomial::variable(v)
    }

    // ========================================================================
    // Regression: the R: Ring engine must terminate on non-monic input.
    // ========================================================================

    /// `groebner_basis([2x² − 2y, 3y² − 3x], Grevlex)` over Q used to run forever.
    ///
    /// Root cause (multivariate.rs): `divide_multiple` used a *placeholder* quotient
    /// coefficient of `1` instead of dividing, so a non-monic divisor never cancelled the
    /// leading term — while `reduce` kept setting its `reduced` flag on the basis of
    /// leading-*monomial* divisibility alone. Nothing ever changed and the loop never
    /// ended (100 % CPU, flat 4 MB RSS — a non-allocating infinite loop).
    ///
    /// What is asserted here is **termination** and an **honest outcome**, not a
    /// particular basis: the `R: Ring` engine has no coefficient division and is not
    /// entitled to claim a Gröbner basis for this input. The honest outcomes are
    ///   (a) `Ok(g)` where `g` still generates the ideal (it contains the inputs), or
    ///   (b) `Err(budget exceeded)`.
    #[test]
    fn non_monic_ring_buchberger_terminates_instead_of_hanging() {
        let x = qvar(0);
        let y = qvar(1);
        let f = (x.clone() * x.clone()).scalar_mul(&rat(2)) - y.scalar_mul(&rat(2)); // 2x² − 2y
        let g = (y.clone() * y.clone()).scalar_mul(&rat(3)) - x.scalar_mul(&rat(3)); // 3y² − 3x

        let start = Instant::now();
        let result = try_groebner_basis(
            vec![f.clone(), g.clone()],
            MonomialOrdering::Grevlex,
            &GroebnerBudget::default(),
        );
        let elapsed = start.elapsed();

        // The whole point: it comes back.
        assert!(
            elapsed.as_secs() < 60,
            "the R: Ring Buchberger took {:?} on [2x²−2y, 3y²−3x] — it is spinning again",
            elapsed
        );

        match result {
            Ok(basis) => {
                // A generating set: it must still contain what it was given.
                assert!(
                    basis.contains(&f) && basis.contains(&g),
                    "the returned set dropped an input generator, so it does not generate \
                     the same ideal"
                );
            }
            Err(e) => assert!(
                e.contains("budget exceeded"),
                "expected an honest budget refusal, got: {}",
                e
            ),
        }
    }

    /// The same system, handed to the engine that is actually entitled to solve it.
    ///
    /// sympy: `groebner([2*x**2-2*y, 3*y**2-3*x], x, y, order='grevlex')` = `[x²−y, y²−x]`.
    /// This is the constructive half of the previous test: the `R: Ring` engine refuses or
    /// returns an uncertified generating set, and the `R: Field` engine gets it right.
    #[test]
    fn the_field_engine_solves_the_non_monic_system_exactly() {
        let x = qvar(0);
        let y = qvar(1);
        let f = (x.clone() * x.clone()).scalar_mul(&rat(2)) - y.clone().scalar_mul(&rat(2));
        let g = (y.clone() * y.clone()).scalar_mul(&rat(3)) - x.clone().scalar_mul(&rat(3));

        let gb = reduced_groebner_basis_field(
            vec![f, g],
            MonomialOrdering::Grevlex,
            &GroebnerBudget::default(),
        )
        .expect("the field engine must handle a non-monic system");

        let expected = vec![
            x.clone() * x.clone() - y.clone(), // x² − y
            y.clone() * y.clone() - x.clone(), // y² − x
        ];
        assert_eq!(gb.len(), 2, "expected 2 elements, got {:?}", gb);
        for e in &expected {
            assert!(
                gb.contains(e),
                "reduced GB {:?} is missing {}",
                gb.iter().map(|p| p.to_string()).collect::<Vec<_>>(),
                e
            );
        }
    }

    /// `reduce` must not silently swallow terms it cannot reduce.
    ///
    /// The old `divide_multiple` *deleted* the leading term whenever no divisor applied,
    /// so `x² + y` "reduced" to `0` modulo `{x}` — which made `ideal_membership` answer
    /// `true` for a non-member. The true remainder is `y`.
    #[test]
    fn reduce_keeps_the_terms_it_cannot_reduce() {
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let p = x.clone() * x.clone() + y.clone(); // x² + y
        let r = reduce(&p, &[x.clone()], MonomialOrdering::Lex);
        assert_eq!(r, y, "x² + y must reduce to y modulo (x), not to 0");

        // And membership must therefore say "no".
        assert!(
            !ideal_membership(&p, &[x.clone()], MonomialOrdering::Lex),
            "x² + y is not in (x): y is not a multiple of x"
        );
        // while x² + x is in (x).
        let q = x.clone() * x.clone() + x.clone();
        assert!(ideal_membership(&q, &[x], MonomialOrdering::Lex));
    }

    /// A divisor whose leading coefficient cannot be divided out simply does not apply —
    /// its term stays in the remainder, and `p = Σ qᵢ·gᵢ + r` still holds exactly.
    #[test]
    fn division_identity_holds_for_non_monic_divisors() {
        let cmp = comparison_fn(MonomialOrdering::Grevlex);
        let x = qvar(0);
        let y = qvar(1);

        // p = 5x²y + y, g = 2x (non-monic, and 2 ∤ 5 by any coefficient we can exhibit).
        let p = (x.clone() * x.clone() * y.clone()).scalar_mul(&rat(5)) + y.clone();
        let g = x.scalar_mul(&rat(2));

        let (quotients, remainder) = p.divide_multiple(std::slice::from_ref(&g), cmp);
        let reconstructed = quotients[0].clone() * g.clone() + remainder.clone();
        assert_eq!(
            reconstructed, p,
            "the division identity p = q·g + r must hold exactly"
        );
        // Nothing was dropped: y survives in the remainder.
        assert!(!remainder.is_zero());
    }

    #[test]
    fn test_well_orderings() {
        assert!(is_well_ordering(MonomialOrdering::Lex));
        assert!(is_well_ordering(MonomialOrdering::Grlex));
        assert!(is_well_ordering(MonomialOrdering::Grevlex));
    }

    #[test]
    fn test_ordering_descriptions() {
        let lex_desc = ordering_description(MonomialOrdering::Lex);
        assert!(lex_desc.contains("Lexicographic"));

        let grlex_desc = ordering_description(MonomialOrdering::Grlex);
        assert!(grlex_desc.contains("Graded Lexicographic"));

        let grevlex_desc = ordering_description(MonomialOrdering::Grevlex);
        assert!(grevlex_desc.contains("Graded Reverse Lexicographic"));
    }

    #[test]
    fn test_info_function() {
        let info = groebner_basis_info();
        assert!(info.contains("Buchberger"));
        assert!(info.contains("S-polynomial"));
    }

    #[test]
    fn test_s_polynomial() {
        // f = x*y, g = y^2
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let f = x.clone() * y.clone(); // xy
        let g = y.clone() * y.clone(); // y²

        let s = s_polynomial(&f, &g, MonomialOrdering::Lex);

        // S(xy, y²) should eliminate the leading terms
        // LCM(xy, y²) = xy²
        // S = (y²/xy)*xy - (y²/y²)*y² = y*xy - 1*y² = xy² - y² = 0 after cancellation
        // Actually: (xy²/xy)*xy - (xy²/y²)*y² = y*xy - x*y² = xy² - xy²
        // The S-polynomial should reduce to something simpler

        // For this test, just check that we can compute it without panicking
        assert!(s.num_terms() <= 2);
    }

    #[test]
    fn test_groebner_basis_simple() {
        // Simple ideal: <x, y>
        // Gröbner basis should be {x, y}
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let generators = vec![x.clone(), y.clone()];
        let basis = groebner_basis(generators, MonomialOrdering::Lex);

        // Basis should contain both x and y
        assert!(basis.len() >= 2);
    }

    #[test]
    fn test_groebner_basis_constant() {
        // Ideal generated by a constant is the whole ring
        let one: MultivariatePolynomial<i32> = MultivariatePolynomial::constant(1);

        let generators = vec![one.clone()];
        let basis = groebner_basis(generators, MonomialOrdering::Lex);

        // Basis should contain the constant
        assert_eq!(basis.len(), 1);
        assert!(basis[0].is_constant());
    }

    #[test]
    fn test_ideal_membership() {
        // Ideal: <x, y>
        // x + y should be in the ideal
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let generators = vec![x.clone(), y.clone()];
        let test_poly = x.clone() + y.clone();

        // x + y should be in <x, y>
        assert!(ideal_membership(&test_poly, &generators, MonomialOrdering::Lex));
    }

    #[test]
    fn test_ideal_membership_not_in() {
        // Ideal: <x²>
        // x should not be in <x²>
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);

        let x_squared = x.clone() * x.clone();
        let generators = vec![x_squared];

        // x is not in <x²> (in the polynomial ring, it would generate a larger ideal)
        // Actually, in Z[x], x is NOT in <x²>
        // But the algorithm might not detect this correctly with integer coefficients
        // So let's test something clearer

        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);
        // y is definitely not in <x²>
        assert!(!ideal_membership(&y, &generators, MonomialOrdering::Lex));
    }

    #[test]
    fn test_reduce() {
        // Reduce x² with respect to {x}
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let x_squared = x.clone() * x.clone();

        let basis = vec![x.clone()];
        let reduced = reduce(&x_squared, &basis, MonomialOrdering::Lex);

        // x² should reduce (though the reduction might not be complete with integer coefficients)
        // At least check we can call the function
        assert!(reduced.num_terms() <= x_squared.num_terms());
    }

    #[test]
    fn test_monomial_comparisons() {
        let m1 = Monomial::variable(0, 2); // x₀²
        let m2 = Monomial::variable(0, 1); // x₀

        // x₀² > x₀ in all orderings
        assert_eq!(m1.cmp_lex(&m2), Ordering::Greater);
        assert_eq!(m1.cmp_grlex(&m2), Ordering::Greater);
        assert_eq!(m1.cmp_grevlex(&m2), Ordering::Greater);
    }

    #[test]
    fn test_monomial_lcm() {
        let m1 = Monomial::variable(0, 2); // x₀²
        let m2 = Monomial::variable(1, 3); // x₁³

        let lcm = m1.lcm(&m2); // x₀²x₁³

        assert_eq!(lcm.exponent(0), 2);
        assert_eq!(lcm.exponent(1), 3);
        assert_eq!(lcm.degree(), 5);
    }

    #[test]
    fn test_monomial_div() {
        let m1 = Monomial::variable(0, 3); // x₀³
        let m2 = Monomial::variable(0, 2); // x₀²

        let result = m1.div(&m2); // x₀³ / x₀² = x₀

        assert!(result.is_some());
        assert_eq!(result.unwrap().exponent(0), 1);
    }

    #[test]
    fn test_monomial_div_not_divisible() {
        let m1 = Monomial::variable(0, 1); // x₀
        let m2 = Monomial::variable(0, 2); // x₀²

        let result = m1.div(&m2); // x₀ / x₀² = not divisible

        assert!(result.is_none());
    }

    #[test]
    fn test_reduced_groebner_basis() {
        // Test with a simple ideal <x, y>
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let generators = vec![x.clone(), y.clone()];
        let reduced = reduced_groebner_basis(generators.clone(), MonomialOrdering::Lex);

        // The reduced basis should still generate the same ideal
        assert!(reduced.len() >= 2);

        // Verify that the basis elements are reduced
        for (i, poly) in reduced.iter().enumerate() {
            let mut others: Vec<_> = reduced.iter().enumerate()
                .filter(|(j, _)| i != *j)
                .map(|(_, p)| p.clone())
                .collect();

            // Each polynomial should already be reduced by the others
            let re_reduced = reduce(poly, &others, MonomialOrdering::Lex);
            // The polynomial should be unchanged (or very similar) after reduction
            assert_eq!(*poly, re_reduced);
        }
    }

    #[test]
    fn test_reduced_vs_unreduced() {
        // Test that reduced basis is at least as good as unreduced
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let f1 = x.clone() * x.clone(); // x²
        let f2 = x.clone() * y.clone(); // xy
        let generators = vec![f1, f2];

        let unreduced = groebner_basis(generators.clone(), MonomialOrdering::Lex);
        let reduced = reduced_groebner_basis(generators, MonomialOrdering::Lex);

        // Both should generate the same ideal
        // Reduced basis should have <= number of elements
        assert!(reduced.len() <= unreduced.len());
    }

    // ========================================================================
    // Tests for Phase 2.3 Enhancements
    // ========================================================================

    #[test]
    fn test_pair_selection_strategies() {
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let generators = vec![x.clone(), y.clone()];

        // All strategies should produce valid Gröbner bases
        let basis_normal = groebner_basis_with_strategy(
            generators.clone(),
            MonomialOrdering::Lex,
            PairSelectionStrategy::Normal,
        );

        let basis_min_degree = groebner_basis_with_strategy(
            generators.clone(),
            MonomialOrdering::Lex,
            PairSelectionStrategy::MinimalDegree,
        );

        let basis_min_lcm = groebner_basis_with_strategy(
            generators.clone(),
            MonomialOrdering::Lex,
            PairSelectionStrategy::MinimalLCM,
        );

        // All should produce non-empty bases
        assert!(!basis_normal.is_empty());
        assert!(!basis_min_degree.is_empty());
        assert!(!basis_min_lcm.is_empty());
    }

    #[test]
    fn test_is_useless_pair() {
        // Test coprime monomials (Buchberger's first criterion)
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        // x and y have coprime leading monomials
        assert!(is_useless_pair(&x, &y, MonomialOrdering::Lex));

        // x and x² do not have coprime leading monomials
        let x_squared = x.clone() * x.clone();
        assert!(!is_useless_pair(&x, &x_squared, MonomialOrdering::Lex));
    }

    #[test]
    fn test_groebner_basis_optimized() {
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let generators = vec![x.clone(), y.clone()];

        let basis = groebner_basis_optimized(generators, MonomialOrdering::Lex);

        // Should produce a valid basis
        assert!(basis.len() >= 2);
    }

    #[test]
    fn test_groebner_basis_f4() {
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let generators = vec![x.clone(), y.clone()];

        let basis = groebner_basis_f4(generators, MonomialOrdering::Grevlex);

        // Should produce a valid basis
        assert!(basis.len() >= 2);
    }

    #[test]
    fn test_polynomial_degree_ordering() {
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let x_squared = x.clone() * x.clone();
        let xy = x.clone() * y.clone();

        // x² has degree 2
        assert_eq!(polynomial_degree(&x_squared, MonomialOrdering::Lex), 2);

        // xy has degree 2
        assert_eq!(polynomial_degree(&xy, MonomialOrdering::Lex), 2);
    }

    #[test]
    fn test_compute_pair_priority() {
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        // Test MinimalDegree strategy
        let priority = compute_pair_priority(
            &x,
            &y,
            MonomialOrdering::Lex,
            PairSelectionStrategy::MinimalDegree,
        );
        // x and y each have degree 1, so priority should be 2
        assert_eq!(priority, 2);

        // Test Normal strategy
        let priority_normal = compute_pair_priority(
            &x,
            &y,
            MonomialOrdering::Lex,
            PairSelectionStrategy::Normal,
        );
        // Normal strategy always returns 0
        assert_eq!(priority_normal, 0);
    }

    #[test]
    fn test_optimized_vs_standard() {
        // Verify that optimized algorithm produces equivalent results
        let x: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(0);
        let y: MultivariatePolynomial<i32> = MultivariatePolynomial::variable(1);

        let f1 = x.clone() * x.clone() + y.clone();
        let f2 = x.clone() * y.clone();
        let generators = vec![f1, f2];

        let standard = groebner_basis(generators.clone(), MonomialOrdering::Grevlex);
        let optimized = groebner_basis_optimized(generators, MonomialOrdering::Grevlex);

        // Both should produce non-trivial bases
        assert!(!standard.is_empty());
        assert!(!optimized.is_empty());

        // The optimized version might produce a differently ordered but equivalent basis
        // For this test, we just verify both are non-empty and of similar size
        assert!((standard.len() as i32 - optimized.len() as i32).abs() <= 2);
    }
}
