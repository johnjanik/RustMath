I agree with the diagnosis in the spec: this is not a mathematical blocker for `belyi`; it is a crate-layout/build-boundary blocker. The current `rustmath-curves` library is all-or-nothing, so unrelated genus-(\ge 2) hyperelliptic/Jacobian errors prevent the genus-0 Belyi/conic path from compiling, even though the spec states there is no dependency edge from `belyi` to `cantor`, `jacobian`, `hyperelliptic`, or `divisor`.  

My recommendation is slightly stronger than the current “B now, A soon”:

[
\boxed{
	\textbf{Do B immediately, but design it so A is forced, visible, and testable.}
}
]

Feature-gate the genus-(\ge 2) stack now so the M23 Belyi/conic work can proceed. Then fix the 25 errors under `--features genus2` as a separate repair branch. The spec already identifies the clean dependency order:

[
\texttt{divisor}
\to
\texttt{hyperelliptic}
\to
\texttt{cantor}
\to
\texttt{jacobian}.
]

That order is correct because `jacobian` depends on all three, while `belyi` is independent. 

---

# 1. Spec improvements

## 1.1 Rename the issue from “build blockers” to “compile-boundary debt”

The blocker is not that `belyi` needs genus-(\ge 2) arithmetic. It is that `lib.rs` exports broken modules unconditionally. I would add this sentence:

```text
The immediate blocker is a compile-boundary problem, not a mathematical
dependency problem. The Belyi/conic path is genus-zero and should be buildable
without the hyperelliptic/Jacobian subsystem.
```

This clarifies why feature-gating is legitimate rather than a hack.

---

## 1.2 Split the plan into two explicit milestones

The current spec says “B now, A soon.” I would make that operational:

```text
Milestone U0 — belyi-unblock:
Feature-gate genus>=2 modules.
cargo build -p rustmath-curves passes without --features genus2.
rustmath-curves/tests/belyi_*.rs can compile and run.

Milestone U1 — genus2-restore:
cargo build -p rustmath-curves --features genus2 passes.
The 25 lib errors are fixed in divisor, hyperelliptic, cantor, jacobian.

Milestone U2 — full-tests-restore:
cargo test -p rustmath-curves --features genus2 passes, including
riemann_roch, singularities, weierstrass, parameterization,
special_divisors, and lfunction test-only repairs.
```

That prevents feature-gating from becoming silent abandonment.

---

## 1.3 Do not blanket-implement `EuclideanDomain` for every `Field` unless the trait contract really permits it

Your spec suggests a root fix:

[
\texttt{impl<F: Field> EuclideanDomain for F}
]

because every field is Euclidean. Mathematically that is true. In Rust, it is only safe if your `EuclideanDomain` trait’s required methods can be meaningfully implemented for arbitrary `Field` elements.

If `EuclideanDomain` requires polynomial-style division, degree, norm, or gcd semantics beyond “every nonzero element divides every element,” a blanket impl may accidentally lie. So I would make the local fix the default:

```rust
F: Field + EuclideanDomain
```

in `hyperelliptic`, `cantor`, and `jacobian`.

Then add a separate foundation ticket:

```text
Audit whether Field should imply EuclideanDomain in rustmath-core.
Only add the blanket impl after proving the trait contract is field-compatible.
```

---

## 1.4 Relax `contains_point` instead of propagating `NumericConversion`, if possible

The spec says `JacobianGroup` should add `NumericConversion` because `contains_point` currently requires it. 

I would first inspect the body of `contains_point`. For a hyperelliptic curve

[
y^2=f(x),
]

checking containment only requires:

[
y^2=f(x),
]

so it should need only:

```rust
F: Field + Clone + PartialEq
```

not `NumericConversion`.

If `NumericConversion` is only used for constructing `0`, `1`, or small integers, move those requirements to helper constructors, not to `contains_point`.

So I would revise RC-3:

```text
Preferred fix:
Relax HyperellipticCurve::contains_point to avoid NumericConversion if its
body only evaluates y^2 == f(x).

Fallback:
If NumericConversion is genuinely used, propagate the bound to
JacobianGroup.
```

This keeps algebraic curve arithmetic over exact symbolic fields usable.

---

# 2. `Cargo.toml` feature-gate patch

Add a feature that is **off by default**:

```toml
# rustmath-curves/Cargo.toml

[features]
default = []
genus2 = []
```

If `belyi` should always compile, keep it outside `genus2`.

---

# 3. `lib.rs` feature-gate patch

Use this structure:

```rust
// rustmath-curves/src/lib.rs

pub mod belyi;
pub mod conic;
pub mod parameterization; // only if it currently compiles in lib mode
pub mod plane;
pub mod projective;

// Genus-zero / Belyi-critical modules stay unconditional.
// Genus >= 2 arithmetic is temporarily gated.

#[cfg(feature = "genus2")]
pub mod hyperelliptic;

#[cfg(feature = "genus2")]
pub mod divisor;

#[cfg(feature = "genus2")]
pub mod cantor;

#[cfg(feature = "genus2")]
pub mod jacobian;

// Optional: if these are lib-green but test-broken, keep them unconditional.
// If any also breaks lib under some configurations, gate them too.
#[cfg(feature = "genus2")]
pub mod riemann_roch;

#[cfg(feature = "genus2")]
pub mod special_divisors;

#[cfg(feature = "genus2")]
pub mod singularities;

#[cfg(feature = "genus2")]
pub mod weierstrass;

#[cfg(feature = "genus2")]
pub mod lfunction;
```

If external users import these modules, expose a clear compile-time message:

````rust
#[cfg(not(feature = "genus2"))]
pub mod genus2_unavailable {
	//! The genus >= 2 hyperelliptic/Jacobian stack is temporarily behind
	//! the `genus2` feature while build debt is repaired.
	//!
	//! Enable with:
	//!
	//! ```text
	//! cargo build -p rustmath-curves --features genus2
	//! ```
}
````

---

# 4. Add a build-status marker

Create a file like:

```markdown
<!-- rustmath-curves/BUILD_STATUS.md -->

# rustmath-curves build status

## Current state

- Default build: intended to compile Belyi/conic/genus-zero modules.
- `--features genus2`: temporarily may fail until the hyperelliptic/Jacobian
stack is repaired.

## Debt

The following modules are feature-gated under `genus2`:

- `divisor`
- `hyperelliptic`
- `cantor`
- `jacobian`
- possibly `riemann_roch`, `singularities`, `weierstrass`,
`parameterization`, `special_divisors`, `lfunction`

## Repair order

1. `divisor`
2. `hyperelliptic`
3. `cantor`
4. `jacobian`
5. test-only modules

## Rule

Do not treat the absence of `genus2` in the default build as completion of the
genus >= 2 stack.
```

That directly implements the status-honesty warning in your spec. 

---

# 5. Fix helper: safe degree handling

The repeated `degree() -> Option<usize>` migration should not be repaired ad hoc in every file. Add a local helper.

```rust
// rustmath-curves/src/util/poly_ext.rs

use rustmath_polynomials::UnivariatePolynomial;

pub trait CurvePolyExt {
	fn degree_or_zero(&self) -> usize;
	fn is_constant_or_zero(&self) -> bool;
	fn has_positive_degree(&self) -> bool;
}

impl<F> CurvePolyExt for UnivariatePolynomial<F> {
	fn degree_or_zero(&self) -> usize {
		self.degree().unwrap_or(0)
	}
	
	fn is_constant_or_zero(&self) -> bool {
		self.degree().unwrap_or(0) == 0
	}
	
	fn has_positive_degree(&self) -> bool {
		self.degree().is_some_and(|d| d > 0)
	}
}
```

Then in `lib.rs`:

```rust
mod util;
```

and:

```rust
// rustmath-curves/src/util/mod.rs
pub mod poly_ext;
```

Use it like:

```rust
use crate::util::poly_ext::CurvePolyExt;

if self.u.has_positive_degree() {
	// ...
}

pub fn degree(&self) -> usize {
	self.u.degree_or_zero()
}
```

This fixes many `E0308` sites with one idiom.

---

# 6. Fix helper: fallible polynomial operations

If `div_rem` now returns `Result<(Q, R), MathError>`, stop destructuring it directly:

```rust
let (q, r) = u.div_rem(&d)?;
```

That means Cantor operations should return `Result<_, CurveError>`.

Add a curve-local error type:

```rust
// rustmath-curves/src/error.rs

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurveError {
	DivisionFailed,
	NonMonicDivisor,
	InvalidDivisor,
	PointNotOnCurve,
	DegreeInvariantFailed,
	PolynomialError(String),
}

impl From<rustmath_core::MathError> for CurveError {
	fn from(e: rustmath_core::MathError) -> Self {
		CurveError::PolynomialError(format!("{e:?}"))
	}
}
```

Then in `lib.rs`:

```rust
pub mod error;
```

---

# 7. Suggested `divisor.rs` repair shape

For Mumford divisors, do not let zero-polynomial degree ambiguity leak everywhere.

A stable pattern:

```rust
// rustmath-curves/src/divisor.rs

use rustmath_core::Field;
use rustmath_polynomials::UnivariatePolynomial;

use crate::util::poly_ext::CurvePolyExt;
use crate::error::CurveError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MumfordDivisor<F: Field> {
	pub u: UnivariatePolynomial<F>,
	pub v: UnivariatePolynomial<F>,
}

impl<F> MumfordDivisor<F>
where
F: Field + Clone + PartialEq,
{
	pub fn new(
	u: UnivariatePolynomial<F>,
	v: UnivariatePolynomial<F>,
	) -> Result<Self, CurveError> {
		if u.degree().is_none() {
			return Err(CurveError::InvalidDivisor);
		}
		
		// Standard Mumford representation requires deg(v) < deg(u),
		// except in special identity conventions.
		let deg_u = u.degree_or_zero();
		let deg_v = v.degree_or_zero();
		
		if !v.is_zero() && deg_v >= deg_u {
			return Err(CurveError::DegreeInvariantFailed);
		}
		
		Ok(Self { u, v })
	}
	
	pub fn degree(&self) -> usize {
		self.u.degree_or_zero()
	}
	
	pub fn is_identity(&self) -> bool {
		// Adjust to your actual identity convention.
		self.u.is_constant_or_zero() && self.v.is_zero()
	}
}
```

If your identity divisor uses (u=1,v=0), prefer:

```rust
pub fn identity() -> Self {
	Self {
		u: UnivariatePolynomial::one(),
		v: UnivariatePolynomial::zero(),
	}
}
```

The important part: `degree()` returning `usize` is now a deliberate curve-level convention, not a direct call to the polynomial API.

---

# 8. Suggested `hyperelliptic.rs` repair shape

For a curve

[
C:y^2=f(x),
]

separate constructors from point checks.

```rust
// rustmath-curves/src/hyperelliptic.rs

use rustmath_core::{EuclideanDomain, Field};
use rustmath_polynomials::UnivariatePolynomial;

use crate::error::CurveError;
use crate::util::poly_ext::CurvePolyExt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperellipticCurve<F: Field> {
	pub f: UnivariatePolynomial<F>,
	pub genus: usize,
}

impl<F> HyperellipticCurve<F>
where
F: Field + EuclideanDomain + Clone + PartialEq,
{
	pub fn new(f: UnivariatePolynomial<F>) -> Result<Self, CurveError> {
		if f.is_zero() {
			return Err(CurveError::InvalidDivisor);
		}
		
		if !f.is_square_free() {
			return Err(CurveError::PolynomialError(
			"hyperelliptic polynomial is not squarefree".to_string(),
			));
		}
		
		let deg = f.degree_or_zero();
		
		if deg < 3 {
			return Err(CurveError::PolynomialError(
			"hyperelliptic polynomial must have degree at least 3".to_string(),
			));
		}
		
		let genus = (deg - 1) / 2;
		
		Ok(Self { f, genus })
	}
}

impl<F> HyperellipticCurve<F>
where
F: Field + Clone + PartialEq,
{
	pub fn contains_point(&self, x: &F, y: &F) -> Result<bool, CurveError> {
		let fx = self
		.f
		.eval(x)
		.map_err(|e| CurveError::PolynomialError(format!("{e:?}")))?;
		
		Ok(y.clone() * y.clone() == fx)
	}
}
```

This is better than requiring `NumericConversion` on `contains_point`, unless the actual `eval` API requires it.

---

# 9. Suggested `cantor.rs` repair shape

Cantor arithmetic genuinely needs Euclidean operations. Make that explicit and return `Result`.

```rust
// rustmath-curves/src/cantor.rs

use rustmath_core::{EuclideanDomain, Field};
use rustmath_polynomials::UnivariatePolynomial;

use crate::divisor::MumfordDivisor;
use crate::error::CurveError;
use crate::hyperelliptic::HyperellipticCurve;
use crate::util::poly_ext::CurvePolyExt;

pub struct CantorAlgorithm;

impl CantorAlgorithm {
	pub fn reduce<F>(
	curve: &HyperellipticCurve<F>,
	divisor: MumfordDivisor<F>,
	) -> Result<MumfordDivisor<F>, CurveError>
	where
	F: Field + EuclideanDomain + Clone + PartialEq,
	{
		let g = curve.genus;
		let mut u = divisor.u;
		let mut v = divisor.v;
		
		while u.degree_or_zero() > g {
			// Typical reduction shape:
			// u' = (f - v^2) / u
			let v2 = v.clone() * v.clone();
			let numerator = curve.f.clone() - v2;
			
			let (q, r) = numerator.div_rem(&u)?;
			if !r.is_zero() {
				return Err(CurveError::DegreeInvariantFailed);
			}
			
			u = q.monic().map_err(|_| CurveError::NonMonicDivisor)?;
			
			// v' = -v mod u
			let (_, rem) = (-v).div_rem(&u)?;
			v = rem;
		}
		
		MumfordDivisor::new(u, v)
	}
	
	pub fn add<F>(
	curve: &HyperellipticCurve<F>,
	a: &MumfordDivisor<F>,
	b: &MumfordDivisor<F>,
	) -> Result<MumfordDivisor<F>, CurveError>
	where
	F: Field + EuclideanDomain + Clone + PartialEq,
	{
		// Placeholder structure.
		// Keep the existing formula from your code, but change every:
		//
		//     let (q, r) = p.div_rem(&d);
		//
		// to:
		//
		//     let (q, r) = p.div_rem(&d)?;
		//
		// and every degree comparison to degree_or_zero/has_positive_degree.
		let composed = Self::compose_unreduced(curve, a, b)?;
		Self::reduce(curve, composed)
	}
	
	fn compose_unreduced<F>(
	_curve: &HyperellipticCurve<F>,
	_a: &MumfordDivisor<F>,
	_b: &MumfordDivisor<F>,
	) -> Result<MumfordDivisor<F>, CurveError>
	where
	F: Field + EuclideanDomain + Clone + PartialEq,
	{
		unimplemented!("Use existing Cantor composition formula; this wrapper gives the desired error shape.")
	}
}
```

This is not a full replacement for your Cantor formulas, but it is the right API correction: Cantor arithmetic is fallible at the code level, even though mathematical invariants should make most operations succeed.

---

# 10. Suggested `jacobian.rs` bound repair

Prefer a type alias-like helper trait, because Rust stable trait aliases are limited. Use a marker trait.

```rust
// rustmath-curves/src/traits.rs

use rustmath_core::{EuclideanDomain, Field};

pub trait CurveArithmeticField:
Field + EuclideanDomain + Clone + PartialEq
{
}

impl<T> CurveArithmeticField for T
where
T: Field + EuclideanDomain + Clone + PartialEq
{
}
```

Then:

```rust
// rustmath-curves/src/jacobian.rs

use crate::traits::CurveArithmeticField;
use crate::cantor::CantorAlgorithm;
use crate::divisor::MumfordDivisor;
use crate::error::CurveError;
use crate::hyperelliptic::HyperellipticCurve;

pub struct JacobianGroup<F: CurveArithmeticField> {
	pub curve: HyperellipticCurve<F>,
}

impl<F> JacobianGroup<F>
where
F: CurveArithmeticField,
{
	pub fn new(curve: HyperellipticCurve<F>) -> Self {
		Self { curve }
	}
	
	pub fn add(
	&self,
	a: &MumfordDivisor<F>,
	b: &MumfordDivisor<F>,
	) -> Result<MumfordDivisor<F>, CurveError> {
		CantorAlgorithm::add(&self.curve, a, b)
	}
	
	pub fn double(
	&self,
	a: &MumfordDivisor<F>,
	) -> Result<MumfordDivisor<F>, CurveError> {
		self.add(a, a)
	}
	
	pub fn scalar_mul(
	&self,
	mut n: u64,
	p: &MumfordDivisor<F>,
	) -> Result<MumfordDivisor<F>, CurveError> {
		let mut acc = MumfordDivisor::identity();
		let mut base = p.clone();
		
		while n > 0 {
			if n & 1 == 1 {
				acc = self.add(&acc, &base)?;
			}
			base = self.double(&base)?;
			n >>= 1;
		}
		
		Ok(acc)
	}
	
	pub fn contains_affine_point(&self, x: &F, y: &F) -> Result<bool, CurveError> {
		self.curve.contains_point(x, y)
	}
}
```

If `contains_point` still genuinely needs `NumericConversion`, then use:

```rust
pub trait CurveArithmeticField:
Field + EuclideanDomain + Clone + PartialEq + rustmath_core::NumericConversion
{}
```

But I would try relaxing `contains_point` first.

---

# 11. Integration test that proves `belyi` is unblocked

Add an integration test that does **not** compile genus2 modules.

```rust
// rustmath-curves/tests/belyi_build_gate.rs

#[test]
fn belyi_module_is_available_without_genus2() {
	// This test should compile and run under:
	//
	// cargo test -p rustmath-curves --test belyi_build_gate
	//
	// without --features genus2.
	//
	// Replace with a real public symbol from rustmath_curves::belyi.
	let _ = std::any::type_name::<rustmath_curves::belyi::BelyiPassport>();
}
```

If you do not yet have `BelyiPassport`, add a minimal type:

```rust
// rustmath-curves/src/belyi/mod.rs

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BelyiPassport {
	pub degree: usize,
	pub cycle_types: Vec<Vec<usize>>,
}

impl BelyiPassport {
	pub fn new(degree: usize, cycle_types: Vec<Vec<usize>>) -> Self {
		Self { degree, cycle_types }
	}
}
```

Then test the ([2,12,5]) passport:

```rust
#[test]
fn passport_2_12_5_has_degree_24() {
	let p = rustmath_curves::belyi::BelyiPassport::new(
	24,
	vec![
	vec![2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1],
	vec![12,12],
	vec![5,5,5,5,1,1,1,1],
	],
	);
	
	assert_eq!(p.degree, 24);
}
```

---

# 12. CI / command script

Add a script that encodes the two tracks.

```bash
#!/usr/bin/env bash
# scripts/check_curves.sh
set -euo pipefail

echo "== default build: should unblock belyi =="
cargo build -p rustmath-curves

echo "== belyi integration tests =="
cargo test -p rustmath-curves --test belyi_build_gate

echo "== genus2 build: allowed to fail until U1 =="
if cargo build -p rustmath-curves --features genus2; then
echo "genus2 build: green"
else
echo "genus2 build: still red; tracked debt"
fi
```

Once U1 is complete, change the last block to hard-fail.

---

# 13. Mechanical fix table, improved

I would rewrite the line checklist into **patch classes**, not just file/line lists:

```text
Patch class P1 — degree Option migration:
Replace direct degree comparisons with degree_or_zero()/has_positive_degree().
Affects divisor.rs, hyperelliptic.rs, cantor.rs.

Patch class P2 — fallible div_rem migration:
Replace `let (q,r) = a.div_rem(&b);` with `let (q,r) = a.div_rem(&b)?;`.
Change enclosing functions to return Result<_, CurveError>.
Affects cantor.rs and possibly divisor.rs.

Patch class P3 — fallible eval migration:
Replace direct eval consumption with `eval(..)?` or explicit error mapping.
Affects hyperelliptic.rs.

Patch class P4 — EuclideanDomain bound:
Add `F: EuclideanDomain` only on methods/impls that call gcd/div_rem/squarefree.
Affects hyperelliptic.rs, cantor.rs, jacobian.rs.

Patch class P5 — contains_point bound:
Prefer relaxing contains_point to Field+Clone+PartialEq.
Fallback: propagate NumericConversion to JacobianGroup.
```

This makes it harder to “fix” the same root cause five different ways.

---

# 14. Important mathematical correction to preserve

The genus-(\ge 2) stack and the Belyi/conic stack should remain conceptually separate.

For the M23 ([2,12,5]) portal, the curve is genus (0). The conic decision is the relevant arithmetic payload. The currently broken hyperelliptic/Jacobian code is for later higher-genus portals and is not needed to read a genus-zero descent conic. Your spec already says the blockers are self-contained and unrelated to `belyi`; I would keep that sentence prominent. 

---

# 15. Final recommended plan

Use this exact sequence:

```text
U0.1 Add `genus2` feature.
U0.2 Gate divisor/hyperelliptic/cantor/jacobian.
U0.3 Confirm `cargo build -p rustmath-curves` is green.
U0.4 Add `tests/belyi_build_gate.rs`.
U0.5 Continue M23 Belyi/conic work.

U1.1 Add CurveError and CurvePolyExt.
U1.2 Repair divisor.rs.
U1.3 Repair hyperelliptic.rs, preferably relaxing contains_point.
U1.4 Repair cantor.rs with Result-returning Cantor operations.
U1.5 Repair jacobian.rs bounds.
U1.6 Require `cargo build -p rustmath-curves --features genus2` green.

U2.1 Repair test-only modules.
U2.2 Require `cargo test -p rustmath-curves --features genus2` green.
```

The key principle is:

[
\boxed{
	\text{Unblock genus-zero Belyi now; repair genus-(\ge2) honestly and visibly next.}
}
]

That keeps the M23 work moving without pretending the hyperelliptic/Jacobian stack is fixed.
