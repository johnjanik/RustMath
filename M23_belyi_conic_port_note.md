The biggest correction is this sentence:

[
\text{“Any conic with a }\mathbb Q\text{-point }\Rightarrow M_{23}/\mathbb Q.”
]

It should be:

[
\boxed{
	\text{Any verified }[2,12,5]\ M_{24}\text{ cover whose descent conic has a }
	\mathbb Q\text{-point, and whose point is outside the bad locus, gives the }
	M_{24}\to M_{23}\text{ portal realization.}
}
]

That distinction matters. A rational point on the conic is necessary for the portal to split, but you still need the exact cover identity, ramification audit, monodromy (M_{24}), bad-locus exclusion, and final residual (M_{23}) verification.

---

# 1. Spec corrections and improvements

## Correction 1: Separate “field of moduli” from “field of definition”

Your spec says “recognise each over its field of moduli.” That is slightly dangerous. Numerically, what you actually recognize first is a **field of coefficients in the chosen normalized coordinate**. That field is often a field of definition for the normalized model, not automatically the minimal field of moduli.

Suggested replacement:

```text
Recognize each solved normalized cover over a coefficient field L. Then compute
the Galois descent cocycle of the normalized P^1-source. The fixed field of the
Galois orbit gives the field of moduli; the obstruction class in Br(Q)[2]
determines whether the genus-0 source descends to P^1_Q or to a nonsplit conic.
```

Mathematically, the object is:

[
[g_\sigma]\in H^1(\operatorname{Gal}(L/\mathbb Q),\operatorname{PGL}_2(L)),
]

and the connecting map

[
H^1(\mathbb Q,\operatorname{PGL}_2)\longrightarrow \operatorname{Br}(\mathbb Q)[2]
]

is what produces the conic.

For a quadratic coefficient field

[
L=\mathbb Q(\sqrt{\delta}),
]

if

[
\widehat g_\sigma\sigma(\widehat g_\sigma)=\beta I,
]

then the conic class is the quaternion

[
(\delta,\beta).
]

---

## Correction 2: “HasRationalPoint = false” on the conic is a local obstruction

Your spec correctly says conics over (\mathbb Q) are complete via Hasse–Minkowski. But name the verdict carefully:

[
C(\mathbb Q)=\varnothing
]

for a conic over (\mathbb Q) means

[
C(\mathbb Q_v)=\varnothing
]

for at least one place (v). So the verdict is usually:

```rust
VerdictKind::LocallyEmpty
```

not a mysterious global obstruction. The obstruction fingerprint is the ramified set of the quaternion algebra.

For example,

[
x^2+y^2+z^2=0
]

corresponds to

[
(-1,-1),
]

ramified at

[
{2,\infty}.
]

That is a local obstruction.

---

## Correction 3: Porting Simon/Holzer should be staged

I would not make the full Simon conic minimization algorithm a P0 blocker. For this project, P0 only needs:

[
\text{diagonal ternary conic}
\longrightarrow
\text{quaternion}
\longrightarrow
\text{Hilbert symbols}
\longrightarrow
\text{rational-point verdict}.
]

Full `MinimalModel`, Holzer reduction, and parametrization can be P1/P2.

So change M1 priority into two stages:

```text
M1a P0: diagonal/symmetric conic + Hilbert-symbol rational-point decision.
M1b P1: rational point construction and parametrization.
M1c P2: Simon minimization / Holzer reduction / polished minimal models.
```

This gets the verdict engine running much faster.

---

## Correction 4: “Conic with (\mathbb Q)-point” must still pass (Z_C)

Add an explicit bad-locus gate:

[
Z_C=
{\text{branch points, cusps, singular points, degenerate specializations, monodromy-drop locus}}.
]

The construction claim should be:

[
P\in \left(X_C\setminus Z_C\right)(\mathbb Q).
]

So the verdict pipeline should be:

```text
cover exact? 
→ branch pattern exact?
→ monodromy M24?
→ descent conic split?
→ rational point outside bad locus?
→ residual degree-23 polynomial has Galois group M23?
→ CONSTRUCTED
```

---

## Correction 5: Approach 3 should be renamed

“Port MAGMA’s Belyi package” is risky wording. Since it is out-of-handbook and source availability is uncertain, call it:

```text
Approach 3 — KMSV/Sijsling–Voight Belyi algorithm reimplementation or package-port
```

Then split:

```text
3A: source-available package port
3B: clean-room reimplementation from KMSV/Sijsling–Voight papers
```

That makes the route not dependent on acquiring package source.

---

# 2. Improved module sequencing

I would change your P0/P1/P2 order slightly.

## P0: Decision skeleton and regression gates

Build these first:

[
\boxed{
	\text{Hilbert symbol}
}
]

[
\boxed{
	\text{quaternion ramification}
}
]

[
\boxed{
	\text{diagonal conic decision}
}
]

[
\boxed{
	\text{quadratic descent cocycle }(\delta,\beta)
}
]

[
\boxed{
	\text{verdict/status types}
}
]

P0 gate:

[
(-1,-1)\mapsto x^2+y^2+z^2=0,\qquad \operatorname{Ram}={2,\infty}.
]

## P1: Exact cover audit

Build:

[
A^2B-\lambda R^5S=cU^{12}
]

coefficient-wise verification, ramification-pattern verification, and permutation triple verification.

## P2: Construction reach

Build one of:

[
\text{circle packing start} \to \text{root Newton},
]

or

[
\text{bounded-frame LM} \to \text{true-root detector},
]

or

[
\text{KMSV reimplementation}.
]

## P3: Full publication gate

Build independent residual (M_{23}) verification by Galois group / monodromy tracing.

---

# 3. Rust code: status and verdict layer

This is the backbone. Every module should return this, not `bool`.

```rust
// rustmath-igp24/src/verdict.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathStatus {
	Theorem,
	Algorithm,
	CertifyingSemialgorithm,
	Conditional,
	Speculative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerdictKind {
	Constructed,
	LocallyEmpty,
	GloballyEmpty,
	Obstructed,
	Unresolved,
	InvalidInput,
}

#[derive(Debug, Clone)]
pub struct Verdict<T> {
	pub kind: VerdictKind,
	pub status: MathStatus,
	pub payload: Option<T>,
	pub notes: Vec<String>,
}

impl<T> Verdict<T> {
	pub fn constructed(payload: T, note: impl Into<String>) -> Self {
		Self {
			kind: VerdictKind::Constructed,
			status: MathStatus::CertifyingSemialgorithm,
			payload: Some(payload),
			notes: vec![note.into()],
		}
	}
	
	pub fn locally_empty(payload: T, note: impl Into<String>) -> Self {
		Self {
			kind: VerdictKind::LocallyEmpty,
			status: MathStatus::Theorem,
			payload: Some(payload),
			notes: vec![note.into()],
		}
	}
	
	pub fn unresolved(note: impl Into<String>) -> Self {
		Self {
			kind: VerdictKind::Unresolved,
			status: MathStatus::CertifyingSemialgorithm,
			payload: None,
			notes: vec![note.into()],
		}
	}
	
	pub fn invalid(note: impl Into<String>) -> Self {
		Self {
			kind: VerdictKind::InvalidInput,
			status: MathStatus::Algorithm,
			payload: None,
			notes: vec![note.into()],
		}
	}
}
```

---

# 4. Rust code: exact rational arithmetic interface

This assumes you already have a rational type. If not, this is a minimal trait boundary that lets the conic/Hilbert/quaternion modules stay independent.

```rust
// rustmath-core/src/ring_traits.rs

use std::fmt::Debug;

pub trait Field:
Clone
+ PartialEq
+ Eq
+ Debug
+ std::ops::Add<Output = Self>
+ std::ops::Sub<Output = Self>
+ std::ops::Mul<Output = Self>
+ std::ops::Neg<Output = Self>
{
	fn zero() -> Self;
	fn one() -> Self;
	fn inv(&self) -> Option<Self>;
	fn is_zero(&self) -> bool {
		*self == Self::zero()
	}
}

pub trait RationalLike: Field {
	fn from_i64(n: i64) -> Self;
	fn numerator_i128(&self) -> Option<i128>;
	fn denominator_i128(&self) -> Option<i128>;
	
	fn sign(&self) -> i8;
	fn valuation(&self, p: u64) -> i64;
	fn unit_mod(&self, p: u64, modulus: u64) -> Option<u64>;
	fn squarefree_part(&self) -> Self;
}
```

---

# 5. Rust code: Hilbert symbols over (\mathbb Q)

This is P0. It is complete for rational quaternion symbols ((a,b)).

```rust
// rustmath-quadraticforms/src/hilbert.rs

use crate::rat::Rat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Place {
	Real,
	Finite(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HilbertSymbol {
	Split,
	Ramified,
}

impl HilbertSymbol {
	pub fn as_i8(self) -> i8 {
		match self {
			HilbertSymbol::Split => 1,
			HilbertSymbol::Ramified => -1,
		}
	}
}

#[derive(Debug)]
pub enum HilbertError {
	ZeroInput,
	InvalidPrime,
	UnitReductionFailed,
}

pub fn hilbert_symbol_q(a: &Rat, b: &Rat, place: Place) -> Result<HilbertSymbol, HilbertError> {
	if a.is_zero() || b.is_zero() {
		return Err(HilbertError::ZeroInput);
	}
	
	let s = match place {
		Place::Real => {
			if a.sign() < 0 && b.sign() < 0 { -1 } else { 1 }
		}
		Place::Finite(2) => hilbert_at_2(a, b)?,
		Place::Finite(p) if p % 2 == 1 => hilbert_at_odd_prime(a, b, p)?,
		_ => return Err(HilbertError::InvalidPrime),
	};
	
	Ok(if s == 1 {
		HilbertSymbol::Split
	} else {
		HilbertSymbol::Ramified
	})
}

fn hilbert_at_odd_prime(a: &Rat, b: &Rat, p: u64) -> Result<i8, HilbertError> {
	if p < 3 || p % 2 == 0 {
		return Err(HilbertError::InvalidPrime);
	}
	
	let alpha = a.valuation(p).rem_euclid(2);
	let beta = b.valuation(p).rem_euclid(2);
	
	let u = a.unit_mod(p, p).ok_or(HilbertError::UnitReductionFailed)?;
	let v = b.unit_mod(p, p).ok_or(HilbertError::UnitReductionFailed)?;
	
	let mut out = 1i8;
	
	// (-1)^(((p-1)/2) alpha beta)
	if alpha == 1 && beta == 1 && ((p - 1) / 2) % 2 == 1 {
		out = -out;
	}
	
	if beta == 1 {
		out *= legendre_symbol(u, p)?;
	}
	
	if alpha == 1 {
		out *= legendre_symbol(v, p)?;
	}
	
	Ok(out)
}

fn hilbert_at_2(a: &Rat, b: &Rat) -> Result<i8, HilbertError> {
	let alpha = a.valuation(2).rem_euclid(2) as u64;
	let beta = b.valuation(2).rem_euclid(2) as u64;
	
	let u = a.unit_mod(2, 8).ok_or(HilbertError::UnitReductionFailed)?;
	let v = b.unit_mod(2, 8).ok_or(HilbertError::UnitReductionFailed)?;
	
	// For odd u:
	// epsilon(u) = (u-1)/2 mod 2
	// omega(u)   = (u^2-1)/8 mod 2
	let eps_u = ((u + 7) % 8 / 2) % 2;
	let eps_v = ((v + 7) % 8 / 2) % 2;
	
	let om_u = ((u * u + 63) / 8) % 2;
	let om_v = ((v * v + 63) / 8) % 2;
	
	let exponent = (eps_u * eps_v + alpha * om_v + beta * om_u) % 2;
	
	Ok(if exponent == 0 { 1 } else { -1 })
}

fn legendre_symbol(a: u64, p: u64) -> Result<i8, HilbertError> {
	if p < 3 || p % 2 == 0 {
		return Err(HilbertError::InvalidPrime);
	}
	
	let a = a % p;
	if a == 0 {
		return Ok(0);
	}
	
	let r = mod_pow(a, (p - 1) / 2, p);
	if r == 1 {
		Ok(1)
	} else if r == p - 1 {
		Ok(-1)
	} else {
		Err(HilbertError::InvalidPrime)
	}
}

fn mod_pow(mut a: u64, mut e: u64, m: u64) -> u64 {
	let mut out = 1u128;
	let mut base = (a % m) as u128;
	let modu = m as u128;
	
	while e > 0 {
		if e & 1 == 1 {
			out = (out * base) % modu;
		}
		base = (base * base) % modu;
		e >>= 1;
	}
	
	out as u64
}
```

---

# 6. Rust code: quaternion Brauer class

```rust
// rustmath-quadraticforms/src/quaternion.rs

use crate::hilbert::{hilbert_symbol_q, HilbertSymbol, Place};
use crate::rat::Rat;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuaternionQ {
	pub a: Rat,
	pub b: Rat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrauerReport {
	pub quaternion: QuaternionQ,
	pub ramified_places: Vec<Place>,
	pub split_everywhere: bool,
}

impl QuaternionQ {
	pub fn new(a: Rat, b: Rat) -> Self {
		Self { a, b }
	}
	
	pub fn candidate_places(&self) -> Vec<Place> {
		let mut primes = vec![2u64];
		
		primes.extend(self.a.support_primes());
		primes.extend(self.b.support_primes());
		
		primes.sort_unstable();
		primes.dedup();
		
		let mut out = vec![Place::Real];
		out.extend(primes.into_iter().map(Place::Finite));
		out
	}
	
	pub fn ramified_places(&self) -> Vec<Place> {
		let mut out = Vec::new();
		
		for place in self.candidate_places() {
			match hilbert_symbol_q(&self.a, &self.b, place) {
				Ok(HilbertSymbol::Ramified) => out.push(place),
				Ok(HilbertSymbol::Split) => {}
				Err(_) => {
					// In production, return Result instead of silently skipping.
				}
			}
		}
		
		out
	}
	
	pub fn report(&self) -> BrauerReport {
		let ramified = self.ramified_places();
		BrauerReport {
			quaternion: self.clone(),
			split_everywhere: ramified.is_empty(),
			ramified_places: ramified,
		}
	}
}
```

---

# 7. Rust code: diagonal conics over (\mathbb Q)

A diagonal conic

[
aX^2+bY^2+cZ^2=0
]

has associated quaternion

[
\left(-\frac a c,-\frac b c\right)
]

when (c\neq0).

```rust
// rustmath-curves/src/conic.rs

use rustmath_quadraticforms::quaternion::{BrauerReport, QuaternionQ};
use rustmath_quadraticforms::hilbert::Place;
use crate::rat::Rat;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagonalConicQ {
	pub a: Rat,
	pub b: Rat,
	pub c: Rat,
}

#[derive(Debug, Clone)]
pub struct ConicDecision {
	pub has_rational_point: bool,
	pub quaternion: QuaternionQ,
	pub ramified_places: Vec<Place>,
}

#[derive(Debug)]
pub enum ConicError {
	Degenerate,
	DivisionByZero,
}

impl DiagonalConicQ {
	pub fn new(a: Rat, b: Rat, c: Rat) -> Result<Self, ConicError> {
		if a.is_zero() || b.is_zero() || c.is_zero() {
			return Err(ConicError::Degenerate);
		}
		Ok(Self { a, b, c })
	}
	
	pub fn quaternion_class(&self) -> Result<QuaternionQ, ConicError> {
		if self.c.is_zero() {
			return Err(ConicError::DivisionByZero);
		}
		
		let q1 = -self.a.clone() / self.c.clone();
		let q2 = -self.b.clone() / self.c.clone();
		
		Ok(QuaternionQ::new(q1.squarefree_part(), q2.squarefree_part()))
	}
	
	pub fn has_rational_point(&self) -> Result<ConicDecision, ConicError> {
		let q = self.quaternion_class()?;
		let report = q.report();
		
		Ok(ConicDecision {
			has_rational_point: report.split_everywhere,
			quaternion: report.quaternion,
			ramified_places: report.ramified_places,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	
	#[test]
	fn hamilton_conic_is_anisotropic() {
		// x^2 + y^2 + z^2 = 0 -> (-1,-1), ramified at {2, infinity}.
		let c = DiagonalConicQ::new(Rat::from_i64(1), Rat::from_i64(1), Rat::from_i64(1)).unwrap();
		let d = c.has_rational_point().unwrap();
		
		assert!(!d.has_rational_point);
		assert!(d.ramified_places.contains(&Place::Real));
		assert!(d.ramified_places.contains(&Place::Finite(2)));
		assert_eq!(d.ramified_places.len(), 2);
	}
	
	#[test]
	fn split_conic_has_point() {
		// x^2 + y^2 - z^2 = 0 has [1:0:1].
		let c = DiagonalConicQ::new(Rat::from_i64(1), Rat::from_i64(1), Rat::from_i64(-1)).unwrap();
		let d = c.has_rational_point().unwrap();
		
		assert!(d.has_rational_point);
		assert!(d.ramified_places.is_empty());
	}
}
```

---

# 8. Rust code: permutation triples and genus

This is the exact group-theoretic Riemann–Hurwitz audit for the passport.

```rust
// rustmath-groups/src/permutation.rs

use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
	pub image: Vec<usize>,
}

#[derive(Debug)]
pub enum PermError {
	NotPermutation,
	DegreeMismatch,
	ProductNotIdentity,
	NotTransitive,
}

impl Permutation {
	pub fn new(image: Vec<usize>) -> Result<Self, PermError> {
		let n = image.len();
		let mut seen = vec![false; n];
		
		for &x in &image {
			if x >= n || seen[x] {
				return Err(PermError::NotPermutation);
			}
			seen[x] = true;
		}
		
		Ok(Self { image })
	}
	
	pub fn degree(&self) -> usize {
		self.image.len()
	}
	
	pub fn identity(n: usize) -> Self {
		Self {
			image: (0..n).collect(),
		}
	}
	
	pub fn compose(&self, rhs: &Self) -> Result<Self, PermError> {
		if self.degree() != rhs.degree() {
			return Err(PermError::DegreeMismatch);
		}
		
		let image = rhs.image.iter().map(|&i| self.image[i]).collect();
		Self::new(image)
	}
	
	pub fn inverse(&self) -> Self {
		let n = self.degree();
		let mut inv = vec![0; n];
		for (i, &j) in self.image.iter().enumerate() {
			inv[j] = i;
		}
		Self { image: inv }
	}
	
	pub fn is_identity(&self) -> bool {
		self.image.iter().enumerate().all(|(i, &j)| i == j)
	}
	
	pub fn cycle_count(&self) -> usize {
		let n = self.degree();
		let mut seen = vec![false; n];
		let mut count = 0;
		
		for start in 0..n {
			if seen[start] {
				continue;
			}
			
			count += 1;
			let mut x = start;
			while !seen[x] {
				seen[x] = true;
				x = self.image[x];
			}
		}
		
		count
	}
	
	pub fn defect(&self) -> usize {
		self.degree() - self.cycle_count()
	}
}

#[derive(Debug, Clone)]
pub struct BelyiTriple {
	pub sigma0: Permutation,
	pub sigma1: Permutation,
	pub sigmainf: Permutation,
}

impl BelyiTriple {
	pub fn validate(&self) -> Result<(), PermError> {
		let n = self.sigma0.degree();
		
		if self.sigma1.degree() != n || self.sigmainf.degree() != n {
			return Err(PermError::DegreeMismatch);
		}
		
		let prod = self.sigma0.compose(&self.sigma1)?.compose(&self.sigmainf)?;
		if !prod.is_identity() {
			return Err(PermError::ProductNotIdentity);
		}
		
		if !is_transitive(n, &[&self.sigma0, &self.sigma1, &self.sigmainf]) {
			return Err(PermError::NotTransitive);
		}
		
		Ok(())
	}
	
	pub fn genus(&self) -> Result<i64, PermError> {
		self.validate()?;
		
		let n = self.sigma0.degree() as i64;
		let defect =
		self.sigma0.defect() as i64 +
		self.sigma1.defect() as i64 +
		self.sigmainf.defect() as i64;
		
		// 2g - 2 = -2n + defect
		Ok(1 - n + defect / 2)
	}
}

fn is_transitive(n: usize, gens: &[&Permutation]) -> bool {
	let mut seen = vec![false; n];
	let mut q = VecDeque::new();
	
	seen[0] = true;
	q.push_back(0);
	
	while let Some(x) = q.pop_front() {
		for g in gens {
			let y = g.image[x];
			if !seen[y] {
				seen[y] = true;
				q.push_back(y);
			}
			
			let inv = g.inverse();
			let z = inv.image[x];
			if !seen[z] {
				seen[z] = true;
				q.push_back(z);
			}
		}
	}
	
	seen.into_iter().all(|b| b)
}
```

---

# 9. Rust code: 4E flag triangulation skeleton

Your spec correctly flags that a naive (2E) half-edge model is not enough for the circle-packing construction. You need flags:

[
(\text{edge},\text{side},\text{end})
]

so there are (4E) flags.

This is the right foundation for the spherical triangulation.

```rust
// rustmath-graphs/src/dessin_flags.rs

use rustmath_groups::permutation::Permutation;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlagId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeEnd {
	Black,
	White,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeSide {
	Left,
	Right,
}

#[derive(Debug, Clone)]
pub struct Flag {
	pub edge: EdgeId,
	pub side: EdgeSide,
	pub end: EdgeEnd,
}

#[derive(Debug, Clone)]
pub struct FlagTriangulation {
	pub flags: Vec<Flag>,
	
	/// Opposite end of same edge, same side.
	pub edge_flip: Vec<FlagId>,
	
	/// Opposite side of same edge, same end.
	pub side_flip: Vec<FlagId>,
	
	/// Rotation around black vertices.
	pub black_rot: Vec<FlagId>,
	
	/// Rotation around white vertices.
	pub white_rot: Vec<FlagId>,
	
	/// Rotation around face centers.
	pub face_rot: Vec<FlagId>,
}

#[derive(Debug)]
pub enum DessinBuildError {
	DegreeMismatch,
	InvalidPermutation,
}

/// Build a 4E flag model from a transitive dessin triple.
/// In a degree-n dessin, edges correspond to sheets.
pub fn build_flag_triangulation(
sigma0: &Permutation,
sigma1: &Permutation,
sigmainf: &Permutation,
) -> Result<FlagTriangulation, DessinBuildError> {
	let n = sigma0.degree();
	
	if sigma1.degree() != n || sigmainf.degree() != n {
		return Err(DessinBuildError::DegreeMismatch);
	}
	
	let mut flags = Vec::with_capacity(4 * n);
	let mut index = HashMap::new();
	
	for e in 0..n {
		for side in [EdgeSide::Left, EdgeSide::Right] {
			for end in [EdgeEnd::Black, EdgeEnd::White] {
				let id = FlagId(flags.len());
				let f = Flag {
					edge: EdgeId(e),
					side,
					end,
				};
				index.insert((e, side, end), id);
				flags.push(f);
			}
		}
	}
	
	let get = |e: usize, side: EdgeSide, end: EdgeEnd, index: &HashMap<(usize, EdgeSide, EdgeEnd), FlagId>| {
		*index.get(&(e, side, end)).unwrap()
	};
	
	let mut edge_flip = vec![FlagId(0); flags.len()];
	let mut side_flip = vec![FlagId(0); flags.len()];
	let mut black_rot = vec![FlagId(0); flags.len()];
	let mut white_rot = vec![FlagId(0); flags.len()];
	let mut face_rot = vec![FlagId(0); flags.len()];
	
	for e in 0..n {
		for side in [EdgeSide::Left, EdgeSide::Right] {
			for end in [EdgeEnd::Black, EdgeEnd::White] {
				let id = get(e, side, end, &index);
				
				let other_end = match end {
					EdgeEnd::Black => EdgeEnd::White,
					EdgeEnd::White => EdgeEnd::Black,
				};
				
				let other_side = match side {
					EdgeSide::Left => EdgeSide::Right,
					EdgeSide::Right => EdgeSide::Left,
				};
				
				edge_flip[id.0] = get(e, side, other_end, &index);
				side_flip[id.0] = get(e, other_side, end, &index);
				
				// Around black vertices: sigma0 rotates incident sheets.
				let eb = sigma0.image[e];
				black_rot[id.0] = get(eb, side, end, &index);
				
				// Around white vertices: sigma1 rotates incident sheets.
				let ew = sigma1.image[e];
				white_rot[id.0] = get(ew, side, end, &index);
				
				// Around faces: sigmainf rotates sheets. Depending on convention,
				// one may need inverse(sigmainf). This should be validated
				// against Euler characteristic and the known passport.
				let ef = sigmainf.image[e];
				face_rot[id.0] = get(ef, side, end, &index);
			}
		}
	}
	
	Ok(FlagTriangulation {
		flags,
		edge_flip,
		side_flip,
		black_rot,
		white_rot,
		face_rot,
	})
}
```

Add a validation function:

```rust
pub fn euler_characteristic_from_perms(
sigma0: &Permutation,
sigma1: &Permutation,
sigmainf: &Permutation,
) -> i64 {
	let v_black = sigma0.cycle_count() as i64;
	let v_white = sigma1.cycle_count() as i64;
	let faces = sigmainf.cycle_count() as i64;
	let edges = sigma0.degree() as i64;
	
	v_black + v_white + faces - edges
}

pub fn genus_from_euler_characteristic(chi: i64) -> i64 {
	(2 - chi) / 2
}
```

For the ([2,12,5]) passport:

[
\sigma_0:2^8 1^8,\quad
\sigma_1:12^2,\quad
\sigma_\infty:5^4 1^4.
]

Cycle counts:

[
16,\quad 2,\quad 8.
]

Thus:

[
\chi=16+2+8-24=2,
]

so

[
g=0.
]

This should be a test.

---

# 10. Rust code: factorized Belyi map representation

This is the exact representation for:

[
\phi(x)=\frac{A(x)^2B(x)}{\lambda R(x)^5S(x)}.
]

```rust
// rustmath-polynomials/src/belyi_factorized.rs

use num_complex::Complex64;

#[derive(Debug, Clone)]
pub struct PolyC {
	/// coefficients low-to-high
	pub c: Vec<Complex64>,
}

impl PolyC {
	pub fn eval(&self, x: Complex64) -> Complex64 {
		let mut acc = Complex64::new(0.0, 0.0);
		for coeff in self.c.iter().rev() {
			acc = acc * x + *coeff;
		}
		acc
	}
	
	pub fn mul(&self, rhs: &Self) -> Self {
		let mut out = vec![Complex64::new(0.0, 0.0); self.c.len() + rhs.c.len() - 1];
		for (i, a) in self.c.iter().enumerate() {
			for (j, b) in rhs.c.iter().enumerate() {
				out[i + j] += *a * *b;
			}
		}
		Self { c: out }
	}
	
	pub fn pow(&self, n: usize) -> Self {
		assert!(n >= 1);
		let mut out = self.clone();
		for _ in 1..n {
			out = out.mul(self);
		}
		out
	}
}

#[derive(Debug, Clone)]
pub struct FactorizedBelyiC {
	pub a: PolyC,
	pub b: PolyC,
	pub r: PolyC,
	pub s: PolyC,
	pub lambda: Complex64,
}

impl FactorizedBelyiC {
	pub fn numerator(&self) -> PolyC {
		self.a.pow(2).mul(&self.b)
	}
	
	pub fn denominator_without_lambda(&self) -> PolyC {
		self.r.pow(5).mul(&self.s)
	}
	
	pub fn phi(&self, x: Complex64) -> Complex64 {
		let p = self.a.eval(x).powu(2) * self.b.eval(x);
		let q = self.lambda * self.r.eval(x).powu(5) * self.s.eval(x);
		p / q
	}
	
	pub fn residual_at(&self, x: Complex64, u: &PolyC, c: Complex64) -> Complex64 {
		let lhs = self.a.eval(x).powu(2) * self.b.eval(x);
		let rhs = self.lambda * self.r.eval(x).powu(5) * self.s.eval(x)
		+ c * u.eval(x).powu(12);
		lhs - rhs
	}
}
```

---

# 11. Rust code: root/evaluation Newton residual

For numerical construction, evaluate the identity at many sample points rather than comparing ill-conditioned coefficients. This is aligned with your spec’s “root/evaluation system.”

```rust
// rustmath-numerical/src/belyi_residual.rs

use num_complex::Complex64;
use rustmath_polynomials::belyi_factorized::{FactorizedBelyiC, PolyC};

#[derive(Debug, Clone)]
pub struct EvaluationGrid {
	pub points: Vec<Complex64>,
	pub weights: Vec<f64>,
}

impl EvaluationGrid {
	pub fn roots_of_unity(n: usize, radius: f64) -> Self {
		let mut points = Vec::with_capacity(n);
		let mut weights = Vec::with_capacity(n);
		
		for k in 0..n {
			let theta = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
			points.push(Complex64::new(radius * theta.cos(), radius * theta.sin()));
			weights.push(1.0);
		}
		
		Self { points, weights }
	}
}

pub fn belyi_identity_residual(
cover: &FactorizedBelyiC,
u: &PolyC,
c: Complex64,
grid: &EvaluationGrid,
) -> Vec<f64> {
	let mut out = Vec::with_capacity(2 * grid.points.len());
	
	for (x, w) in grid.points.iter().zip(&grid.weights) {
		let z = cover.residual_at(*x, u, c);
		out.push(w * z.re);
		out.push(w * z.im);
	}
	
	out
}

pub fn residual_norm2(res: &[f64]) -> f64 {
	res.iter().map(|x| x * x).sum::<f64>().sqrt()
}
```

The next layer is automatic differentiation or finite-difference Jacobian. Start with finite differences; replace with analytic Jacobian later.

```rust
pub fn finite_difference_jacobian<F>(
x: &[f64],
f: F,
eps: f64,
) -> Vec<Vec<f64>>
where
F: Fn(&[f64]) -> Vec<f64>,
{
	let fx = f(x);
	let m = fx.len();
	let n = x.len();
	
	let mut j = vec![vec![0.0; n]; m];
	
	for col in 0..n {
		let mut xp = x.to_vec();
		xp[col] += eps;
		
		let fp = f(&xp);
		
		for row in 0..m {
			j[row][col] = (fp[row] - fx[row]) / eps;
		}
	}
	
	j
}
```

---

# 12. Rust code: Levenberg–Marquardt skeleton

This is Approach 2: cheap, bounded, multi-restart.

```rust
// rustmath-numerical/src/lm.rs

#[derive(Debug, Clone)]
pub struct LmConfig {
	pub max_iters: usize,
	pub initial_lambda: f64,
	pub lambda_up: f64,
	pub lambda_down: f64,
	pub grad_tol: f64,
	pub step_tol: f64,
	pub residual_tol: f64,
}

#[derive(Debug, Clone)]
pub struct LmResult {
	pub x: Vec<f64>,
	pub residual_norm: f64,
	pub iterations: usize,
	pub converged: bool,
}

pub fn levenberg_marquardt<F, J>(
mut x: Vec<f64>,
f: F,
jac: J,
cfg: &LmConfig,
) -> LmResult
where
F: Fn(&[f64]) -> Vec<f64>,
J: Fn(&[f64]) -> Vec<Vec<f64>>,
{
	let mut lambda = cfg.initial_lambda;
	let mut res = f(&x);
	let mut norm = norm2(&res);
	
	for iter in 0..cfg.max_iters {
		if norm < cfg.residual_tol {
			return LmResult {
				x,
				residual_norm: norm,
				iterations: iter,
				converged: true,
			};
		}
		
		let j = jac(&x);
		let jt = transpose(&j);
		let jtj = matmul(&jt, &j);
		let jtr = matvec(&jt, &res);
		
		let mut lhs = jtj;
		for i in 0..lhs.len() {
			lhs[i][i] += lambda;
		}
		
		let rhs: Vec<f64> = jtr.into_iter().map(|v| -v).collect();
		
		let Some(step) = solve_linear(lhs, rhs) else {
			lambda *= cfg.lambda_up;
			continue;
		};
		
		if norm2(&step) < cfg.step_tol {
			return LmResult {
				x,
				residual_norm: norm,
				iterations: iter,
				converged: true,
			};
		}
		
		let trial: Vec<f64> = x.iter().zip(&step).map(|(a, b)| a + b).collect();
		let trial_res = f(&trial);
		let trial_norm = norm2(&trial_res);
		
		if trial_norm < norm {
			x = trial;
			res = trial_res;
			norm = trial_norm;
			lambda *= cfg.lambda_down;
		} else {
			lambda *= cfg.lambda_up;
		}
	}
	
	LmResult {
		x,
		residual_norm: norm,
		iterations: cfg.max_iters,
		converged: false,
	}
}

fn norm2(x: &[f64]) -> f64 {
	x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
	if a.is_empty() {
		return vec![];
	}
	let m = a.len();
	let n = a[0].len();
	let mut t = vec![vec![0.0; m]; n];
	for i in 0..m {
		for j in 0..n {
			t[j][i] = a[i][j];
		}
	}
	t
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
	let m = a.len();
	let k = b.len();
	let n = b[0].len();
	
	let mut out = vec![vec![0.0; n]; m];
	
	for i in 0..m {
		for h in 0..k {
			for j in 0..n {
				out[i][j] += a[i][h] * b[h][j];
			}
		}
	}
	
	out
}

fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
	a.iter()
	.map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum())
	.collect()
}

fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
	let n = b.len();
	
	for col in 0..n {
		let mut pivot = col;
		for row in col + 1..n {
			if a[row][col].abs() > a[pivot][col].abs() {
				pivot = row;
			}
		}
		
		if a[pivot][col].abs() < 1e-14 {
			return None;
		}
		
		a.swap(col, pivot);
		b.swap(col, pivot);
		
		let diag = a[col][col];
		for j in col..n {
			a[col][j] /= diag;
		}
		b[col] /= diag;
		
		for row in 0..n {
			if row == col {
				continue;
			}
			
			let factor = a[row][col];
			for j in col..n {
				a[row][j] -= factor * a[col][j];
			}
			b[row] -= factor * b[col];
		}
	}
	
	Some(b)
}
```

This is not production numerical linear algebra, but it is a useful local implementation to unblock the bounded-frame gamble. Later replace `solve_linear` with QR/SVD.

---

# 13. Rust code: descent from quadratic coefficient field

This is the Route A descent module: (L=\mathbb Q(\sqrt{\delta})).

```rust
// rustmath-numberfields/src/quad.rs

use crate::rat::Rat;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuadField {
	pub delta: Rat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuadElem {
	pub field: QuadField,
	pub a: Rat,
	pub b: Rat,
}

impl QuadField {
	pub fn elem(&self, a: Rat, b: Rat) -> QuadElem {
		QuadElem {
			field: self.clone(),
			a,
			b,
		}
	}
	
	pub fn zero(&self) -> QuadElem {
		self.elem(Rat::zero(), Rat::zero())
	}
	
	pub fn one(&self) -> QuadElem {
		self.elem(Rat::one(), Rat::zero())
	}
}

impl QuadElem {
	pub fn conjugate(&self) -> Self {
		self.field.elem(self.a.clone(), -self.b.clone())
	}
	
	pub fn norm(&self) -> Rat {
		self.a.clone() * self.a.clone()
		- self.field.delta.clone() * self.b.clone() * self.b.clone()
	}
	
	pub fn is_rational(&self) -> bool {
		self.b.is_zero()
	}
	
	pub fn as_rational(&self) -> Option<Rat> {
		if self.is_rational() {
			Some(self.a.clone())
		} else {
			None
		}
	}
}
```

```rust
// rustmath-curves/src/descent_quad.rs

use rustmath_numberfields::quad::{QuadElem, QuadField};
use rustmath_quadraticforms::quaternion::QuaternionQ;
use crate::rat::Rat;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P1Quad {
	pub y: QuadElem,
	pub z: QuadElem,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gl2Quad {
	pub a: QuadElem,
	pub b: QuadElem,
	pub c: QuadElem,
	pub d: QuadElem,
}

impl Gl2Quad {
	pub fn conjugate(&self) -> Self {
		Self {
			a: self.a.conjugate(),
			b: self.b.conjugate(),
			c: self.c.conjugate(),
			d: self.d.conjugate(),
		}
	}
	
	pub fn mul(&self, rhs: &Self) -> Self {
		Self {
			a: self.a.clone() * rhs.a.clone() + self.b.clone() * rhs.c.clone(),
			b: self.a.clone() * rhs.b.clone() + self.b.clone() * rhs.d.clone(),
			c: self.c.clone() * rhs.a.clone() + self.d.clone() * rhs.c.clone(),
			d: self.c.clone() * rhs.b.clone() + self.d.clone() * rhs.d.clone(),
		}
	}
	
	pub fn scalar_rational(&self) -> Option<Rat> {
		if !self.b.is_zero() || !self.c.is_zero() {
			return None;
		}
		
		if self.a != self.d {
			return None;
		}
		
		self.a.as_rational()
	}
}

#[derive(Debug)]
pub enum DescentError {
	NotScalarCoboundary,
}

pub fn quaternion_from_quadratic_cocycle(
delta: Rat,
g_sigma_lift: &Gl2Quad,
) -> Result<QuaternionQ, DescentError> {
	let cob = g_sigma_lift.mul(&g_sigma_lift.conjugate());
	
	let Some(beta) = cob.scalar_rational() else {
		return Err(DescentError::NotScalarCoboundary);
	};
	
	Ok(QuaternionQ::new(delta.squarefree_part(), beta.squarefree_part()))
}
```

You also need Möbius gluing from three point-pairs. I gave a version before; for this spec, the critical certification rule is:

[
\phi^\sigma = \phi\circ g_\sigma
]

must be checked as a homogeneous binary-form identity, not merely on four points.

Add this trait:

```rust
pub trait HomogeneousBelyiCover<L> {
	fn conjugate(&self, sigma: usize) -> Self;
	fn compose_source_pgl2(&self, g: &Gl2Quad) -> Self;
	fn equals_as_binary_forms(&self, other: &Self) -> bool;
}
```

Then:

```rust
pub fn certify_descent_gluing<C>(
phi: &C,
sigma_index: usize,
g_sigma: &Gl2Quad,
) -> bool
where
C: HomogeneousBelyiCover<QuadField>,
{
	let lhs = phi.conjugate(sigma_index);
	let rhs = phi.compose_source_pgl2(g_sigma);
	lhs.equals_as_binary_forms(&rhs)
}
```

---

# 14. Rust code: hard cover-verification gate

No construction output should bypass this.

```rust
// rustmath-igp24/src/cover_gate.rs

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoverGateFailure {
	ExactIdentityFailed,
	BranchLocusNotZeroOneInfinity,
	RamificationPatternWrong,
	GenusWrong,
	MonodromyNotM24,
	MonodromyUnresolved,
	DescentConicUnreadable,
	BadLocusNotCleared,
	ResidualM23NotVerified,
}

#[derive(Debug, Clone)]
pub struct CoverGateReport {
	pub exact_identity_ok: bool,
	pub branch_locus_ok: bool,
	pub ramification_ok: bool,
	pub genus_ok: bool,
	pub monodromy_m24_ok: bool,
	pub descent_conic_ok: bool,
	pub bad_locus_ok: bool,
	pub residual_m23_ok: bool,
	pub failures: Vec<CoverGateFailure>,
}

impl CoverGateReport {
	pub fn is_constructed(&self) -> bool {
		self.failures.is_empty()
		&& self.exact_identity_ok
		&& self.branch_locus_ok
		&& self.ramification_ok
		&& self.genus_ok
		&& self.monodromy_m24_ok
		&& self.descent_conic_ok
		&& self.bad_locus_ok
		&& self.residual_m23_ok
	}
}
```

---

# 15. Suggested spec patch

I would add this directly after your section 0.

```text
## 0.1 Exact claim boundary

The computation proves only the M24→M23 point-stabilizer portal.

Given a verified [2,12,5] M24 Belyi cover φ and its M23-fixed genus-0
source X_C, a rational point of the descent conic proves that X_C ≅ P¹_Q.
It gives an M23/Q realization only after:

1. φ satisfies the exact Belyi identity;
2. the ramification pattern is exactly 2^8 1^8 · 12^2 · 5^4 1^4;
3. analytic/algebraic monodromy is M24;
4. the rational point lies outside the bad locus Z_C;
5. the residual degree-23 object has Galois group M23.

If any bounded construction search fails, output UNRESOLVED. If the conic is
anisotropic, output LOCALLY_EMPTY for this portal/component, not a global
negative result for M23/Q.
```

And add this under Approach 3:

```text
Approach 3 is not a MAGMA-handbook port. It is either:
(A) a source port of the external Belyi package, if source and license permit; or
(B) a clean-room KMSV/Sijsling–Voight implementation.
No planning item may assume the external package source is available.
```

---

# 16. Final recommendation

The spec is good, but I would make the build plan more aggressive and narrower:

```text
P0:
hilbert
quaternion
diagonal conic
verdict types
quadratic cocycle → quaternion

P1:
exact factorized Belyi identity gate
permutation/genus/ramification pattern audit
4E flag triangulation
bounded-frame LM + true-root detector

P2:
circle packing
robust root/evaluation Newton
monodromy tracker

P3:
KMSV reimplementation or package port
homotopy only if circle/KMSV stalls
```

The crucial correction is to stop treating “Belyi construction” and “M23 verdict” as one thing. They are two different logical halves. Your spec already sees that; the improved version should enforce it in types:

[
\texttt{NumericalCoverCandidate}
\to
\texttt{ExactCover}
\to
\texttt{VerifiedM24Cover}
\to
\texttt{DescentConic}
\to
\texttt{PortalVerdict}.
]

That type pipeline is the best protection against false mathematical claims.
