> **EMPIRICAL ADDENDUM (2026-06-24, tested in `inverse_galois/frobenius/sunit_batch_v2.sage`).**
> The headline §4 lever — LLL-on-log-lattice reduction of the S-unit generators — was
> implemented and tested on the production 12T34→24T2672 case (totally complex, sig (0,6)).
> **It did NOT improve the field discriminant and actually hurt:** LLL-reduced generators gave
> best `log10|D_L| = 36.2` and only 2 dim-4 hits, vs raw generators `33.4` / more hits (the
> original deep run found `~30.3`). Reason: the best dim-4 hits have **odd_disc ≈ 0** yet
> `log10|D_L| ≈ 33`, so the discriminant is **entirely 2-adic (wild) ramification** above 2 —
> which an *archimedean* log-lattice reduction cannot touch, while LLL's low-weight products
> also miss the dim-4 modules. **Conclusion: for these 2-adically-wild bases, LLL is the wrong
> disc lever.** The real levers (evidence-based): (1) **exact-disc ranking** over the dim-V hits
> (v1 banks by a gen-weight proxy, not actual disc — bank the true minimum); (2) **2-adic
> conditioning** (γ ≡ 1 mod 4 / local square at primes above 2) to kill the dominant wild term.
> The note's **GF(2) finite-support targeting** (`M_S e = v`, §3) remains valuable — but for
> *directed module-hitting*, not disc. NB: disc only matters for *contested* pairs; open (k=0)
> pairs score 1.0 regardless, so this is a secondary lever behind open-pair harvesting (105
> sunit pairs banked so far with v1).

an LLL-on-log-lattice/Buchmann unit routine for S-unit computation should be a **bounded (S)-unit squareclass engine**, not a full general-purpose class-field package at first.

The object you need is:

[
K(S,2)=
\left{
	\gamma\in K^\times/K^{\times 2}:
	v_{\mathfrak p}(\gamma)\equiv 0 \pmod 2
	\text{ for all } \mathfrak p\notin S
	\right}.
]

For block-2 relative quadratic construction,

[
L=K(\sqrt{\gamma}),
]

this is exactly the controlled squareclass space. It lets you target:

[
(V_\gamma,\ [a_\gamma],\ r(\gamma),\ \mathfrak d_{L/K})
]

with much less accidental ramification than arbitrary (\gamma=A(\theta)).

## Core design

Build this in two layers:

[
\boxed{
	\text{Layer A: Sage/PARI prototype now}
}
]

[
\boxed{
	\text{Layer B: RustMath Buchmann/log-lattice implementation later}
}
]

Do not block the IGP24 work on native Rust class groups. Use Sage/PARI to generate reduced (S)-unit squareclass pools now, then port the pieces once the hit-rate logic is validated.

---

# 1. What the routine should output

Input:

[
K=\mathbb Q(\theta),
\qquad
S={\mathfrak p_1,\dots,\mathfrak p_s},
\qquad
r_{\mathrm{target}},
\qquad
\text{optional finite parity constraints}.
]

Output a list of candidate squareclasses:

[
\gamma_1,\dots,\gamma_N\in K^\times/K^{\times 2}
]

with metadata:

```text
gamma
bits in S-unit basis
real sign vector
finite valuation parity vector on S
relative discriminant estimate
exact relative discriminant if computed
height/log-norm score
target module support vector
```

For your block-2 engine, every candidate (\gamma) should then be tested through:

[
f_\gamma(x)=N_{K/\mathbb Q}(x^2-\gamma).
]

---

# 2. (S)-unit squareclass basis

Let

[
\mathcal O_{K,S}^{\times}
]

be the (S)-unit group. Its rank is

[
r_S=r_1+r_2-1+|S|.
]

Modulo squares, ignoring roots of unity subtleties, the search space is an (\mathbb F_2)-vector space:

[
\mathcal O_{K,S}^{\times}/\mathcal O_{K,S}^{\times 2}
\cong
\mathbb F_2^{r_S+\epsilon},
]

where (\epsilon) accounts for (2)-torsion roots of unity. For your mostly totally real degree-12 bases, usually roots of unity are just ({\pm1}), and (-1) matters for signatures.

Given generators

[
g_1,\dots,g_m
]

for the (S)-unit group, enumerate

[
\gamma(e)=\prod_{i=1}^m g_i^{e_i},
\qquad
e_i\in{0,1}.
]

But do **not** enumerate all (2^m) blindly once (m) grows. First use linear constraints over (\mathbb F_2).

---

# 3. Linear constraints over (\mathbb F_2)

For each generator (g_i), compute:

### Real sign vector

For real embeddings

[
\sigma_1,\dots,\sigma_{r_1}:K\hookrightarrow\mathbb R,
]

define

[
\operatorname{sgn}_j(g_i)=
\begin{cases}
	0,& \sigma_j(g_i)>0,\
	1,& \sigma_j(g_i)<0.
\end{cases}
]

This gives a matrix

[
M_\infty\in M_{r_1\times m}(\mathbb F_2).
]

For a bit vector (e), the sign pattern is

[
M_\infty e.
]

For target row (r), you need

[
#{j:\sigma_j(\gamma)>0}=r/2
]

for quadratic relative fibers.

### Finite parity vector

For primes in (S),

[
M_S[j,i]=v_{\mathfrak p_j}(g_i)\bmod 2.
]

Then the finite support parity is:

[
M_S e.
]

If you want a target split-prime support vector (v), impose:

[
M_S e=v.
]

### Combined constraint

Solve:

[
\begin{pmatrix}
	M_S\
	M_{\mathrm{extra}}
\end{pmatrix}e
==============

\begin{pmatrix}
	v\
	b
\end{pmatrix}
]

over (\mathbb F_2), and then filter by the nonlinear condition:

[
#{\text{positive real signs}}=r/2.
]

This turns the (S)-unit search into a small affine subspace enumeration.

---

# 4. LLL-on-log-lattice reduction

The (S)-unit generators returned by Sage/PARI can be ugly. Before enumeration, reduce them.

For ordinary units, use the logarithmic embedding:

[
\lambda(u)=
\left(
\log|\sigma_1(u)|,\dots,\log|\sigma_{r_1}(u)|,
2\log|\tau_1(u)|,\dots,2\log|\tau_{r_2}(u)|
\right),
]

lying in the hyperplane

[
\sum_i \lambda_i(u)=0.
]

For (S)-units, use the extended logarithmic embedding:

[
\lambda_S(\alpha)=
\left(
\lambda_\infty(\alpha),
\ v_{\mathfrak p_1}(\alpha)\log N\mathfrak p_1,
\dots,
v_{\mathfrak p_s}(\alpha)\log N\mathfrak p_s
\right),
]

again modulo the product formula relation.

Given generators

[
g_1,\dots,g_m,
]

form the real matrix

[
L=
\left[
\lambda_S(g_1)\ \cdots\ \lambda_S(g_m)
\right].
]

Scale to an integer matrix:

[
A=\operatorname{round}(C L),
]

with

[
C=2^b
]

for (b) large enough, e.g. (b=80) or (b=120) in prototypes.

Apply LLL to the column lattice. If the LLL transformation matrix is

[
T\in GL_m(\mathbb Z),
]

replace the generators by

[
g'*i=\prod_j g_j^{T*{ij}}.
]

Then use the reduced (g'_i) for enumeration.

This does not change the generated group, but it gives smaller algebraic numbers, smaller norm/discriminant behavior, and faster minimal polynomial computation.

---

# 5. Sage/PARI prototype outline

Use this first. The exact Sage APIs can vary, so wrap everything behind your own functions.

```python
def build_S_primes(K, rational_primes):
OK = K.ring_of_integers()
S = []
for p in rational_primes:
fac = OK.ideal(p).factor()
for P, e in fac:
S.append(P)
return S
```

A wrapper for (S)-unit generators:

```python
def get_S_unit_generators(K, S):
"""
Prototype wrapper.

Try Sage's S-unit group interface if available.
Otherwise call PARI/GP's bnf/bnfsunit layer.
Return actual elements of K.
"""
try:
G, phi = K.S_unit_group(S=tuple(S))
return [phi(g) for g in G.gens()]
except Exception:
return pari_bnfsunit_wrapper(K, S)
```

Log embedding:

```python
import math
from sage.all import matrix, ZZ, RR

def log_embedding_S(K, alpha, S, prec=200):
vals = []

# Infinite logs.
for emb in K.embeddings(RealField(prec)):
vals.append(RR(abs(emb(alpha))).log())

# Finite S-valuations.
for P in S:
vals.append(ZZ(P.valuation(alpha)) * RR(P.absolute_norm()).log())

return vals
```

LLL reduction:

```python
def lll_reduce_S_units(K, gens, S, scale_bits=100):
C = ZZ(2) ** scale_bits

cols = []
for g in gens:
v = log_embedding_S(K, g, S)
cols.append([ZZ(round(C * x)) for x in v])

# Matrix with generators as rows for easier transformation tracking.
A = matrix(ZZ, cols)

# LLL returns reduced basis rows.
R = A.LLL()

# To recover the unimodular transformation, use Sage's transformation
# variant if available:
# R, T = A.LLL(transformation=True)
#
# If transformation=True is unavailable, use PARI/qflll or track via
# augmented matrix.
R, T = A.LLL(transformation=True)

new_gens = []
for i in range(T.nrows()):
u = K(1)
for j in range(T.ncols()):
e = ZZ(T[i, j])
if e != 0:
u *= gens[j] ** e
new_gens.append(u)

return new_gens, T
```

Real sign matrix:

```python
def sign_matrix(K, gens, prec=200):
embs = K.embeddings(RealField(prec))
rows = []
for emb in embs:
row = []
for g in gens:
val = emb(g)
if val > 0:
row.append(0)
elif val < 0:
row.append(1)
else:
raise ValueError("zero embedding value")
rows.append(row)
return Matrix(GF(2), rows)
```

Finite parity matrix:

```python
def finite_parity_matrix(gens, S):
rows = []
for P in S:
row = []
for g in gens:
row.append(ZZ(P.valuation(g)) % 2)
rows.append(row)
return Matrix(GF(2), rows)
```

Enumerate squareclasses satisfying finite constraints:

```python
from itertools import product

def enumerate_affine_GF2_solutions(A, b, max_solutions=None):
"""
Return e with A*e=b over GF(2).
Use right kernel basis.
"""
F = GF(2)
A = Matrix(F, A)
b = vector(F, b)

# Solve one solution.
try:
e0 = A.solve_right(b)
except ValueError:
return []

K = A.right_kernel().basis()

out = []
for bits in product([0, 1], repeat=len(K)):
e = e0
for bit, basis_vec in zip(bits, K):
if bit:
e += basis_vec
out.append(e)
if max_solutions is not None and len(out) >= max_solutions:
break
return out
```

Build (\gamma):

```python
def element_from_bits(K, gens, e):
gamma = K(1)
for bit, g in zip(e, gens):
if int(bit) == 1:
gamma *= g
return gamma
```

Quadratic signature:

```python
def quadratic_signature_from_gamma(K, gamma, prec=200):
pos = 0
neg = 0
for emb in K.embeddings(RealField(prec)):
val = emb(gamma)
if val > 0:
pos += 1
elif val < 0:
neg += 1
else:
raise ValueError("zero real embedding")
return 2 * pos
```

Search routine:

```python
def search_S_unit_squareclasses(K, S, target_r, finite_A=None, finite_b=None,
max_solutions=10000):
gens = get_S_unit_generators(K, S)
gens, T = lll_reduce_S_units(K, gens, S)

M_inf = sign_matrix(K, gens)
M_S = finite_parity_matrix(gens, S)

if finite_A is None:
A = Matrix(GF(2), 0, len(gens), [])
b = vector(GF(2), [])
else:
A = finite_A
b = finite_b

solutions = enumerate_affine_GF2_solutions(A, b, max_solutions=max_solutions)

hits = []
for e in solutions:
gamma = element_from_bits(K, gens, e)

r = quadratic_signature_from_gamma(K, gamma)
if r != target_r:
continue

score = S_unit_height_score(K, gamma, S)
hits.append((score, gamma, e))

hits.sort(key=lambda x: x[0])
return hits
```

Height/log score:

```python
def S_unit_height_score(K, gamma, S, prec=200):
total = 0.0

for emb in K.embeddings(RealField(prec)):
total += abs(float(abs(emb(gamma)).log()))

for P in S:
total += abs(int(P.valuation(gamma))) * float(RR(P.absolute_norm()).log())

return total
```

This is enough to begin generating controlled (\gamma)’s.

---

# 6. Relative discriminant scoring

For fast filtering, use the cheap odd-prime approximation.

For (\mathfrak p\nmid 2), the quadratic extension

[
K(\sqrt{\gamma})/K
]

ramifies at (\mathfrak p) if

[
v_{\mathfrak p}(\gamma)
]

is odd. So approximate:

[
N(\mathfrak d_{L/K}^{\mathrm{odd}})
===================================

\prod_{\mathfrak p\nmid 2,\ v_{\mathfrak p}(\gamma)\text{ odd}}
N\mathfrak p.
]

Then separately compute the exact relative discriminant for survivors using PARI/Sage.

```python
def odd_relative_discriminant_score(gamma, S):
score = 0.0
norm_prod = ZZ(1)

for P in S:
if P.residue_characteristic() == 2:
continue
if ZZ(P.valuation(gamma)) % 2 == 1:
norm_prod *= ZZ(P.absolute_norm())

return norm_prod
```

For exact scoring, compute the relative polynomial

[
x^2-\gamma
]

over (K), form the relative extension, and ask PARI/Sage for the relative or absolute discriminant. This may be too slow for every candidate, so do it only after log-lattice filtering.

---

# 7. Native RustMath Buchmann routine

For the Rust port, separate it into four modules.

## Module A: factor base

Choose a factor base:

[
\mathcal B={\mathfrak p:N\mathfrak p\leq B}
]

including all (S)-primes and primes above (2).

```rust
pub struct PrimeIdeal {
	pub rational_p: u64,
	pub residue_degree: u32,
	pub ramification_index: u32,
	pub norm: BigInt,
	pub id: PrimeIdealId,
}

pub struct FactorBase {
	pub primes: Vec<PrimeIdeal>,
}
```

## Module B: relation collection

Collect principal ideal relations:

[
(\alpha_i)=\prod_{\mathfrak p\in\mathcal B}\mathfrak p^{e_{i,\mathfrak p}}.
]

Each relation stores:

```rust
pub struct Relation {
	pub alpha: NumberFieldElement,
	pub valuations: Vec<i64>,
	pub log_vector: Vec<BigFloat>,
}
```

Generate (\alpha_i) using small combinations in the integral basis, random short elements, or ideals reduced by LLL.

The relation matrix is:

[
R=(e_{i,\mathfrak p}).
]

## Module C: Buchmann class/unit extraction

Use HNF/SNF on the relation matrix.

Conceptually:

* the cokernel of the relation lattice approximates the class group;
* dependencies among relations yield units;
* the logarithms of those units generate the unit lattice.

In practice, implement this as a heuristic first, with verification:

```rust
pub struct UnitData {
	pub units: Vec<NumberFieldElement>,
	pub log_lattice: Matrix<BigFloat>,
	pub regulator_estimate: BigFloat,
	pub relation_rank: usize,
}
```

For rigorous certification, compare against analytic class number formula bounds or use external PARI initially.

## Module D: log-lattice LLL

```rust
pub fn reduce_units_log_lll(
units: &[NumberFieldElement],
s_primes: &[PrimeIdeal],
scale_bits: u32,
) -> ReducedUnits {
	// 1. compute extended log vectors
	// 2. scale to integer matrix
	// 3. LLL with transformation matrix
	// 4. reconstruct new exact units
}
```

You need the transformation matrix from LLL. Without it you get shorter vectors but cannot reconstruct the corresponding exact units.

---

# 8. Rust API for the (S)-unit squareclass engine

```rust
pub struct SUnitSearchInput {
	pub field: NumberField,
	pub s_primes: Vec<PrimeIdeal>,
	pub target_r: usize,
	pub finite_constraints: Option<GF2LinearSystem>,
	pub max_candidates: usize,
	pub scale_bits: u32,
}

pub struct SUnitCandidate {
	pub gamma: NumberFieldElement,
	pub bits: Vec<u8>,
	pub real_sign_vector: Vec<i8>,
	pub finite_parity_vector: Vec<u8>,
	pub odd_disc_norm_estimate: BigInt,
	pub log_height_score: f64,
}

pub fn search_sunit_squareclasses(
input: &SUnitSearchInput,
) -> Vec<SUnitCandidate>;
```

Internal steps:

```text
1. Get unit/S-unit generators.
2. LLL-reduce extended log lattice.
3. Build finite parity matrix over GF(2).
4. Build real sign matrix over GF(2).
5. Solve finite constraints.
6. Enumerate affine solution space in increasing log-height.
7. Filter by target signature.
8. Score by odd relative discriminant and 2-adic penalty.
9. Return best candidates.
```

---

# 9. Important: (S)-units versus (K(S,2))

Strictly, (\mathcal O_{K,S}^{\times}) gives elements with valuations zero outside (S). This is ideal for low-discriminant construction.

The larger group

[
K(S,2)
]

allows valuations outside (S) to be even. Modulo squares, this is close to (S)-units but may include class group (2)-torsion effects.

For your first implementation, use (S)-units:

[
\gamma\in\mathcal O_{K,S}^{\times}/\mathcal O_{K,S}^{\times2}.
]

That is enough to control ramification and signatures.

Later, add the full Selmer-style squareclass group:

[
K(S,2).
]

That requires class group (2)-torsion / principalization logic. Useful, but not necessary for the first discriminant-quality engine.

---

# 10. How this plugs into your (V,[a]) targeting

For every candidate (\gamma), compute or infer:

[
V_\gamma,
\qquad
[a_\gamma],
\qquad
r(\gamma),
\qquad
\mathfrak d_{K(\sqrt\gamma)/K}.
]

Then classify:

```text
wrong V
right V, wrong cocycle
right V, right cocycle
exact t hit
```

The (S)-unit engine makes this experiment much cleaner because uncontrolled extra ramification is minimized. That means:

[
V_\gamma
]

is less likely to accidentally enlarge beyond the intended module.

---

# 11. What to build first

The fastest useful version is:

```text
Sage/PARI S-unit wrapper
LLL-reduce generators
GF(2) finite parity constraints
GF(2) sign constraints/filter
relative quadratic norm constructor
PARI field discriminant scoring
ghost classifier by (B,V,[a])
```

Native Rust Buchmann should wait until you know this squareclass strategy actually improves hit rate.

---

## Bottom line

You need this routine because arbitrary (\gamma=A(\theta)) has uncontrolled prime support, which causes:

[
V_\gamma\supsetneq V_{\mathrm{target}},
]

large relative discriminant, and noisy cocycle behavior.

The (S)-unit/log-lattice routine replaces that with a controlled finite search:

[
\gamma\in
\mathcal O_{K,S}^{\times}/\mathcal O_{K,S}^{\times2},
]

with:

[
\text{finite support constraints}
+
\text{signature constraints}
+
\text{LLL-reduced small generators}
+
\text{relative discriminant scoring}.
]

That is the right quality engine for the directed block-2 constructor.
