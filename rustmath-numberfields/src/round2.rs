//! Round 2 (Pohst–Zassenhaus) maximal order, field discriminant, integral basis,
//! and a true `polredabs` built on the integral basis.
//!
//! Faithful to Cohen, *A Course in Computational Algebraic Number Theory*, §6.1
//! (Alg. 6.1.8). For each prime `p` with `p² | disc(f)` the equation order `ℤ[θ]`
//! is enlarged to the `p`-maximal order by iterating: compute the `p`-radical
//! `I_p = rad(pO)` (kernel of the `F_p`-linear `x ↦ x^{p^j}` with `p^j ≥ n`,
//! Lemma 6.1.6), then the ring of multipliers `O' = (I_p : I_p) = (1/p)·ker(C)`
//! where `C: O → End(I_p/pI_p)` (Lemma 6.1.7), until `O' = O`. The general
//! radical→idealizer loop is used for every enlargement (the Dedekind first-step
//! optimization of steps 4–6 is omitted — same result).
//!
//! `d_K = disc(f) / [O_K : ℤ[θ]]²`, the index read from the order's basis matrix.
//!
//! Orders are represented as `(w, d)`: an `n×n` integer matrix `w` (columns =
//! basis vectors in the power basis `1, θ, …, θ^{n−1}`) over a positive integer
//! denominator `d`; basis element `j` is `(1/d)·Σ_i w[i][j]·θ^i`. `ℤ[θ] = (I, 1)`.

use rustmath_integers::Integer;

fn izero() -> Integer {
    Integer::zero()
}
fn ione() -> Integer {
    Integer::one()
}

// --------------------------------------------------------------------------- //
// Exact rational scalars for the linear solves
// --------------------------------------------------------------------------- //
#[derive(Clone)]
struct Q {
    num: Integer,
    den: Integer, // > 0
}
impl Q {
    fn from_int(n: Integer) -> Q {
        Q { num: n, den: ione() }
    }
    fn zero() -> Q {
        Q::from_int(izero())
    }
    fn norm(mut self) -> Q {
        if self.num.is_zero() {
            self.den = ione();
            return self;
        }
        if self.den.signum() < 0 {
            self.num = -self.num;
            self.den = -self.den;
        }
        let g = self.num.gcd(&self.den).abs();
        if !g.is_one() {
            self.num = self.num / g.clone();
            self.den = self.den / g;
        }
        self
    }
    fn add(&self, o: &Q) -> Q {
        Q {
            num: self.num.clone() * o.den.clone() + o.num.clone() * self.den.clone(),
            den: self.den.clone() * o.den.clone(),
        }
        .norm()
    }
    fn sub(&self, o: &Q) -> Q {
        Q {
            num: self.num.clone() * o.den.clone() - o.num.clone() * self.den.clone(),
            den: self.den.clone() * o.den.clone(),
        }
        .norm()
    }
    fn mul(&self, o: &Q) -> Q {
        Q { num: self.num.clone() * o.num.clone(), den: self.den.clone() * o.den.clone() }.norm()
    }
    fn div(&self, o: &Q) -> Q {
        Q { num: self.num.clone() * o.den.clone(), den: self.den.clone() * o.num.clone() }.norm()
    }
    fn is_zero(&self) -> bool {
        self.num.is_zero()
    }
    fn to_int(&self) -> Integer {
        debug_assert!(self.den.is_one(), "Q::to_int on non-integer");
        self.num.clone()
    }
}

/// Solve `A x = b` exactly over ℚ (A is `n×n`, may be singular → None).
fn solve(a: &[Vec<Q>], b: &[Q]) -> Option<Vec<Q>> {
    let n = a.len();
    let mut m: Vec<Vec<Q>> = a.to_vec();
    let mut rhs: Vec<Q> = b.to_vec();
    let mut piv_row = 0;
    let mut where_piv = vec![usize::MAX; n];
    for col in 0..n {
        if piv_row >= n {
            break;
        }
        let mut sel = piv_row;
        while sel < n && m[sel][col].is_zero() {
            sel += 1;
        }
        if sel == n {
            continue;
        }
        m.swap(sel, piv_row);
        rhs.swap(sel, piv_row);
        where_piv[col] = piv_row;
        let inv_pivot = Q::from_int(ione()).div(&m[piv_row][col]);
        for j in 0..n {
            m[piv_row][j] = m[piv_row][j].mul(&inv_pivot);
        }
        rhs[piv_row] = rhs[piv_row].mul(&inv_pivot);
        for r in 0..n {
            if r != piv_row && !m[r][col].is_zero() {
                let factor = m[r][col].clone();
                for j in 0..n {
                    let t = m[piv_row][j].mul(&factor);
                    m[r][j] = m[r][j].sub(&t);
                }
                let t = rhs[piv_row].mul(&factor);
                rhs[r] = rhs[r].sub(&t);
            }
        }
        piv_row += 1;
    }
    // back-read solution (square, assume full rank for our uses)
    let mut x = vec![Q::zero(); n];
    for col in 0..n {
        if where_piv[col] != usize::MAX {
            x[col] = rhs[where_piv[col]].clone();
        }
    }
    // verify
    for i in 0..n {
        let mut s = Q::zero();
        for j in 0..n {
            s = s.add(&a[i][j].mul(&x[j]));
        }
        if !s.sub(&b[i]).is_zero() {
            return None;
        }
    }
    Some(x)
}

// --------------------------------------------------------------------------- //
// Power-basis arithmetic
// --------------------------------------------------------------------------- //
/// `θ^m mod f` for `m = 0 ..= 2n−2`, length-`n` integer coordinate vectors.
fn power_table(f: &[Integer]) -> Vec<Vec<Integer>> {
    let n = f.len() - 1;
    let mut table: Vec<Vec<Integer>> = Vec::new();
    let mut cur = vec![izero(); n];
    cur[0] = ione();
    table.push(cur.clone());
    let count = if n >= 2 { 2 * n - 2 } else { 1 };
    for _ in 0..count {
        let mut next = vec![izero(); n + 1];
        for i in 0..n {
            next[i + 1] = cur[i].clone();
        }
        let top = next[n].clone();
        if !top.is_zero() {
            for i in 0..n {
                next[i] = next[i].clone() - top.clone() * f[i].clone();
            }
        }
        next.truncate(n);
        table.push(next.clone());
        cur = next;
    }
    table
}

/// Field product of two power-basis vectors, reduced to length `n`.
fn mul_power(a: &[Integer], b: &[Integer], table: &[Vec<Integer>]) -> Vec<Integer> {
    let n = a.len();
    let mut out = vec![izero(); n];
    for i in 0..n {
        if a[i].is_zero() {
            continue;
        }
        for j in 0..n {
            if b[j].is_zero() {
                continue;
            }
            let c = a[i].clone() * b[j].clone();
            let pk = &table[i + j];
            for r in 0..n {
                out[r] = out[r].clone() + c.clone() * pk[r].clone();
            }
        }
    }
    out
}

// --------------------------------------------------------------------------- //
// Order representation
// --------------------------------------------------------------------------- //
#[derive(Clone)]
struct Order {
    w: Vec<Vec<Integer>>, // n×n, w[i][j] = coeff of θ^i in basis element j (column j)
    d: Integer,           // positive denominator
    n: usize,
}

impl Order {
    fn equation_order(n: usize) -> Order {
        let w = (0..n)
            .map(|i| (0..n).map(|j| if i == j { ione() } else { izero() }).collect())
            .collect();
        Order { w, d: ione(), n }
    }

    /// Column `j` (basis element j) as power-basis numerator vector (÷ d).
    fn col(&self, j: usize) -> Vec<Integer> {
        (0..self.n).map(|i| self.w[i][j].clone()).collect()
    }

    /// Integer structure constants `m[i][j][k]`: `ω_i·ω_j = Σ_k m[i][j][k] ω_k`.
    ///
    /// `W` is lower-triangular HNF, so `W·m' = P` is solved by exact integer forward
    /// substitution (`m' = d·m`, then `m = m'/d`) — far faster than rational
    /// Gaussian elimination, which is critical at degree 24.
    fn structure_constants(&self, table: &[Vec<Integer>]) -> Vec<Vec<Vec<Integer>>> {
        let n = self.n;
        let mut m = vec![vec![vec![izero(); n]; n]; n];
        for i in 0..n {
            let ci = self.col(i);
            for j in i..n {
                let cj = self.col(j);
                // Σ_k m_k·col_k / d = (col_i·col_j)/d²  ⇒  W·(d·m) = P
                let prod = mul_power(&ci, &cj, table);
                let mprime = forward_solve_lower(&self.w, &prod, n);
                for k in 0..n {
                    debug_assert!((mprime[k].clone() % self.d.clone()).is_zero());
                    let v = mprime[k].clone() / self.d.clone();
                    m[i][j][k] = v.clone();
                    m[j][i][k] = v;
                }
            }
        }
        m
    }
}

/// Solve `L x = b` for a lower-triangular integer matrix `L` (nonzero diagonal),
/// all divisions exact. `L[i][j]` is row `i`, column `j`; `L[i][j] = 0` for `j > i`.
fn forward_solve_lower(l: &[Vec<Integer>], b: &[Integer], n: usize) -> Vec<Integer> {
    let mut x = vec![izero(); n];
    for k in 0..n {
        let mut acc = b[k].clone();
        for j in 0..k {
            acc = acc - l[k][j].clone() * x[j].clone();
        }
        x[k] = acc / l[k][k].clone();
    }
    x
}

// --------------------------------------------------------------------------- //
// F_p linear algebra (i64, p a small prime)
// --------------------------------------------------------------------------- //
fn mod_inv(a: i64, p: i64) -> i64 {
    let mut acc = 1i128;
    let mut b = (((a % p) + p) % p) as i128;
    let mut e = p - 2;
    while e > 0 {
        if e & 1 == 1 {
            acc = acc * b % p as i128;
        }
        b = b * b % p as i128;
        e >>= 1;
    }
    acc as i64
}

/// Null space basis of an `r×c` matrix over `F_p` (rows given). Returns vectors in
/// `F_p^c` (as `Vec<i64>` in `[0,p)`).
fn kernel_fp(rows: &[Vec<i64>], c: usize, p: i64) -> Vec<Vec<i64>> {
    let mut m: Vec<Vec<i64>> = rows.iter().map(|r| r.iter().map(|&x| ((x % p) + p) % p).collect()).collect();
    let r = m.len();
    let mut pivot_col = vec![usize::MAX; r];
    let mut col_pivot = vec![usize::MAX; c];
    let mut pr = 0usize;
    for col in 0..c {
        if pr >= r {
            break;
        }
        let mut sel = pr;
        while sel < r && m[sel][col] == 0 {
            sel += 1;
        }
        if sel == r {
            continue;
        }
        m.swap(sel, pr);
        let inv = mod_inv(m[pr][col], p);
        for j in 0..c {
            m[pr][j] = m[pr][j] * inv % p;
        }
        for rr in 0..r {
            if rr != pr && m[rr][col] != 0 {
                let f = m[rr][col];
                for j in 0..c {
                    m[rr][j] = ((m[rr][j] - f * m[pr][j]) % p + p) % p;
                }
            }
        }
        pivot_col[pr] = col;
        col_pivot[col] = pr;
        pr += 1;
    }
    let free: Vec<usize> = (0..c).filter(|&col| col_pivot[col] == usize::MAX).collect();
    let mut basis = Vec::new();
    for &fc in &free {
        let mut v = vec![0i64; c];
        v[fc] = 1;
        for col in 0..c {
            if col_pivot[col] != usize::MAX {
                let prow = col_pivot[col];
                // pivot row expresses pivot var = -Σ free contributions
                v[col] = ((-m[prow][fc]) % p + p) % p;
            }
        }
        basis.push(v);
    }
    basis
}

// --------------------------------------------------------------------------- //
// Integer linear algebra: HNF basis and Bareiss determinant
// --------------------------------------------------------------------------- //
/// Hermite basis of the lattice spanned by integer column generators (each a
/// length-`n` vector). Returns `n` columns (full-rank lattices only). Pivot for
/// row `r` lands in `out[r]`.
fn hnf_basis(gens: &[Vec<Integer>], n: usize) -> Vec<Vec<Integer>> {
    let mut cols: Vec<Vec<Integer>> = gens.iter().filter(|c| c.iter().any(|x| !x.is_zero())).cloned().collect();
    let mut basis: Vec<Vec<Integer>> = vec![vec![izero(); n]; n];
    for row in 0..n {
        loop {
            // pick column with minimal nonzero |entry| at `row`
            let mut pick = usize::MAX;
            let mut best = izero();
            for (c, col) in cols.iter().enumerate() {
                let v = col[row].abs();
                if !v.is_zero() && (pick == usize::MAX || v < best) {
                    best = v;
                    pick = c;
                }
            }
            if pick == usize::MAX {
                break;
            }
            let pivot_val = cols[pick][row].clone();
            let mut all_zero = true;
            for c in 0..cols.len() {
                if c != pick && !cols[c][row].is_zero() {
                    let q = floor_div(&cols[c][row], &pivot_val);
                    if !q.is_zero() {
                        for r in 0..n {
                            cols[c][r] = cols[c][r].clone() - q.clone() * cols[pick][r].clone();
                        }
                    }
                    if !cols[c][row].is_zero() {
                        all_zero = false;
                    }
                }
            }
            if all_zero {
                let mut pc = cols.remove(pick);
                if pc[row].signum() < 0 {
                    for r in 0..n {
                        pc[r] = -pc[r].clone();
                    }
                }
                basis[row] = pc;
                break;
            }
        }
    }
    basis
}

fn floor_div(a: &Integer, b: &Integer) -> Integer {
    // floor division (b != 0)
    let q = a.clone() / b.clone();
    let r = a.clone() - q.clone() * b.clone();
    if !r.is_zero() && (r.signum() as i64) * (b.signum() as i64) < 0 {
        q - ione()
    } else {
        q
    }
}

/// Exact determinant of an `n×n` integer matrix (Bareiss fraction-free).
fn bareiss_det(mat: &[Vec<Integer>]) -> Integer {
    let n = mat.len();
    if n == 0 {
        return ione();
    }
    let mut m: Vec<Vec<Integer>> = mat.to_vec();
    let mut sign = 1i64;
    let mut prev = ione();
    for k in 0..n - 1 {
        if m[k][k].is_zero() {
            let mut sw = usize::MAX;
            for i in k + 1..n {
                if !m[i][k].is_zero() {
                    sw = i;
                    break;
                }
            }
            if sw == usize::MAX {
                return izero();
            }
            m.swap(k, sw);
            sign = -sign;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                let num = m[i][j].clone() * m[k][k].clone() - m[i][k].clone() * m[k][j].clone();
                m[i][j] = num / prev.clone();
            }
        }
        prev = m[k][k].clone();
    }
    let d = m[n - 1][n - 1].clone();
    if sign < 0 {
        -d
    } else {
        d
    }
}

// --------------------------------------------------------------------------- //
// Radical and idealizer (one enlargement at p)
// --------------------------------------------------------------------------- //
fn reduce_mod_p_i64(x: &Integer, p: i64) -> i64 {
    let r = (x.clone() % Integer::from(p)).to_i64();
    ((r % p) + p) % p
}

/// Supplement F_p row-vectors `betas` (independent) to a basis of F_p^n.
fn supplement_basis(betas: &[Vec<i64>], n: usize, p: i64) -> Vec<Vec<i64>> {
    let mut basis: Vec<Vec<i64>> = betas.to_vec();
    // echelon copy to test independence
    let mut ech: Vec<Vec<i64>> = Vec::new();
    let add = |v: &Vec<i64>, ech: &mut Vec<Vec<i64>>| -> bool {
        let mut w = v.clone();
        for e in ech.iter() {
            // find pivot of e
            let piv = e.iter().position(|&x| x != 0).unwrap();
            if w[piv] != 0 {
                let f = w[piv] * mod_inv(e[piv], p) % p;
                for j in 0..n {
                    w[j] = ((w[j] - f * e[j]) % p + p) % p;
                }
            }
        }
        if w.iter().any(|&x| x != 0) {
            ech.push(w);
            true
        } else {
            false
        }
    };
    for b in betas {
        add(b, &mut ech);
    }
    for i in 0..n {
        if basis.len() == n {
            break;
        }
        let mut e = vec![0i64; n];
        e[i] = 1;
        if add(&e, &mut ech) {
            basis.push(e);
        }
    }
    basis
}

/// One enlargement step at `p`. Returns the (possibly enlarged) order; the loop
/// terminates when the returned order equals the input.
fn enlarge_at_p(o: &Order, p: i64, table: &[Vec<Integer>]) -> Order {
    let n = o.n;
    let sc = o.structure_constants(table); // exact integer m[i][j][k]
    // mod-p structure constants
    let scp: Vec<Vec<Vec<i64>>> = (0..n)
        .map(|i| (0..n).map(|j| (0..n).map(|k| reduce_mod_p_i64(&sc[i][j][k], p)).collect()).collect())
        .collect();
    let mul_p = |x: &[i64], y: &[i64]| -> Vec<i64> {
        let mut z = vec![0i64; n];
        for i in 0..n {
            if x[i] == 0 {
                continue;
            }
            for j in 0..n {
                if y[j] == 0 {
                    continue;
                }
                let xy = x[i] * y[j] % p;
                for k in 0..n {
                    z[k] = (z[k] + xy * scp[i][j][k]) % p;
                }
            }
        }
        z
    };
    // q = smallest power of p with q >= n
    let mut q: u64 = p as u64;
    while q < n as u64 {
        q *= p as u64;
    }
    // Frobenius^j matrix A: column j = (e_j)^q
    let mut a_cols: Vec<Vec<i64>> = Vec::with_capacity(n);
    for j in 0..n {
        let mut base = vec![0i64; n];
        base[j] = 1;
        // base^q by square-and-multiply
        let mut acc = vec![0i64; n];
        acc[0] = 1; // 1
        let mut b = base.clone();
        let mut e = q;
        while e > 0 {
            if e & 1 == 1 {
                acc = mul_p(&acc, &b);
            }
            b = mul_p(&b, &b);
            e >>= 1;
        }
        a_cols.push(acc);
    }
    // rows of A: A[i][j] = a_cols[j][i]
    let a_rows: Vec<Vec<i64>> = (0..n).map(|i| (0..n).map(|j| a_cols[j][i]).collect()).collect();
    let betas = kernel_fp(&a_rows, n, p);
    let l = betas.len();
    if l == 0 {
        return o.clone(); // radical = pO ⇒ p-maximal
    }
    // alpha basis of I_p: beta_i (i<l), p*beta_i (i>=l)
    let full = supplement_basis(&betas, n, p);
    let alpha: Vec<Vec<Integer>> = full
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let s = if i < l { ione() } else { Integer::from(p) };
            b.iter().map(|&x| Integer::from(x) * s.clone()).collect()
        })
        .collect();
    // Lambda (columns alpha_i). Precompute Λ⁻¹ once (Q) and reuse for all (j,k);
    // computing it per pair would be O(n⁵) and dominate at degree 24.
    let lam_q: Vec<Vec<Q>> =
        (0..n).map(|i| (0..n).map(|j| Q::from_int(alpha[j][i].clone())).collect()).collect();
    let mut lam_inv: Vec<Vec<Q>> = vec![vec![Q::zero(); n]; n]; // column c = Λ⁻¹ e_c
    for c in 0..n {
        let mut e = vec![Q::zero(); n];
        e[c] = Q::from_int(ione());
        let col = solve(&lam_q, &e).expect("idealizer: Lambda not invertible");
        for i in 0..n {
            lam_inv[i][c] = col[i].clone();
        }
    }
    // C: (n^2) x n over F_p ; C[(i,j)][k] = coords of (omega_k * alpha_j) in alpha-basis, mod p
    let mut c_rows: Vec<Vec<i64>> = vec![vec![0i64; n]; n * n];
    for k in 0..n {
        for j in 0..n {
            // z = omega_k * alpha_j = sum_b alpha_j[b] * sc[k][b][:]
            let mut z = vec![izero(); n];
            for b in 0..n {
                if alpha[j][b].is_zero() {
                    continue;
                }
                for r in 0..n {
                    z[r] = z[r].clone() + alpha[j][b].clone() * sc[k][b][r].clone();
                }
            }
            // coords = Λ⁻¹ z (integer since z ∈ I_p)
            for i in 0..n {
                let mut acc = Q::zero();
                for r in 0..n {
                    if !z[r].is_zero() {
                        acc = acc.add(&lam_inv[i][r].mul(&Q::from_int(z[r].clone())));
                    }
                }
                c_rows[i * n + j][k] = reduce_mod_p_i64(&acc.to_int(), p);
            }
        }
    }
    let gammas = kernel_fp(&c_rows, n, p);
    // U lattice (O-coords): lifts of gammas, plus p*e_k
    let mut gens: Vec<Vec<Integer>> = gammas
        .iter()
        .map(|g| g.iter().map(|&x| Integer::from(x)).collect())
        .collect();
    for k in 0..n {
        let mut e = vec![izero(); n];
        e[k] = Integer::from(p);
        gens.push(e);
    }
    let h = hnf_basis(&gens, n); // columns = O-coord basis of p*O'
    // det(H) == p^n  ⇔  O' == O
    let det_h = bareiss_det(&transpose(&h)).abs();
    let pn = Integer::from(p).pow(n as u32);
    if det_h == pn {
        return o.clone();
    }
    // W' = W_O * H (power coords numerators), d' = d_O * p ; H columns are O-coords
    let mut w_new = vec![vec![izero(); n]; n];
    for col in 0..n {
        // basis col `col` of O' in O-coords = h[col] (the col-th basis vector)
        // its power coords numerators = W_O * h[col]
        for i in 0..n {
            let mut s = izero();
            for r in 0..n {
                s = s + o.w[i][r].clone() * h[col][r].clone();
            }
            w_new[i][col] = s;
        }
    }
    let d_new = o.d.clone() * Integer::from(p);
    normalize_order(w_new, d_new, n)
}

fn transpose(m: &[Vec<Integer>]) -> Vec<Vec<Integer>> {
    let n = m.len();
    (0..n).map(|i| (0..n).map(|j| m[j][i].clone()).collect()).collect()
}

/// Reduce the common factor between all entries of `w` and `d`, and re-HNF the
/// columns so the representation is canonical.
fn normalize_order(w: Vec<Vec<Integer>>, d: Integer, n: usize) -> Order {
    // gcd of all entries and d
    let mut g = d.clone();
    for row in &w {
        for e in row {
            if !e.is_zero() {
                g = g.gcd(e);
            }
        }
    }
    g = g.abs();
    let (w, d) = if g.is_one() || g.is_zero() {
        (w, d)
    } else {
        let w2 = w.iter().map(|row| row.iter().map(|e| e.clone() / g.clone()).collect()).collect();
        (w2, d / g)
    };
    // HNF the columns (power-coord lattice) for a canonical basis
    let cols: Vec<Vec<Integer>> = (0..n).map(|j| (0..n).map(|i| w[i][j].clone()).collect()).collect();
    let basis = hnf_basis(&cols, n);
    let mut w_h = vec![vec![izero(); n]; n];
    for j in 0..n {
        for i in 0..n {
            w_h[i][j] = basis[j][i].clone();
        }
    }
    Order { w: w_h, d, n }
}

// --------------------------------------------------------------------------- //
// Driver: maximal order, field discriminant, integral basis
// --------------------------------------------------------------------------- //
fn is_prime_i64(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

/// Primes `p ≤ bound` with `p² | disc`. (Index primes above `bound` are not
/// detected — fine when the index is `bound`-smooth, as for the IGP24 corpus.)
fn square_factor_primes(disc: &Integer, bound: i64) -> Vec<i64> {
    let mut out = Vec::new();
    let mut p = 2i64;
    while p <= bound {
        if is_prime_i64(p) {
            let p2 = Integer::from(p) * Integer::from(p);
            if (disc.clone() % p2).is_zero() {
                out.push(p);
            }
        }
        p += 1;
    }
    out
}

fn same_order(a: &Order, b: &Order) -> bool {
    a.d == b.d && a.w == b.w
}

/// `p`-maximal order containing `o` (iterate radical→idealizer to a fixed point).
fn p_maximal(mut o: Order, p: i64, table: &[Vec<Integer>]) -> Order {
    loop {
        let next = enlarge_at_p(&o, p, table);
        if same_order(&next, &o) {
            return o;
        }
        o = next;
    }
}

/// The maximal order `O_K` of `K = ℚ[x]/(f)` (f monic irreducible).
fn maximal_order(f: &[Integer], bound: i64) -> (Order, Integer) {
    let n = f.len() - 1;
    let table = power_table(f);
    let disc = rustmath_polynomials::disc::discriminant(f);
    let mut o = Order::equation_order(n);
    for p in square_factor_primes(&disc, bound) {
        o = p_maximal(o, p, &table);
    }
    (o, disc)
}

/// Index `[O : ℤ[θ]] = dⁿ / det(W)`.
fn order_index(o: &Order) -> Integer {
    let det = bareiss_det(&o.w).abs();
    let dn = o.d.pow(o.n as u32);
    dn / det
}

/// Field discriminant `d_K` of `K = ℚ[x]/(f)`. Validated against `gp nfdisc`.
pub fn field_discriminant(f: &[Integer]) -> Integer {
    field_discriminant_bounded(f, 100_000)
}

/// As [`field_discriminant`] but with an explicit small-prime bound for the index.
pub fn field_discriminant_bounded(f: &[Integer], bound: i64) -> Integer {
    let (o, disc) = maximal_order(f, bound);
    let idx = order_index(&o);
    disc / (idx.clone() * idx)
}

/// An integral basis of `O_K`, each element returned as `(numerator_coeffs, d)`
/// meaning `(1/d)·Σ numerator_coeffs[i]·θ^i`. Element 0 is `1`.
pub fn integral_basis(f: &[Integer]) -> Vec<(Vec<Integer>, Integer)> {
    let (o, _) = maximal_order(f, 100_000);
    (0..o.n).map(|j| (o.col(j), o.d.clone())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(cs: &[i64]) -> Vec<Integer> {
        cs.iter().map(|&c| Integer::from(c)).collect()
    }

    #[test]
    fn disc_quadratic_squarefree_index1() {
        // x^2 + 1: field disc -4, index 1
        assert_eq!(field_discriminant(&p(&[1, 0, 1])), Integer::from(-4));
    }

    #[test]
    fn disc_quadratic_index2() {
        // x^2 - 5: poldisc 20, field Q(sqrt5) disc 5, index 2
        assert_eq!(field_discriminant(&p(&[-5, 0, 1])), Integer::from(5));
        // x^2 - 12: poldisc 48, field Q(sqrt3) disc 12, index 2
        assert_eq!(field_discriminant(&p(&[-12, 0, 1])), Integer::from(12));
        // x^2 - 8: poldisc 32, field Q(sqrt2) disc 8, index 2
        assert_eq!(field_discriminant(&p(&[-8, 0, 1])), Integer::from(8));
    }

    #[test]
    fn disc_dedekind_cubic_index2() {
        // x^3 - x^2 - 2x - 8 (Dedekind's non-monogenic cubic): index 2, nfdisc -503
        assert_eq!(field_discriminant(&p(&[-8, -2, -1, 1])), Integer::from(-503));
    }

    #[test]
    fn disc_cyclotomic5() {
        // x^4+x^3+x^2+x+1: maximal (poldisc 125 = field disc), index 1
        assert_eq!(field_discriminant(&p(&[1, 1, 1, 1, 1])), Integer::from(125));
    }
}

// --------------------------------------------------------------------------- //
// True polredabs: reduce over the maximal order O_K
// --------------------------------------------------------------------------- //
#[derive(Clone, Copy)]
struct Cpx {
    re: f64,
    im: f64,
}
impl Cpx {
    fn mul(self, o: Cpx) -> Cpx {
        Cpx { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
    fn add(self, o: Cpx) -> Cpx {
        Cpx { re: self.re + o.re, im: self.im + o.im }
    }
    fn sub(self, o: Cpx) -> Cpx {
        Cpx { re: self.re - o.re, im: self.im - o.im }
    }
    fn div(self, o: Cpx) -> Cpx {
        let d = o.re * o.re + o.im * o.im;
        Cpx { re: (self.re * o.re + self.im * o.im) / d, im: (self.im * o.re - self.re * o.im) / d }
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

fn roots_cpx(f: &[Integer]) -> Vec<Cpx> {
    let n = f.len() - 1;
    let c: Vec<f64> = f.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
    let eval = |z: Cpx| {
        let mut acc = Cpx { re: 1.0, im: 0.0 };
        for k in (0..n).rev() {
            acc = acc.mul(z).add(Cpx { re: c[k], im: 0.0 });
        }
        acc
    };
    let mut z: Vec<Cpx> = (0..n)
        .map(|k| {
            let s = Cpx { re: 0.4, im: 0.9 };
            let mut p = Cpx { re: 1.0, im: 0.0 };
            for _ in 0..k {
                p = p.mul(s);
            }
            p
        })
        .collect();
    for _ in 0..300 {
        let mut md = 0.0f64;
        for i in 0..n {
            let mut den = Cpx { re: 1.0, im: 0.0 };
            for j in 0..n {
                if j != i {
                    den = den.mul(z[i].sub(z[j]));
                }
            }
            let d = eval(z[i]).div(den);
            z[i] = z[i].sub(d);
            md = md.max(d.abs());
        }
        if md < 1e-14 {
            break;
        }
    }
    z
}

/// Characteristic polynomial of multiplication-by-α in O_K (structure constants
/// `sc`), where `α = Σ coeff_i ω_i`. Faddeev–LeVerrier; little-endian, monic.
fn charpoly_from_sc(coeff: &[Integer], sc: &[Vec<Vec<Integer>>], n: usize) -> Vec<Integer> {
    // M[k][j] = Σ_i coeff_i · sc[i][j][k]  (multiply-by-α matrix in O_K basis)
    let mut mat = vec![vec![izero(); n]; n];
    for k in 0..n {
        for j in 0..n {
            let mut s = izero();
            for i in 0..n {
                if !coeff[i].is_zero() {
                    s = s + coeff[i].clone() * sc[i][j][k].clone();
                }
            }
            mat[k][j] = s;
        }
    }
    let mut c = vec![izero(); n + 1];
    c[0] = ione();
    let mut mk = vec![vec![izero(); n]; n];
    for kk in 1..=n {
        let mut tmp = mk.clone();
        for i in 0..n {
            tmp[i][i] = tmp[i][i].clone() + c[kk - 1].clone();
        }
        // mk = mat * tmp
        let mut nm = vec![vec![izero(); n]; n];
        for i in 0..n {
            for t in 0..n {
                if mat[i][t].is_zero() {
                    continue;
                }
                for j in 0..n {
                    nm[i][j] = nm[i][j].clone() + mat[i][t].clone() * tmp[t][j].clone();
                }
            }
        }
        mk = nm;
        let mut tr = izero();
        for i in 0..n {
            tr = tr + mk[i][i].clone();
        }
        c[kk] = -(tr / Integer::from(kk as i64));
    }
    let mut le: Vec<Integer> = c.into_iter().rev().collect();
    while le.len() > 1 && le.last().unwrap().is_zero() {
        le.pop();
    }
    le
}

fn supnorm(f: &[Integer]) -> Integer {
    f.iter().map(|c| c.abs()).max().unwrap_or_else(izero)
}

/// True `polredabs`-style reduction: a small same-field defining polynomial found
/// by LLL-reducing the `T₂` lattice of the **maximal order** `O_K` (not just the
/// equation order). Returns the smallest squarefree degree-`n` model.
pub fn polredabs(f: &[Integer]) -> Vec<Integer> {
    let n = f.len() - 1;
    if n < 2 {
        return f.to_vec();
    }
    let (o, _disc) = maximal_order(f, 100_000);
    let table = power_table(f);
    let sc = o.structure_constants(&table);
    let rts = roots_cpx(f);
    // classify embeddings: reals, one per conjugate pair
    let mut reals: Vec<Cpx> = Vec::new();
    let mut cplx: Vec<Cpx> = Vec::new();
    let mut used = vec![false; rts.len()];
    for i in 0..rts.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        if rts[i].im.abs() < 1e-6 {
            reals.push(rts[i]);
        } else {
            cplx.push(rts[i]);
            let mut best = usize::MAX;
            let mut bd = f64::INFINITY;
            for j in 0..rts.len() {
                if j != i && !used[j] {
                    let d = (rts[i].re - rts[j].re).abs() + (rts[i].im + rts[j].im).abs();
                    if d < bd {
                        bd = d;
                        best = j;
                    }
                }
            }
            if best != usize::MAX {
                used[best] = true;
            }
        }
    }
    // T2 lattice rows = Minkowski embedding of each integral basis element ω_j
    let scale = 1e6;
    let s2 = std::f64::consts::SQRT_2;
    let dd = o.d.to_f64().unwrap_or(1.0);
    let mut lattice: Vec<Vec<Integer>> = Vec::with_capacity(n);
    for j in 0..n {
        let col = o.col(j); // numerators of ω_j (÷ d)
        let coef: Vec<f64> = col.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
        let embed = |r: Cpx| -> Cpx {
            // Σ_i coef_i r^i / d
            let mut acc = Cpx { re: 0.0, im: 0.0 };
            let mut pw = Cpx { re: 1.0, im: 0.0 };
            for i in 0..n {
                acc = acc.add(Cpx { re: coef[i], im: 0.0 }.mul(pw));
                pw = pw.mul(r);
            }
            Cpx { re: acc.re / dd, im: acc.im / dd }
        };
        let mut row: Vec<Integer> = Vec::with_capacity(n);
        for r in &reals {
            row.push(Integer::from((embed(*r).re * scale).round() as i64));
        }
        for r in &cplx {
            let e = embed(*r);
            row.push(Integer::from((s2 * e.re * scale).round() as i64));
            row.push(Integer::from((s2 * e.im * scale).round() as i64));
        }
        lattice.push(row);
    }
    let (_red, u) = rustmath_matrix::lll::lll_reduce(&lattice);

    let mut best = f.to_vec();
    for row in &u {
        // Don't skip rows by coordinate shape: with the HNF integral basis the first
        // element need not be 1 (e.g. (1+θ)/2). Rational α are excluded anyway by the
        // squarefree filter — their charpoly is (x-a)^n, discriminant 0.
        let cp = charpoly_from_sc(row, &sc, n);
        if cp.len() == n + 1 && rustmath_polynomials::disc::discriminant(&cp) != izero() {
            if supnorm(&cp) < supnorm(&best) {
                best = cp;
            }
        }
    }
    best
}

#[cfg(test)]
mod polredabs_tests {
    use super::*;
    use rustmath_polynomials::disc::discriminant;

    fn p(cs: &[i64]) -> Vec<Integer> {
        cs.iter().map(|&c| Integer::from(c)).collect()
    }

    #[test]
    fn polredabs_quadratic_uses_maximal_order() {
        // x^2 - 5: O_K = Z[(1+√5)/2]; polredabs should reach disc 5 (e.g. x^2-x-1),
        // which the power-basis polred of x^2-5 (disc 20) cannot.
        let r = polredabs(&p(&[-5, 0, 1]));
        assert_eq!(r.len(), 3);
        assert_eq!(discriminant(&r).abs(), Integer::from(5));
    }

    #[test]
    fn polredabs_quadratic_already_maximal() {
        // x^2 + 1 already maximal, disc -4
        let r = polredabs(&p(&[1, 0, 1]));
        assert_eq!(discriminant(&r).abs(), Integer::from(4));
    }

    #[test]
    fn polredabs_dedekind_cubic_same_field() {
        // x^3 - x^2 - 2x - 8 is Dedekind's NON-monogenic cubic: no generator gives a
        // poldisc-503 model (every poldisc = -503·index²). The invariant is that
        // polredabs defines the same field — field discriminant still -503.
        let r = polredabs(&p(&[-8, -2, -1, 1]));
        assert_eq!(r.len(), 4);
        assert_eq!(field_discriminant(&r).abs(), Integer::from(503));
        // its poldisc is 503 times a perfect square (the squared index)
        let ratio = discriminant(&r).abs() / Integer::from(503);
        let s = ratio.sqrt().unwrap();
        assert_eq!(s.clone() * s, ratio);
    }
}
