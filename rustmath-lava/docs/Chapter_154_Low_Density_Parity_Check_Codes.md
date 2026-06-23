# Chapter 154 — Low Density Parity Check Codes

**Handbook part:** XXI — Coding Theory
**Handbook pages:** 5157–5165 (PDF pages 5288–5299)

---

## Scope and overview

Low density parity check (LDPC) codes are among the best-performing codes in practice,
capable of correcting errors close to the Shannon limit. Magma provides facilities for the
construction, decoding, simulation and analysis of LDPC codes.

LDPC codes come in two main varieties, *regular* and *irregular*, defined by the row and
column weights of the sparse parity check matrix. If all columns of the parity check matrix
have a constant weight `a` and all rows have a constant weight `b`, the code is said to be
`(a, b)`-regular; when either the columns or rows have a distribution of weights the code is
*irregular*. Few explicit construction techniques exist; more commonly LDPC codes are
selected at random from an *ensemble* and their properties determined through simulation.

In Magma a code is considered LDPC solely on the basis of whether a (sparse-type, `MtrxSprs`)
LDPC parity check matrix has been assigned to it. Decoding uses the iterative LDPC decoding
algorithm, and transmission can be simulated over a binary symmetric channel (bit-flips with
probability `p < 0.5`) or a white Gaussian noise channel (real values mapped to ±1 with
errors normally distributed about 0 with standard deviation `σ`).

The asymptotic performance of an ensemble (a pair of degree distributions for the variable
and check nodes of the Tanner graph) is analysed via *density evolution*. The critical
parameter for a channel is the ensemble's *threshold*: below it a random code from the
ensemble decodes with error probability tending to zero, above it there is a non-vanishing
finite error probability.

---

## 154.1 Introduction

### 154.1.1 Constructing LDPC Codes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LDPCCode(H)` | Given a sparse binary matrix `H`, return the LDPC code which has `H` as its parity check matrix. | Direct construction from the assigned sparse parity check matrix. |
| `GallagerCode(n, a, b)` | Return a random `(a, b)`-regular LDPC code of length `n`. The row weight `a` must divide the length `n`. | Gallager's original method of construction. |
| `RegularLDPCEnsemble(n, a, b)` | Return a random code from the ensemble of `(a, b)`-regular binary LDPC codes. | Random selection from the regular ensemble. |
| `IrregularLDPCEnsemble(n, Sv, Sc)` | Given (unnormalized) distributions `Sv` and `Sc` for the variable and check weights, return a length-`n` irregular LDPC code whose degree distributions match the given distribution. `Sv`, `Sc` are sequences of real numbers, where the `i`-th entry indicates the percentage of variable (resp. check) nodes that should have weight `i`. Distributions are not matched perfectly unless everything is in complete balance. | Random selection matching the prescribed degree distributions. |
| `MargulisCode(p)` | Return the `(3,6)`-regular binary LDPC code of length `2(p³ − p)` using the group-based construction of Margulis. | Margulis group-based explicit construction. |

*Worked example:* H154E1 (re-using a random ensemble code by saving its sparse parity check matrix via `LDPCMatrix` and rebuilding with `LDPCCode`).

### 154.1.2 Access Functions

A code can have many different parity check matrices, so the matrix that defines a code as
being LDPC must be assigned specifically. Any parity check matrix can be assigned for this
purpose, and once an LDPC matrix is assigned the code is considered LDPC regardless of the
density or other properties of the matrix. The matrix must be of sparse type (`MtrxSprs`).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsLDPC(C)` | Return `true` if `C` is an LDPC code (i.e. if it has been assigned an LDPC matrix). | — |
| `AssignLDPCMatrix(∼C, H)` | Given a sparse matrix `H` which is a parity check matrix of the code `C`, assign `H` as the LDPC matrix of `C`. | — |
| `LDPCMatrix(C)` | Given an LDPC code `C`, return the sparse matrix which has been assigned as its low density parity check matrix. | — |
| `LDPCDensity(C)` | Given an LDPC code `C`, return the density of the sparse matrix which has been assigned as its low density parity check matrix. | — |
| `IsRegularLDPC(C)` | Return `true` if `C` is an LDPC code and has regular column and row weights. If `true`, the row and column weights are also returned. | — |
| `TannerGraph(C)` | For an LDPC code `C`, return its Tanner graph. If there are `n` variables and `m` checks, the graph has `n + m` nodes, the first `n` of which are the variable nodes. | — |
| `LDPCGirth(C)` | For an LDPC code `C`, return the girth of its Tanner graph. | Girth of the Tanner graph. |
| `LDPCEnsembleRate(v, c)` / `LDPCEnsembleRate(Sv, Sc)` | Return the theoretical rate of LDPC codes from the ensemble described by the given inputs (either two integers for a `(v, c)`-regular ensemble, or two density distributions `Sv`, `Sc`). | Theoretical ensemble rate from the degree distributions. |

*Worked example:* H154E2 (assigning a sparse parity check matrix of a `RandomLinearCode` to make any code LDPC; comparing `LDPCDensity` of a random code with that of a `RegularLDPCEnsemble`).

### 154.1.3 LDPC Decoding and Simulation

The performance of LDPC codes lies in their iterative decoding algorithm. Magma provides
facilities to decode using LDPC codes, as well as simulating transmission over a binary
symmetric or white Gaussian noise channel. The binary symmetric channel transmits binary
values and is defined by `p < 0.5`, each bit independently sustaining a bit-flip error with
probability `p`. The Gaussian channel is analog, transmitting real values; binary values are
mapped to −1 and 1 before transmission, and each value independently sustains an error
normally distributed about 0 with standard deviation `σ`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LDPCDecode(C, v)` | For an LDPC code `C` and a received vector `v`, decode `v` to a codeword of `C` using the LDPC iterative decoding algorithm. The channel is described by `Channel` (`"BinarySymmetric"` (default) or `"Gaussian"`); errors on the binary symmetric channel are described by `p` (default `0.1`), on the Gaussian channel by `StdDev` (default `0.25`). `v` must be over a ring corresponding to the channel: a binary vector over `F₂` for the binary symmetric channel, real-valued for the Gaussian channel. `Iterations` (default `Dimension(C)`) sets the maximum number of iterations; the default is much larger than normally used in practice, giving maximum error-correcting performance at possible cost to efficiency. | LDPC iterative (message-passing) decoding algorithm. |
| `LDPCSimulate(C, N)` | For an LDPC code `C`, simulate `N` transmissions across the given channel and return the accumulated bit error rate and word error rate. Variable arguments (`Channel` default `"BinarySymmetric"`, `p` default `0.1`, `StdDev` default `0.25`, `Iterations` default `Dimension(C)`) are as for `LDPCDecode`; the channel controls both the decoding algorithm and the nature of the errors introduced during simulation. | Repeated channel simulation plus iterative decoding. |

*Worked examples:* H154E3 (bit-flip errors over the binary symmetric channel, decoding a received vector with `LDPCDecode`); H154E4 (mapping a codeword into the real domain, introducing normally-distributed errors via an `Erf`-based discrete distribution, and decoding over the Gaussian channel); H154E5 (`LDPCSimulate` over the Gaussian channel at increasing `StdDev`, showing bit error rate always below word error rate).

### 154.1.4 Density Evolution

The asymptotic performance of ensembles of LDPC codes can be determined using *density
evolution*. An ensemble (regular or irregular) is defined by a pair of degree distributions
corresponding to the degrees at the variable and check nodes of the Tanner graph. Over a
specific channel, the critical parameter defining the asymptotic performance is the ensemble's
*threshold*: a value of the channel parameter (`p` for the binary symmetric channel, `σ` for
the Gaussian channel). Below the threshold a code from the ensemble decodes with error
probability tending to zero; above it there is a non-vanishing finite error probability.

Determining the threshold over the binary symmetric channel is relatively trivial; over the
real-valued Gaussian channel it can involve extensive computation. The speed depends heavily
on the granularity of the discretization used, which also affects accuracy. The default
settings use a reasonably coarse discretization, emphasizing speed over accuracy; these
approximate results can help reduce the workload of finer-discretization calculations when
more accuracy is required.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `LDPCBinarySymmetricThreshold(v, c)` / `LDPCBinarySymmetricThreshold(Sv, Sc)` | Determine the threshold of the described ensemble over the binary symmetric channel — the critical channel-parameter value above which there is a non-vanishing (asymptotic) error probability. The ensemble may be given by two integers for a `(v, c)`-regular code, or by two density distributions `Sv`, `Sc` (sequences of non-negative reals, not necessarily normalized; the first entry, corresponding to weight-1 nodes in the Tanner graph, should always be zero). `Precision` (default `0.00005`) controls the precision of the threshold. | Establishes lower and upper bounds on the threshold, then narrows the range by repeatedly performing density evolution on the midpoint. |
| `DensityEvolutionBinarySymmetric(v, c, p)` / `DensityEvolutionBinarySymmetric(Sv, Sc, p)` | Perform density evolution on the binary symmetric channel using channel parameter `p` and determine the asymptotic behaviour for the given LDPC ensemble. Returns a boolean; `true` indicates `p` is below the threshold and the ensemble has error probability asymptotically tending to zero. | Density evolution at the single channel parameter `p`. |
| `LDPCGaussianThreshold(v, c)` / `LDPCGaussianThreshold(Sv, Sc)` | Determine the threshold of the described ensemble over the Gaussian channel — the critical standard-deviation value above which there is a non-vanishing (asymptotic) error probability. Ensemble given by two integers `(v, c)` or by two density distributions `Sv`, `Sc` (non-negative reals, first entry zero). `Lower` (default `0`) / `Upper` (default `∞`): real-valued bounds on the threshold that, if tight, reduce the search range (validity verified before the search; error if incorrect). `Points` (default `500`) / `MaxLLR` (default `25`): define the discretized basis of log likelihood ratios on which density evolution is performed — the probability mass function is defined on `[−MaxLLR, …, MaxLLR]` over `2·Points + 1` discretized points. `MaxIterations` (default `∞`): finite limit on iterations per channel parameter (may speed computation but result may not be valid). `QuickCheck` (default `true`): selects how asymptotic behaviour is identified — if `false`, the density must evolve to within an infinitesimal of unity; if `true`, behaviour is assumed to go to unity once the rate of change is successively increasing (empirically accurate but without theoretical justification). `Precision` (default `0.00005`): precision of the threshold. Verbose mode `Code` prints the threshold bounds as evolutions narrow the range. | Establishes lower and upper bounds on the threshold, then narrows by repeated density evolution on the midpoint over the discretized LLR basis. |
| `DensityEvolutionGaussian(v, c, σ)` / `DensityEvolutionGaussian(Sv, Sc, σ)` | Perform density evolution on the Gaussian channel using standard deviation `σ` and determine the asymptotic behaviour for the given LDPC ensemble. Returns a boolean; `true` indicates `σ` is below the threshold and the ensemble has error probability asymptotically tending to zero. Variable arguments (`Points` default `500`, `MaxLLR` default `25`, `MaxIterations` default `∞`, `QuickCheck` default `true`) are as for `LDPCGaussianThreshold`. | Density evolution at the single standard deviation `σ`. |
| `GoodLDPCEnsemble(i)` | Access a small database of density distributions defining good irregular LDPC ensembles. Returns the published threshold of the ensemble over the Gaussian channel, along with the variable and check degree distributions. The input `i` is a non-negative integer indexing the database (in no particular order). | Lookup in a built-in database of published good ensembles. |

*Worked examples:* H154E6 (`LDPCBinarySymmetricThreshold` for `(3,6)`, `(4,8)`, `(4,10)` regular ensembles, showing the computation is not intensive); H154E7 (comparing a published Gaussian-channel threshold from `GoodLDPCEnsemble` against outputs from different discretization levels).

---

## 154.2 Bibliography (canonical references)

The chapter contains no formal bibliography section. Algorithms are attributed in the prose to
their originators (Gallager's construction method and the Margulis group-based construction),
but no keyed reference list is printed in the handbook for this chapter.

| Key | Reference |
|-----|-----------|
| — | (No formal bibliography is given in Chapter 154. Construction methods are attributed by name to R. G. Gallager and G. A. Margulis.) |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Gallager construction (regular LDPC) | `GallagerCode`, `RegularLDPCEnsemble` |
| Margulis group-based construction | `MargulisCode` |
| Random ensemble selection (regular / irregular) | `RegularLDPCEnsemble`, `IrregularLDPCEnsemble` |
| Direct construction from parity check matrix | `LDPCCode`, `AssignLDPCMatrix` |
| LDPC matrix / property access | `IsLDPC`, `LDPCMatrix`, `LDPCDensity`, `IsRegularLDPC`, `TannerGraph`, `LDPCGirth`, `LDPCEnsembleRate` |
| Iterative (message-passing) decoding | `LDPCDecode`, `LDPCSimulate` |
| Density evolution (binary symmetric channel) | `LDPCBinarySymmetricThreshold`, `DensityEvolutionBinarySymmetric` |
| Density evolution (Gaussian channel) | `LDPCGaussianThreshold`, `DensityEvolutionGaussian` |
| Published good-ensemble database | `GoodLDPCEnsemble` |
