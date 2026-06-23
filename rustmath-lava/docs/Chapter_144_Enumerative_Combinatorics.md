# Chapter 144 — Enumerative Combinatorics

**Handbook part:** XX — Combinatorics
**Handbook pages:** 4807–4809 (PDF pages 4938–4943)

---

## Scope and overview

This chapter presents some of the tools provided by Magma for enumerative combinatorics. It
is short and almost entirely a catalogue of self-contained intrinsics: classical counting
functions (factorials, binomial and multinomial coefficients, Fibonacci/Lucas/generalized
Fibonacci numbers, Catalan, Stirling, Bell, Eulerian, harmonic and Bernoulli numbers, and
the Bernoulli polynomial), and constructors that enumerate the subsets, multisets,
subsequences and permutations of a finite set.

Each function computes a standard combinatorial quantity from its closed form or defining
recursion; no specialised algorithmic machinery or bibliography is associated with the
chapter.

---

## 144.1 Introduction

This chapter presents some of the tools provided by Magma for enumerative combinatorics.

---

## 144.2 Combinatorial Functions

Classical counting functions, each returning a number (or polynomial) defined by a standard
formula or recursion.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Factorial(n)` | The factorial `n!` for non-negative small integer `n`. | Direct product. |
| `NumberOfPermutations(n, k)` | The number of permutations of `n` distinct objects taken `k` at a time. | Falling factorial `n!/(n−k)!`. |
| `Binomial(n, r)` | The binomial coefficient `C(n, r)` = `(n choose r)`. | Closed form. |
| `Multinomial(n, [r_1, ... r_n])` | Given a sequence `Q = [r_1,…,r_k]` of positive integers with `n = r_1 + … + r_k`, the multinomial coefficient `n! / (r_1! ⋯ r_k!)`. | Closed form. |
| `Fibonacci(n)` | For integer `n`, the `n`-th Fibonacci number `F_n`, defined by `F_0 = 0`, `F_1 = 1`, `F_n = F_{n−1} + F_{n−2}` for all integers `n`. `n` may be negative, with `F_{−n} = (−1)^{n+1} F_n`. | Fibonacci recursion (extended to negative indices). |
| `Catalan(n)` | For a small non-negative integer `n`, the `n`-th Catalan number `C_n`, defined by `C_0 = 1`, `C_{n+1} = C_n · (4n + 2)/(n + 2)`. | Catalan recursion. |
| `Lucas(n)` | For integer `n`, the `n`-th Lucas number `L_n`, defined by `L_0 = 2`, `L_1 = 1`, `L_n = L_{n−1} + L_{n−2}` for all integers `n`. `n` may be negative, with `L_{−n} = (−1)^n L_n`. | Lucas recursion (extended to negative indices). |
| `GeneralizedFibonacciNumber(g0, g1, n)` | The `n`-th member of the generalized Fibonacci sequence defined by `G_0 = g0`, `G_1 = g1`, `G_n = G_{n−1} + G_{n−2}` for all integers `n` (`n` may be negative). Fibonacci and Lucas numbers are the special cases `(g0, g1) = (0, 1)` and `(2, 1)` respectively. | Two-term linear recursion (extended to negative indices). |
| `StirlingFirst(n, k)` | The Stirling number of the first type, `[n k]` (`c(n, k)`), for non-negative integers `n`, `k`. | Stirling-number-of-the-first-kind recursion. |
| `StirlingSecond(n, k)` | The Stirling number of the second type, `{n k}` (`S(n, k)`), for non-negative integers `n`, `k`. | Stirling-number-of-the-second-kind recursion. |
| `Bell(n)` | The `n`-th Bell number, the number of partitions of a set of size `n`. (Not to be confused with `NumberOfPartitions(n)`, which gives the number of partitions of the *integer* `n`.) Equals the sum of `StirlingSecond(n, k)` for `k` from `0` to `n` inclusive. | Sum of second-kind Stirling numbers. |
| `EulerianNumber(n, r)` | The number `E(n, r)` of permutations `p` of `{1,…,n}` having exactly `r` ascents (places where `p_i < p_{i+1}`). | Eulerian-number recursion. |
| `HarmonicNumber(n)` | The `n`-th harmonic number `H_n = Σ_{i=1}^{n} 1/i`. | Direct summation. |
| `BernoulliNumber(n)` | The `n`-th Bernoulli number `B_n` as a rational number. | — |
| `BernoulliApproximation(n)` | A real approximation to the `n`-th Bernoulli number `B_n`. | Real (floating-point) approximation. |
| `BernoulliPolynomial(n)` | The `n`-th Bernoulli polynomial `B_n(x) = Σ_{k=0}^{n} (n choose k) B_k x^{n−k}`, where `B_n` is the `n`-th Bernoulli number. | Closed form in terms of Bernoulli numbers. |

---

## 144.3 Subsets of a Finite Set

Constructors that enumerate the subsets, multisets, subsequences and permutations of a finite
set `S`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Subsets(S)` | The set of all subsets of the set `S`. | Power-set enumeration. |
| `Subsets(S, k)` | The set of subsets of the set `S` of size `k`. If `k` is larger than the cardinality of `S`, the result is empty. | `k`-subset enumeration. |
| `Multisets(S, k)` | The set of multisets consisting of `k` not necessarily distinct elements of the set `S`. | `k`-multiset enumeration. |
| `Subsequences(S, k)` | The set of sequences of length `k` with elements from the set `S`. | Length-`k` tuple enumeration. |
| `Permutations(S)` | The set of permutations (stored as sequences) of the elements of the set `S`. | Permutation enumeration. |
| `Permutations(S, k)` | The set of permutations (stored as sequences) of each of the subsets of the set `S` of cardinality `k`. | `k`-subset permutation enumeration. |

*Worked example:* H144E1 (`Subsets` used to construct the Petersen graph as the third Odd Graph: the `n`-th Odd Graph has vertices in correspondence with the `(n−1)`-element subsets of `{1…2n−1}`, with edges between vertices whose sets have empty intersection — via `Subsets({1..2*n-1}, n-1)`, `IsDisjoint`, and `Graph<…>`).

---

## 144.4 Bibliography (canonical references)

The chapter contains no bibliography.

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Classical counting functions (closed form / recursion) | `Factorial`, `NumberOfPermutations`, `Binomial`, `Multinomial`, `Catalan`, `EulerianNumber`, `HarmonicNumber` |
| Two-term linear recursions | `Fibonacci`, `Lucas`, `GeneralizedFibonacciNumber` |
| Stirling / Bell numbers | `StirlingFirst`, `StirlingSecond`, `Bell` |
| Bernoulli numbers and polynomial | `BernoulliNumber`, `BernoulliApproximation`, `BernoulliPolynomial` |
| Finite-set enumeration | `Subsets`, `Multisets`, `Subsequences`, `Permutations` |
