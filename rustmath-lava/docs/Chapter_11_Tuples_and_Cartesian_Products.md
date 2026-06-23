# Chapter 11 — Tuples and Cartesian Products

**Handbook part:** II — Sets, Sequences, and Mappings
**Handbook pages:** 215–219 (PDF pages 346–353)

---

## Scope and overview

A cartesian product in Magma may be constructed from a finite number of factors, each of
which may be a set or algebraic structure. An element of a cartesian product is called a
**tuple**.

Tuples are semantically distinct from sequences. Sequences are elements of a cartesian
product of *n* copies of a single fixed set or structure, and may grow and shrink during
their lifetime (implying a varying parent cartesian product). Tuples, by contrast, are
elements of cartesian products whose factors may be *different* sets or structures, and
the parent cartesian product of a tuple is **fixed once and for all** at creation time.

The chapter covers: construction of cartesian products (including nested and power
variants); creation and in-place modification of tuples; access and conversion functions;
equality testing; and the fold-product operator.

---

## 11.1 Introduction

No intrinsics are introduced in the introductory section; it describes the conceptual
distinction between tuples and sequences as summarised above.

---

## 11.2 Cartesian Product Constructor and Functions

The special constructor `car< ... >` is used to create cartesian products of structures.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `car< R1, ..., Rk >` | Given a list of sets or algebraic structures R1, …, Rk, construct the cartesian product set R1 × ··· × Rk. | — |
| `CartesianProduct(R, S)` | Given structures R and S, construct the cartesian product R × S. Equivalent to calling the `car` constructor with two arguments. | — |
| `CartesianProduct(L)` | Given a sequence or tuple L of structures, construct the cartesian product of the elements of L. | — |
| `CartesianPower(R, k)` | Given a structure R and a non-negative integer k, construct the cartesian power set Rk. | — |
| `Flat(C)` | Given a cartesian product C whose factors may themselves be cartesian products, return the cartesian product of the base (non-cartesian-product) structures considered in depth-first order. See also `Flat` for the element version. | — |
| `NumberOfComponents(C)` | Return the number of components (factors) of the cartesian product C. | — |
| `Component(C, i)` / `C[i]` | Return the i-th component (factor) of the cartesian product C. | — |
| `#C` | Return the cardinality of the cartesian product C. | — |
| `Rep(C)` | Return a representative element of the cartesian product C. | — |
| `Random(C)` | Return a random element of the cartesian product C. | — |

*Worked examples: H11E1 (creating the product of Q and Z using `car< RationalField(), Integers() >`).*

---

## 11.3 Creating and Modifying Tuples

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `elt< C \| a1, a2, ..., ak >` / `C ! < a1, a2, ..., ak >` | Given a cartesian product C = R1 × ··· × Rk and elements a1, …, ak with ai ∈ Ri, create the tuple T = ⟨a1, a2, …, ak⟩ of C. | — |
| `< a1, a2, ..., ak >` | Given elements a1, …, ak with ai ∈ Ri, create the tuple T = ⟨a1, …, ak⟩. If the corresponding cartesian product C does not already exist it is created at evaluation time. | — |
| `Append(T, x)` | Return the tuple formed by adding object x to the end of tuple T. The result lies in a new cartesian product. | — |
| `Append(~T, x)` | (Procedure.) Destructively add object x to the end of tuple T. The new T lies in a new cartesian product. | — |
| `Prune(T)` | Return the tuple formed by removing the last term of tuple T. The length of T must be greater than 1. The result lies in a new cartesian product. | — |
| `Prune(~T)` | (Procedure.) Destructively remove the last term of tuple T. The length of T must be greater than 1. The new T lies in a new cartesian product. | — |
| `Flat(T)` | Construct the flattened version of tuple T, performing the flattening depth-first (analogous to `Flat` for cartesian products). | — |

*Worked examples: H11E2 (creating the product C of Z and Q; coercing `< 26/13, 13/26 >`; building a set of pairs of primes and their reciprocals using a set comprehension).*

---

## 11.4 Tuple Access Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Parent(T)` | Return the cartesian product to which the tuple T belongs. | — |
| `#T` | Return the number of components of the tuple T. | — |
| `T[i]` | Return the i-th component of the tuple T. This indexing may also be used on the left-hand side to modify T in place. | — |
| `Explode(T)` | Given a tuple T of length n, return the n entries of T in order (as multiple return values). | — |
| `TupleToList(T)` / `Tuplist(T)` | Given a tuple T, return a list (sequence) containing the entries of T. | — |

*Worked examples: H11E3 (creating a tuple `< 11/2, 13/3, RootOfUnity(3, CyclotomicField(3)) >`; querying `#f`, `Parent(f)`, arithmetic on components; assigning `f[3] := 7`).*

---

## 11.5 Equality

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `T eq U` | Return `true` if and only if tuples T and U are equal. | — |
| `T ne U` | Return `true` if and only if tuples T and U are distinct. | — |

---

## 11.6 Other Operations

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `&*T` | For a tuple T whose components all lie in a structure supporting multiplication and with a common over-structure, return the product of all entries. | — |

---

## Bibliography

No bibliography is present in Chapter 11. The chapter introduces built-in language
constructs and data structures with no external algorithmic references.

---

## Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Cartesian product construction | `car< >`, `CartesianProduct`, `CartesianPower` |
| Cartesian product inspection | `NumberOfComponents`, `Component`/`C[i]`, `#C`, `Rep`, `Random`, `Flat(C)` |
| Tuple construction | `elt< >`, `C ! < >`, `< >` |
| Tuple modification | `Append`, `Append(~T, x)`, `Prune`, `Prune(~T)`, `Flat(T)` |
| Tuple access and conversion | `Parent`, `#T`, `T[i]`, `Explode`, `TupleToList`/`Tuplist` |
| Equality testing | `eq`, `ne` |
| Fold product | `&*` |
