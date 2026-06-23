# Chapter 12 — Lists

**Handbook part:** II — Sets, Sequences, and Mappings
**Handbook pages:** 223–225 (PDF pages 356–358)

---

## Scope and overview

A list in Magma is an ordered finite collection of objects. Unlike sequences, lists are not
required to consist of objects that have some common parent. Lists are not stored compactly
and the operations provided for them are not extensive. They are mainly provided to enable
the user to gather assorted objects temporarily together.

Lists support construction via special bracket notation `[* … *]`, concatenation, insertion,
pruning, conversion from sequences and tuples, indexed access, and assignment. No algorithmic
references are given in this chapter — all operations are definitional or purely structural.

---

## 12.1 Introduction

An ordered finite collection of heterogeneous objects. There are no compactness or
uniformity requirements; elements need not share a common parent.

---

## 12.2 Construction of Lists

Lists are constructed by expressions enclosed in special brackets `[*` and `*]`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `[* *]` | The empty list. | — |
| `[* e1, e2, ..., en *]` | Given expressions e1, …, en defining elements a1, a2, …, an, create the list containing a1, a2, …, an. | — |

---

## 12.3 Creation of New Lists

Throughout this section, S denotes the list `[* s1, …, sn *]` and T denotes the list
`[* t1, …, tm *]`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `S cat T` | The list formed by concatenating the terms of S with the terms of T, i.e. `[* s1, …, sn, t1, …, tm *]`. | — |
| `S cat:= T` | (Procedure.) Destructively concatenate the terms of T to S; S becomes `[* s1, …, sn, t1, …, tm *]`. | — |
| `Append(S, x)` | The list formed by adding the object x to the end of S, i.e. `[* s1, …, sn, x *]`. | — |
| `Append(~S, x)` | (Procedure.) Destructively add the object x to the end of S; S becomes `[* s1, …, sn, x *]`. | — |
| `Insert(~S, i, x)` / `Insert(S, i, x)` | Create the list formed by inserting x at position i in S and moving S[i], …, S[n] down one place, i.e. `[* s1, …, s_{i−1}, x, si, …, sn *]`. i must not exceed n + 1. The procedural form (reference `~S`) replaces S in place and is much more efficient since S is not copied; the functional form returns the new list. | — |
| `Prune(S)` | The list formed by removing the last term of S, i.e. `[* s1, …, s_{n−1} *]`. | — |
| `Prune(~S)` | (Procedure.) Destructively remove the last term of S; S becomes `[* s1, …, s_{n−1} *]`. | — |
| `SequenceToList(Q)` / `Seqlist(Q)` | Given a sequence Q, construct a list whose terms are the elements of Q taken in the same order. | — |
| `TupleToList(T)` / `Tuplist(T)` | Given a tuple T, construct a list whose terms are the elements of T taken in the same order. | — |
| `Reverse(L)` | Given a list L, return the same list in reverse order. | — |

---

## 12.4 Access Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#S` | The length of the list S. | — |
| `IsEmpty(S)` | Return whether S is empty (has zero length). | — |
| `S[i]` | Return the i-th term of the list S. An error results if i ≤ 0 or i > #S + 1. i is allowed to be a multi-index (see Section 8.3.1). | — |
| `S[I]` | Return the sublist of S given by the indices in the sequence I. Each index in I must be in the range [1..l], where l is the length of S. | — |
| `IsDefined(L, i)` | Checks whether the i-th item in L is defined; returns true if i ≤ #L and false otherwise. | — |

---

## 12.5 Assignment Operator

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `S[i] := x` | Redefine the i-th term of the list S to be x. If i ≤ 0, an error results. If i = #S + 1, x is appended to S. If i > #S + 1, an error results. i is allowed to be a multi-index. | — |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| List construction (literal syntax) | `[* *]`, `[* e1, …, en *]` |
| Concatenation | `cat`, `cat:=` |
| Append / insert / remove | `Append`, `Append(~S, x)`, `Insert`, `Insert(~S, i, x)`, `Prune`, `Prune(~S)` |
| Conversion from other types | `SequenceToList` / `Seqlist`, `TupleToList` / `Tuplist`, `Reverse` |
| Access and membership | `#`, `IsEmpty`, `S[i]`, `S[I]`, `IsDefined` |
| Assignment | `S[i] := x` |
