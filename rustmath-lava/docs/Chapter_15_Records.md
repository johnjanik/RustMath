# Chapter 15 — Records

**Handbook part:** II — Sets, Sequences, and Mappings
**Handbook pages:** 241–244 (PDF pages 372–377)

---

## Scope and overview

A record in Magma is a data structure that collects several objects into named fields. Fields
are accessed by fieldname rather than by integer index, distinguishing records from tuples.
Unlike sets or sequences, the objects in a record need not all be of the same kind.

Records differ from tuples in two key ways. First, the components of a tuple are indexed by
integers and every component must be defined, whereas the fields of a record are indexed by
fieldnames and may be assigned or deleted at any time — a record may be entirely empty or
partially assigned. Second, every record must be constructed according to a pre-defined
**record format**, whereas Magma can deduce the parent of a tuple from the tuple itself.

When a record format is defined, each field may optionally be given a parent magma or a
category; any record built from that format must then store only values conforming to that
restriction in the corresponding field. If no restriction is given, the field may hold any
value, and different records in the same format may hold values of entirely different kinds in
that field.

Because of this flexibility — whether a field is assigned and what kind of value it holds —
Boolean comparison operators are not available for records.

---

## 15.1 Introduction

*(Prose-only section; no intrinsics. See Scope and overview above.)*

---

## 15.2 The Record Format Constructor

A record format must be created before any records in that format can be created. The special
constructor `recformat< ... >` is used for this purpose.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `recformat< L >` | Construct the record format corresponding to the non-empty fieldname list `L`. Each term of `L` must be one of: (a) `fieldname` — no restriction on values; (b) `fieldname : expression` — expression evaluates to a magma that is the required parent; or (c) `fieldname : expression` — expression evaluates to a category that is the required category. Fieldnames must consist of characters that form a valid identifier name (not a string). | — |

*Worked examples: H15E1 (creating a record format with fields `n : Integers()`, `misc` (unrestricted), and `seq : SeqEnum`; inspecting the format with `Names`).*

---

## 15.3 Creating a Record

Before a record is created its record format must be defined. A record may be created with as
few or as many of its fields assigned as desired; there is no requirement to assign every field
at construction time.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `rec< F \| L >` | Given a record format `F`, construct the record corresponding to the field assignment list `L`. Each term of `L` must have the form `fieldname := expression`, where `fieldname` is declared in `F` and the value of the expression conforms (directly or by coercion) to any restriction on that field. `L` may be empty and the fieldnames may appear in any order. | — |

*Worked examples: H15E2 (building four records `r`, `s`, `t`, `u` in the format RF — empty record, record with all fields assigned including a GF(13) sequence, a record where `n := 51/3` is coerced to 17, and a record whose `misc` field holds an RModule).*

---

## 15.4 Access and Modification Functions

Fields of records may be inspected, assigned, and deleted at any time after the record is
created.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Format(r)` | The record format of record `r`. | — |
| `Names(F)` | The fieldnames of record format `F`, returned as a sequence of strings. | — |
| `Names(r)` | The fieldnames of record `r`, returned as a sequence of strings. | — |
| `r'fieldname` | Return the value of the field named `fieldname` in record `r`. The format of `r` must include this fieldname and the field must currently be assigned. | — |
| `r'fieldname := expression;` | Assign (or reassign) the field named `fieldname` in record `r` to the value of `expression`. The format of `r` must include this fieldname, and the value must satisfy (directly or by coercion) any restriction on the field. | — |
| `delete r'fieldname` | (Statement.) Delete the current value of the field named `fieldname` in record `r`. | — |
| `assigned r'fieldname` | Returns `true` if and only if the field named `fieldname` in record `r` currently holds a value. | — |
| `r''s` | Given an expression `s` that evaluates to a string, return the field of record `r` whose fieldname corresponds to that string. The format of `r` must include this fieldname and the field must be assigned. This syntax may be used anywhere that `r'fieldname` may be used, including on the left-hand side of assignment, in `assigned`, and in `delete`. | — |

*Worked examples: H15E3 (using `assigned`, field access `r'seq`, dynamic assignment `r'seq := Append(t'seq, t'n)`, string-based field access `t''(s'misc)` (produces a runtime error — field `adsifaj` does not exist), and `delete u''("m" cat "isc")`).*

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| Record format construction | `recformat< L >` |
| Record construction | `rec< F \| L >` |
| Field introspection | `Format`, `Names` |
| Field access (by name or by string) | `r'fieldname`, `r''s` |
| Field mutation and lifetime management | `r'fieldname := expression`, `delete r'fieldname`, `assigned r'fieldname` |
