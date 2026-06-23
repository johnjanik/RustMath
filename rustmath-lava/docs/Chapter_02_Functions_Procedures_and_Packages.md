# Chapter 2 — Functions, Procedures and Packages

**Handbook part:** I — THE MAGMA LANGUAGE
**Handbook pages:** 35–62 (PDF pages 166–195)

---

## Scope and overview

Chapter 2 covers the fundamental mechanisms for defining and using callable units of code in Magma: ordinary user functions, procedures, and the package/intrinsic system that lets users extend Magma with new named functions that behave exactly like system built-ins.

The first half (§2.2) describes the two syntactic forms for user-defined functions — the full `function ... end function;` form and the compact `func< | >` short form — and their procedural counterparts. Both support optional named parameters with default values, variadic argument lists (trailing `...`), and self-reference for recursion. The `forward` declaration supports mutual recursion between separately defined functions/procedures.

The second half (§2.3–§2.6) covers the Magma *package* system, which is the mechanism for adding *user intrinsics* to Magma's global signature table. Intrinsics are compiled to pseudo-code, are globally visible once attached, support full overloading and type-dispatch, and are indistinguishable from system functions at the call site. Supporting topics include: spec files for organising collections of packages; the `import` statement for sharing constants across packages; argument-checking directives (`require`, `requirerange`, `requirege`); user-defined attributes on structures; user-defined verbose flags; and user-defined types (since Magma V2.19).

---

## 2.1 Introduction

No intrinsic table (introductory prose only). The section summarises the layout: §2.2 covers ordinary functions and procedures; the remainder covers user-defined intrinsics (packages).

---

## 2.2 Functions and Procedures

Magma provides two slightly different syntactic forms for defining user functions (as opposed to intrinsic functions). An abbreviated form is provided when the definition can be expressed as a single expression. The syntax for user procedures is similar. Names for functions and procedures are ordinary identifiers, obeying the same rules as other variables (Chapter 1).

### 2.2.1 Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `f := function(x1, ..., xn: parameters) statements end function;` | Creates a function of n ≥ 0 arguments and assigns it to `f`. At least one `return expression;` statement is required in every branch. Multiple return values are supported; the underscore `_` may be used for final undefined return slots. Optional named parameters are specified as `identifier := default_value` clauses after the colon; if omitted at call time the default is used. Inside this form, `f` cannot be used recursively (use `$$` to refer to the function itself). | Language construct |
| `function f(x1, ..., xn: parameters) statements end function;` | As above, but `f` is visible inside its own body, enabling direct recursion by name. | Language construct |
| `f := function(x1, ..., xn, ...: parameters) statements end function;` | Variadic form: accepts m ≥ n arguments. Arguments y₁…y_{n-1} bind to x₁…x_{n-1}; arguments y_n…y_m are gathered into a list `[* yn, ..., ym *]` bound to the last parameter xn. | Language construct |
| `function f(x1, ..., xn, ...: parameters) statements end function;` | Variadic form with self-reference by name. | Language construct |
| `f := func< x1, ..., xn: parameters \| expression>;` | Short function constructor: creates a function returning the value of a single expression. Optional parameters permitted. The name `f` is not visible inside (use `$$` for recursion). | Language construct |
| `f := func< x1, ..., xn, ...: parameters \| expression>;` | Variadic short form of the function constructor. | Language construct |

*Worked examples: H2E1 (recursive functions including Fibonacci via `$$` and via named form), H2E2 (functions with optional named parameters), H2E3 (returning undefined values with `_`), H2E4 (variadic functions).*

### 2.2.2 Procedures

Procedures are like functions but do not return values. Arguments may be passed by value (`yi`) or by reference (`~yi`); only referenced arguments (and local variables) may be assigned inside a procedure.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `p := procedure(x1, ..., xn: parameters) statements end procedure;` | Creates a procedure of n ≥ 0 arguments and assigns it to `p`. Reference arguments are prefixed with `~`. Inside this form, `p` cannot be referred to by name for recursion. | Language construct |
| `procedure p(x1, ..., xn: parameters) statements end procedure;` | As above, but `p` is visible inside its own body for direct recursive calls. | Language construct |
| `p := procedure(x1, ..., xn, ...: parameters) statements end procedure;` | Variadic procedure: semantics identical to variadic functions. | Language construct |
| `procedure p(x1, ..., xn, ...: parameters) statements end procedure;` | Variadic procedure with self-reference by name. | Language construct |
| `p := proc< x1, ..., xn: parameters \| expression>;` | Short procedure constructor: `expression` must be a simple procedure call (possibly involving the arguments). Optional parameters permitted. | Language construct |
| `p := proc< x1, ..., xn, ...: parameters \| expression>;` | Variadic short form of the procedure constructor. | Language construct |

*Worked examples: H2E5 (procedure `CheckPythagoras` with a referenced argument `~h` to find Pythagorean triples).*

### 2.2.3 The forward Declaration

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `forward f;` | Forward declaration of a function or procedure `f`. Although the assignment of a value to `f` is deferred, `f` may be called from within another function or procedure defined before `f`. Must occur at the main (top) level, outside other functions or procedures. | Language construct |

*Worked examples: H2E6 (mutual recursion between `isPrime` and `primeDivisors` using `forward primeDivisors`).*

---

## 2.3 Packages

### 2.3.1 Introduction

Introductory prose (no intrinsic table). Explains the distinction between *system intrinsics* (part of Magma's C core or Magma-language standard library) and *user intrinsics* (added via the package mechanism). Key properties of intrinsics not shared by ordinary user functions: (1) signatures are stored in Magma's global table (printing a function name yields its signature summary); (2) intrinsics are compiled to Magma pseudo-code and do not need recompilation on each load. A package is a Magma source file defining constants, one or more intrinsics, and optionally ordinary helper functions; non-intrinsic contents are not visible outside the package unless explicitly imported.

### 2.3.2 Intrinsics

Argument types in the arg-list use `name::type` (pass by value) or `~name::type` / `~name` (pass by reference). The type may be a simple category name, an extended type, or one of the following type-specifier tokens:

| Specifier | Meaning |
|-----------|---------|
| `.` | Any type |
| `[ ]` | Sequence type |
| `{ }` | Set type |
| `{[ ]}` | Set or sequence type |
| `{@ @}` | Indexed set type |
| `{* *}` | Multiset type |
| `< >` | Tuple type |
| `[type]` | Sequences over type |
| `{type}` | Sets over type |
| `{[type]}` | Sets or sequences over type |
| `{@type@}` | Indexed sets over type |
| `{*type*}` | Multisets over type |

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `intrinsic name(arg-list [, ...]) [-> ret-list] {comment-text} statements end intrinsic;` | Defines a user intrinsic named `name`. `arg-list` is a comma-separated list of typed arguments (see type specifiers above). Optional `-> ret-list` (comma-separated simple types) makes the intrinsic functional; omitting it makes it procedural. `comment-text` is documentation stored with the intrinsic (use `"` to repeat comment from preceding intrinsic). If `arg-list` ends with `...` the intrinsic is variadic. Parameters (with default values) may follow a colon in the arg-list, as in ordinary functions. Intrinsics may only be defined inside package files. | Language construct — compiled to Magma pseudo-code on first attach; auto-recompiled when source changes. |

*Worked examples: H2E7 (functional intrinsic `myGCD(x::RngIntElt, y::RngIntElt) -> RngIntElt`; procedural intrinsic `Append(~Q::SeqEnum, x)`; functional intrinsic with composite argument type `IsConjugate(G::GrpPerm, R::[{ }], S::[{ }]) -> BoolElt`).*

### 2.3.3 Resolving Calls to Intrinsics

Prose section explaining overload resolution (no additional intrinsic table). When multiple intrinsics share a name (*overloaded intrinsics*), Magma resolves the call by finding, for each argument position i, the set S_i of overloads that are the best match for the type t_i of argument p_i (an overload `s` is in S_i if t_i ISA u_i and no competing overload has a strictly more specific type at position i). The actual overload called is the unique element of ∩ S_i; if the intersection is empty there is no match; if it has more than one element the call is ambiguous.

*Worked examples: H2E8 (four overloads of `overloaded(x, y)` with different combinations of `RngUPolElt` and `RngUPolElt[RngInt]` arguments, demonstrating the four resolution outcomes).*

### 2.3.4 Attaching and Detaching Package Files

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Attach(F)` | (Procedure.) Attaches the package file `F`. All intrinsics within it are incorporated into Magma. If the file is subsequently modified, it is automatically recompiled before the next statement is executed — no explicit re-attach is needed. If recompilation fails, all intrinsics of the package are removed until successful recompilation. | — |
| `Detach(F)` | (Procedure.) Detaches the package file `F`, removing all its intrinsics from the Magma session. | — |
| `freeze;` | Directive placed at the top of a package file. Marks the package as *frozen*: Magma will not check whether the source has changed between statements (avoiding overhead for stable packages). The package is still recompiled when it is explicitly re-attached if needed. | Language construct |

### 2.3.5 Related Files

Prose section (no intrinsic table). For a package source file `file.m` Magma maintains two companion files: `file.sig` (compiled signature information) and `file.lck` (lock file present during compilation). Stale `.lck` files left by crashes will cause Magma to wait indefinitely for the lock to clear; they should be deleted manually.

### 2.3.6 Importing Constants

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `import "filename": ident_list;` | Imports the identifiers in the comma-separated `ident_list` from the package file named `"filename"`. Each identifier is declared in the current scope; its value is the object assigned to the same identifier in the named file. The file must already be attached when the identifiers are used. Recommended use: sharing constants and helper functions between related package files. | Language construct |

*Worked examples: H2E9 (a `defs.m` file defining `MY_LIMIT` and `fred`; other package files importing them with `import "defs.m": MY_LIMIT, fred;`).*

### 2.3.7 Argument Checking

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `require condition: print_args;` | If the Boolean expression `condition` is false, prints `print_args` and aborts execution with the error pointer referring to the *caller* of the intrinsic (not to the intrinsic itself). `print_args` may be any expressions involving arguments or already-defined variables. | Language construct |
| `requirerange v, L, U;` | Checks that integer argument variable `v` (which must be an argument or parameter of the enclosing intrinsic) lies in the range [L, U]. If not, prints an appropriate message and aborts with the error pointer at the caller. `L` and `U` may be any integer-valued expressions. | Language construct |
| `requirege v, L;` | Checks that integer argument variable `v` satisfies v ≥ L. If not, prints an appropriate message and aborts with the error pointer at the caller. `L` may be any integer-valued expression. | Language construct |

*Worked examples: H2E10 (a `Binomial(n, k)` intrinsic using `requirege n, 0` and `requirerange k, 0, n`; a `pElement(G, p)` intrinsic using `require IsPrime(p): "Argument 2 is not prime"`).*

### 2.3.8 Package Specification Files

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AttachSpec(S)` | (Procedure.) Attaches all package files listed in the spec file `S` (a string giving a pathname). The spec file is a tree description: files and subdirectories are listed as space-separated tokens with `{ }` grouping; a token of the form `+specfile` is recursively expanded as another spec file. All pathnames in the spec file are relative to the directory containing the spec file. | — |
| `DetachSpec(S)` | (Procedure.) Detaches all package files listed in the spec file `S`. | — |

*Worked examples: H2E11 (a spec file listing `Group/{chiefseries.m, socle.m}` and `Ring/{funcs.m, Field/{galois.m}}`; calling `AttachSpec("/home/user/spec")` attaches all four files).*

### 2.3.9 User Startup Specification Files

Prose section (no intrinsic table). The environment variable `MAGMA_USER_SPEC` may be set to a colon-separated list of spec file pathnames; Magma attaches all packages listed in those spec files automatically at startup. (The variable `MAGMA_SYSTEM_SPEC` is used analogously for Magma system packages.)

*Worked examples: H2E12 (setting `MAGMA_USER_SPEC` in `.cshrc` to attach personal and shared Magma packages at startup).*

---

## 2.4 Attributes

Attributes are named fields stored within any Magma structure, accessed using the backquote (`'`) operator (analogous to record fields). There are two kinds: predefined system attributes (whose valid field names are established at Magma startup) and user-defined attributes.

### 2.4.1 Predefined System Attributes

Prose section (no additional intrinsic table). Predefined system attributes replace the older `AssertAttribute` / `HasAttribute` procedure/function pair (which remain for backward compatibility). The backquote syntax `S'Name := x` is equivalent to `AssertAttribute(S, "Name", x)`, and `assigned S'Name` / `x := S'Name` together replicate `HasAttribute(S, "Name")`. Accessing an unassigned predefined attribute raises a runtime error; use `assigned` to test first.

### 2.4.2 User-defined Attributes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `AddAttribute(C, F)` | (Procedure.) Given a category `C` and a string `F`, appends the field name `F` to the list of valid attribute field names for structures of category `C`. For use in interactive sessions only — not inside packages. Previous fields for `C` remain valid. | — |
| `declare attributes C: F1, ..., Fn;` | Package declaration directive (not a runtime statement): given a category `C` and a comma-separated list of identifiers `F1, ..., Fn`, registers those field names as valid attribute fields for category `C`. Must be used within package files. The registration is stored with the package's compiled information and takes effect when the package is attached. | Language construct |

### 2.4.3 Accessing Attributes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `S'fieldname` | Returns the current value of attribute field `fieldname` in structure `S`. Raises a runtime error if the field is not assigned or is not valid for the category of `S`. | — |
| `S''N` | As `S'fieldname` but `N` is a string variable holding the field name. Useful when the field name is computed dynamically. | — |
| `assigned S'fieldname` | Returns `true` if attribute field `fieldname` in `S` currently has a value, `false` otherwise. | — |
| `assigned S''N` | As above, with `N` a string holding the field name. | — |
| `S'fieldname := expression;` | Assigns the value of `expression` to attribute field `fieldname` of structure `S` (discarding any previous value). | — |
| `S''N := expression;` | As above, with `N` a string holding the field name. | — |
| `delete S'fieldname;` | Deletes (unassigns) attribute field `fieldname` of structure `S`. The field must currently be assigned and must be a valid user-defined attribute (not a predefined system attribute). | — |
| `delete S''N;` | As above, with `N` a string holding the field name. | — |
| `GetAttributes(C)` | Returns the valid attribute field names for category `C` as a sorted sequence of strings. | — |
| `ListAttributes(C)` | (Procedure.) Prints the valid attribute field names for category `C`. | — |

*Worked examples: H2E13 (predefined system attributes: `G'Order` for a permutation group, `C'MinimumWeight` for a code), H2E14 (interactive user attribute: `AddAttribute(GrpMat, "MyStuff")`), H2E15 (package attributes: `declare attributes GrpMat: PermRep, PermRepMap;` with caching in `PermutationRepresentation`).*

---

## 2.5 User-defined Verbose Flags

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `declare verbose F, m;` | Package directive (only valid inside package files): creates a new verbose flag named `F` (without quotes) with maximum allowable level `m` (a literal integer). Once declared, the flag is available throughout Magma via the standard `SetVerbose` / `GetVerbose` / `vprintf` / `vprint` mechanisms. | Language construct |

### 2.5.1 Examples

*Worked example: H2E15 (adding `declare verbose MyAlgorithm, 3;` to a package file, giving a three-level verbose flag `MyAlgorithm` usable anywhere in Magma).*

---

## 2.6 User-Defined Types

Since Magma V2.19, users may declare entirely new type names within packages and supply intrinsic functions for creating, printing, and operating on objects of those types. The new types are known as *user-defined types*. Typical usage: (1) declare the type; (2) define intrinsics to create objects (using `New`) and set their defining attributes; (3) define Print/Parent/IsCoercible and other primitives; (4) define further computation intrinsics.

### 2.6.1 Declaring User-Defined Types

Declarations may appear at the top level in any package file, in any order relative to the code that uses the type.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `declare type T;` | Declares `T` (without quotes) as a user-defined type name. | Language construct |
| `declare type T : P1, ..., Pn;` | Declares `T` as a user-defined type that inherits from the user-defined types `P1, ..., Pn` (each of which must be separately declared). `ISA(T, Pi)` is true for each i; an object of type `T` will match a signature argument of type `Pi`. Note: inheriting from Magma internal types or virtual types (categories) is not currently supported. | Language construct |
| `declare type T[E];` | Declares both `T` and `E` as user-defined types, and declares `E` as the *element type* of `T`: any object whose parent is of type `T` must have type `E`. This relationship is required for constructing sets and sequences with universe of type `T`. | Language construct |
| `declare type T[E] : P1, ..., Pn;` | Combination of the two previous forms: `T` and `E` are declared as user-defined types with `E` the element type of `T`, and `T` inherits from `P1, ..., Pn`. | Language construct |

### 2.6.2 Creating an Object

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `New(T)` | Creates and returns an empty object of user-defined type `T`. The object has no attributes set; the caller should set relevant attributes to define the object's properties. | Language construct |

### 2.6.3 Special Intrinsics Provided by the User

The following intrinsic signatures are special: Magma calls them automatically in specific situations. They must be provided by the user for each user-defined type `T` as appropriate.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `intrinsic Print(X::T) {Print X} ... end intrinsic;` | Required for every user-defined type `T`. Called automatically whenever an object `X` of type `T` is printed. Must be a procedure (no return value); should use `printf` without a trailing newline. Exactly one of this form or the two-argument form below must be provided. | Language construct |
| `intrinsic Print(X::T, L::MonStgElt) {Print X at level L} ... end intrinsic;` | Alternative Print intrinsic receiving a print-level string `L` ∈ `{"Default", "Minimal", "Maximal", "Magma"}`. If provided instead of the one-argument form, allows level-dependent output. Must be a procedure; use `printf` without trailing newline. | Language construct |
| `intrinsic Parent(X::T) -> . {Parent of X} ... end intrinsic;` | Required only when `T` is an element type (objects of type `T` have parents). Should return the parent of `X`; typically returns `X'Parent`. | Language construct |
| `intrinsic 'in'(e::., X::T) -> BoolElt {Return whether e is in X} ... end intrinsic;` | Required only when objects of type `T` have elements. Takes any object `e` and an object `X` of type `T`; returns whether `e` is an element of `X`. | Language construct |
| `intrinsic IsCoercible(X::T, y::.) -> BoolElt, . {Return whether y is coercible into X and the result if so} ... end intrinsic;` | Required when objects of type `T` have elements (so coercion `X!y` makes sense). Returns `true` and the coerced element if `y` can be coerced into `X`, or `false` and an error string otherwise. When provided, the coercion operator `X!y` automatically calls this intrinsic. | Language construct |

### 2.6.4 Examples

*Worked examples: H2E16 (user-defined type `MyRat` for rational numbers with `declare type MyRat`, attributes `Numer`/`Denom`, intrinsics `MyRational`, `Print`, `'+'`, `'*'`); H2E17 (parent/element type pair `DirProd[DirProdElt]` for direct products of rings, illustrating `IsCoercible`, `Parent`, and arithmetic intrinsics).*

---

## Algorithm-to-function quick reference

| Construct / mechanism | Functions / directives |
|-----------------------|------------------------|
| User function definition (full form) | `function f(...) ... end function;`, `f := function(...) ... end function;` |
| User function definition (short form) | `f := func< ... \| expr>;` |
| Variadic functions | `function f(x1, ..., xn, ...) ... end function;`, `f := func< x1, ..., xn, ... \| expr>;` |
| User procedure definition | `procedure p(...) ... end procedure;`, `p := proc< ... \| expr>;` |
| Forward declaration for mutual recursion | `forward f;` |
| Self-reference within anonymous function/procedure | `$$` |
| User intrinsic definition (package) | `intrinsic name(arg-list) [-> ret-list] {comment} ... end intrinsic;` |
| Intrinsic overload resolution | Automatic — type-based best-match intersection across all argument positions |
| Attaching / detaching single package files | `Attach(F)`, `Detach(F)` |
| Freezing stable packages | `freeze;` |
| Attaching / detaching collections of packages | `AttachSpec(S)`, `DetachSpec(S)` |
| Importing constants and helpers across packages | `import "filename": ident_list;` |
| Argument checking within intrinsics | `require condition: msg;`, `requirerange v, L, U;`, `requirege v, L;` |
| User-defined structure attributes (interactive) | `AddAttribute(C, F)` |
| User-defined structure attributes (package) | `declare attributes C: F1, ..., Fn;` |
| Accessing / assigning / deleting attributes | `S'fieldname`, `S''N`, `assigned S'fieldname`, `delete S'fieldname;` |
| Querying valid attribute fields | `GetAttributes(C)`, `ListAttributes(C)` |
| User-defined verbose flags | `declare verbose F, m;` |
| Declaring user-defined types | `declare type T;`, `declare type T : P1,...,Pn;`, `declare type T[E];`, `declare type T[E] : P1,...,Pn;` |
| Creating objects of user-defined type | `New(T)` |
| Special intrinsics for user-defined types | `Print(X::T)`, `Parent(X::T)`, `'in'(e::., X::T)`, `IsCoercible(X::T, y::.)` |
