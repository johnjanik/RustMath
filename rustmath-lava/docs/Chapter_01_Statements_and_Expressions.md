# Chapter 1 — Statements and Expressions

**Handbook part:** I — THE MAGMA LANGUAGE
**Handbook pages:** 5–32 (PDF pages 136–165)

---

## Scope and overview

Chapter 1 provides a terse but complete overview of the foundational elements of the Magma language: how to start and terminate sessions; identifiers and the rules governing them; the various forms of assignment; Boolean logic; coercion; conditional and iterative control flow; error handling; runtime evaluation; comments; timing; type introspection; and random-number generation. These are language constructs, not mathematical algorithms; the chapter contains no theorems and only a short bibliography (two entries on the pseudo-random number generator used internally).

The language follows an expression-oriented style: most constructs (conditionals, `where`, `select ... else`, `case<...>`) have both statement and expression forms. Short-circuit evaluation applies to `and` and `or`. Mutation assignments (`o:=`) provide optimised in-place update. Generator assignment (`E<x1,...>`) is the canonical way to name generators of algebraic structures at creation time.

Random numbers since V2.7 (June 2000) use Marsaglia's Monster generator (period ≈ 10^8859) combined with the MD5 hash function (since V2.13, July 2006); the state is captured as a seed/step pair.

---

## 1.1 Introduction

Brief overview only; no intrinsics defined in this section.

---

## 1.2 Starting, Interrupting and Terminating

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `<Ctrl>-C` | Interrupt Magma while it is running (no prompt visible) to obtain a new prompt. Magma interrupts at a convenient point. If typed twice within half a second, Magma exits immediately. | Language construct |
| `quit;` | Terminate the current Magma session. | Language construct |
| `<Ctrl>-D` | Terminate the current Magma session (same effect as `quit;`). | Language construct |
| `<Ctrl>-\` | Immediately quit Magma by sending SIGQUIT to the process (Unix). Useful when `<Ctrl>-C` is unresponsive. | Language construct |

---

## 1.3 Identifiers

No intrinsics. Rules: identifiers must begin with a letter (underscore treated as a letter); followed by letters, digits, or underscores; not a reserved word; case-sensitive. A single underscore `_` is itself a reserved word. Intrinsic names conventionally begin with a capital letter (exceptions: `pCore`, `pQuotient`, etc.); they are not reserved and may be shadowed by user assignments.

---

## 1.4 Assignment

### 1.4.1 Simple Assignment

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x := e;` | Assign the value of expression `e` to identifier `x`. | Language construct |
| `x1, x2, ..., xn := e;` | Assign the first `n` of `m ≥ n` values returned by `e` to `x1, ..., xn` respectively. | Language construct |
| `:= e;` | Evaluate `e` and discard its return value(s). | Language construct |
| `assigned x` | Expression returning `true` if the local identifier `x` currently has an assigned value, `false` otherwise. Returns `false` for intrinsic function names (not local variables). | Language construct |

### 1.4.2 Indexed Assignment

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x[e1][e2]...[en] := e;` | Access and modify the entry of `x` (which must support at least `n` levels of indexing) indicated by the expressions `e1, ..., en`. Equivalent to the comma-separated form. | Language construct |
| `x[e1,e2,...,en] := e;` | Equivalent comma-separated form of indexed assignment. Most important case is (nested) sequences. | Language construct |

### 1.4.3 Generator Assignment

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `E<x1, x2, ...xn> := e;` | If the right-hand side returns a structure supporting named generators (finitely generated groups, algebras, polynomial rings, etc.), assign the first `n` generator names to `x1, ..., xn` for both printing and as variable bindings. Equivalent to `E := e; x1 := E.1; ...` plus the print-name side effect. | Language construct |
| `E<[x]> := e;` | If the right-hand side returns a structure `S` supporting named generators, assign names formed by appending `1, 2, ...` (in brackets) to the string `x`, and assign `x` to the sequence of generator names of `S`. | Language construct |
| `AssignNames(~S, [s1, ... sn])` | Procedure. For a structure `S` that supports named generators, reassign generator print-names to the strings `s1, ..., sn`. The length of the sequence must match the number of generators. Creates a new structure. | Language construct |

### 1.4.4 Mutation Assignment

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x o:= e;` | Mutation (in-place) assignment: evaluates `e`, applies operator `o` to the result and the current value of `x`, and reassigns `x`. Equivalent to (but an optimised form of) `x := x o e;`. Supported operators: `join`, `meet`, `diff`, `sdiff`, `cat`, `*`, `+`, `-`, `/`, `^`, `div`, `mod`, `and`, `or`, `xor`. | Language construct |

### 1.4.5 Deletion of Values

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `delete x` | Statement. Delete the current value of identifier `x`, freeing memory (unless other variables still reference it). If `x` previously shadowed an intrinsic function, the intrinsic is restored. Intrinsic functions themselves cannot be deleted. | Language construct |

---

## 1.5 Boolean Values

### 1.5.1 Creation of Booleans

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Booleans()` | Return the Boolean structure `B`. | Language construct |
| `#B` | Cardinality of the Boolean structure (always 2). | Language construct |
| `true` | The Boolean element true. | Language construct |
| `false` | The Boolean element false. | Language construct |
| `Random(B)` | Return a random Boolean element. | Uses the Monster pseudo-random generator |

### 1.5.2 Boolean Operators

Truth values of `and` and `or` are always evaluated left-to-right with short-circuit semantics.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x and y` | Returns `true` if both `x` and `y` are true. If `x` is `false`, `y` is not evaluated. | Short-circuit left-to-right evaluation |
| `x or y` | Returns `true` if `x` or `y` (or both) is true. If `x` is `true`, `y` is not evaluated. | Short-circuit left-to-right evaluation |
| `x xor y` | Returns `true` if exactly one of `x`, `y` is true. | Language construct |
| `not x` | Negate the truth value of `x`. | Language construct |

### 1.5.3 Equality Operators

Magma distinguishes *strong* equality (`eq`/`ne`) from *weak* equality (`cmpeq`/`cmpne`). Objects `x` and `y` are *comparable* if both are elements of a common structure `S` (via automatic coercion if necessary) for which an equality test is defined.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `x eq y` | If `x` and `y` are comparable, return `true` iff `x = y`; otherwise raise an error. Recommended for general use so type errors are caught. | Language construct |
| `x ne y` | If `x` and `y` are comparable, return `true` iff `x ≠ y`; otherwise raise an error. | Language construct |
| `x cmpeq y` | If `x` and `y` are comparable, return whether `x = y`; otherwise return `false`. Never raises an error. Use when comparing objects of completely different types where no error is desired. | Language construct |
| `x cmpne y` | If `x` and `y` are comparable, return whether `x ≠ y`; otherwise return `true`. Never raises an error. | Language construct |

### 1.5.4 Iteration

A Boolean structure `B` may be used in enumeration contexts: `for x in B do` and in set/sequence constructors `x in B`.

---

## 1.6 Coercion

Coercion is a fundamental concept in Magma: a natural mathematical mapping (embedding, projection, etc.) from one structure to another. Natural and obvious coercions are supported throughout.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `S ! x` | Attempt to coerce object `x` into structure `S`; return the result if successful. Raise an error if the coercion fails. | Structure-dependent coercion rules |
| `IsCoercible(S, x)` | Attempt to coerce `x` into `S`; return `true` and the coerced element if successful, otherwise return `false`. | Structure-dependent coercion rules |

---

## 1.7 The `where ... is` Construction

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `e1 where id is e2` | Expression that temporarily binds identifier `id` to the value of `e2`, then evaluates and returns `e1`. Scope of `id` is the `where` construction alone (unless inside an expression list or set/sequence constructor — see below). The token `:=` is a synonym for `is`. | Language construct |
| `e1 where id := e2` | Synonym for the `is` form above. | Language construct |

Semantics: `where` is left-associative, so multiple `where ... is` clauses may be chained; later bindings can refer to earlier ones. Within a set or sequence constructor, identifiers bound in `where` constructions in the predicate are visible in the left-hand expression unless the predicate is parenthesised. In an expression list (argument list, print statement, return statement, etc.), `where` extends leftward: identifiers are visible to all expressions to the left of the `where` clause; a `where` construction overrides any `where` to its right in the same list. Parentheses limit scope.

---

## 1.8 Conditional Statements and Expressions

### 1.8.1 The Simple Conditional Statement

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `if Boolean expression then statements1 else statements2 end if;` | Evaluate the Boolean expression; if true execute `statements1`, otherwise execute `statements2`. | Language construct |
| `if Boolean expression then statements end if;` | Abbreviated form with no else branch. | Language construct |
| `if Boolean expression1 then statements1 elif Boolean expression2 then statements2 else statements3 end if;` | `elif` is a convenient abbreviation for `else if`, restricting nesting level. Multiple `elif` branches are allowed. | Language construct |

### 1.8.2 The Simple Conditional Expression

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Boolean expression select expression1 else expression2` | An expression whose value is `expression1` if the Boolean is true, otherwise `expression2`. Particularly important for in-line conditionals inside set and sequence constructors. Nesting is allowed. | Language construct |

### 1.8.3 The Case Statement

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `case expression : when expression, ..., expression: statements ... else: statements end case;` | Evaluate the `case` expression. Execute the statements following the first `when` expression list that contains a matching value; then exit. If no match, execute the `else` branch (which may be omitted if no action is desired). | Language construct |

### 1.8.4 The Case Expression

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `case< expression \| expressionleft,1 : expressionright,1, ..., expressionleft,n : expressionright,n, default : expressiondef >` | Expression form of `case`. Evaluates the discriminant to value `v`; evaluates each left-hand expression in order until one equals `v`, then returns the corresponding right-hand expression. If no match, returns `expressiondef`. The `default` case is mandatory and must come last. | Language construct |

---

## 1.9 Error Handling Statements

### 1.9.1 The Error Objects

All Magma errors are of type `Err`. Error objects carry a description, location, and type.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Error(x)` | Construct an error object with user-defined payload `x` (any type). Stores `x` in `e'Object`; sets `e'Type` to `"ErrUser"`. Positional attributes (`e'Position`, `e'Traceback`) are undefined until the error is raised. | Language construct |
| `e'Position` | Attribute of error object `e`: the position (file/line) at which `e` was raised. Undefined if not yet raised. | — |
| `e'Traceback` | Attribute of error object `e`: the stack traceback at the point `e` was raised. Undefined if not yet raised. | — |
| `e'Object` | Attribute of error object `e`: the user-defined payload. For system errors, a string describing the error. | — |
| `e'Type` | Attribute of error object `e`: either `"Err"` (system error) or `"ErrUser"` (user-raised error). | — |

### 1.9.2 Error Checking and Assertions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `error e, ..., e;` | Statement. Raise an error whose description is the printed values of the given expressions. Useful for reporting illegal argument values. | Language construct |
| `error if bool, e, ..., e;` | If the Boolean expression evaluates to `true`, raise an error with the given expressions as description. Designed for precondition checking. | Language construct |
| `assert bool;` | Assertion at level 1. If the internal `Assertions` flag ≥ 1, evaluate `bool`; if false, raise an error. Recommended for important correctness checks always active in normal mode. | Language construct |
| `assert2 bool;` | Assertion at level 2. Active only when `Assertions` ≥ 2 (debug mode). For more expensive checks. | Language construct |
| `assert3 bool;` | Assertion at level 3. Active only when `Assertions` ≥ 3 (extremely stringent checking). | Language construct |

The `Assertions` flag defaults to 1. Setting it to 0 disables all assertion checks; setting to 2 enables debug checks; setting to 3 enables the most stringent checks.

### 1.9.3 Catching Errors

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `try statements1 catch e statements2 end try;` | Execute `statements1`. If no error is raised, `statements2` is skipped. If an error is raised anywhere in `statements1`, execution transfers immediately to `statements2` (the remainder of `statements1` is abandoned), and the identifier `e` is bound to the raised error object. The catch block can re-raise `e` or any other error object using `error`. | Language construct |

---

## 1.10 Iterative Statements

Three types of iterative statement: the `for` statement (definite iteration) and the `while`/`repeat` statements (indefinite iteration). Iteration may be over an arithmetic progression of integers or over any finite enumerated structure. Iterative statements may be nested; nested loops over the same structure may be written compactly (e.g., `for x, y in X do`).

### 1.10.1 Definite Iteration

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `for i := e1 to e2 by e3 do statements end for;` | Definite loop with step. Expressions must return integers `b`, `e`, `s` (begin, end, step). Loop is skipped if `s > 0` and `b > e`, or `s < 0` and `b < e`. Error if `s = 0`. Otherwise assigns `b + k·s` to `i` for `k = 0, 1, 2, ...` while `b + k·s ≤ e` (positive step) or `≥ e` (negative step). | Language construct |
| `for i := e1 to e2 do statements end for;` | Abbreviated form with step size 1. | Language construct |
| `for x in S do statements end for;` | Iterate over all elements of finite enumerated structure `S`, assigning each in turn to `x`. | Language construct |
| `for x11, ..., x1n1 in S1, ..., xm1, ..., xmnm in Sm do statements end for;` | Compact nested iteration over multiple structures; leftmost identifier corresponds to outermost loop. | Language construct |

### 1.10.2 Indefinite Iteration

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `while Boolean expression do statements end while;` | Check the Boolean expression; if true, execute `statements`. Repeat until the expression is false. | Language construct |
| `repeat statements until Boolean expression;` | Execute `statements`, then check the Boolean expression. Repeat until the expression is true, then exit the loop. (Note: condition `true` exits, opposite of `while`.) | Language construct |

### 1.10.3 Early Exit from Iterative Statements

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `continue;` | Jump to the end of the innermost enclosing loop; the termination condition is checked immediately. | Language construct |
| `continue identifier;` | In nested `for` loops, jump to the end of the loop whose variable is `identifier`; the termination condition of that loop is checked immediately. | Language construct |
| `break;` | Immediately exit the innermost enclosing loop. | Language construct |
| `break identifier;` | In nested `for` loops, immediately exit the loop whose variable is `identifier` (allows breaking out of several loops at once). | Language construct |

---

## 1.11 Runtime Evaluation: the `eval` Expression

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `eval expression` | Evaluate `expression` (must yield a string), then parse and execute that string as Magma code and return the result. The string may be either a Magma expression (no semicolon needed; result is the expression value) or a sequence of Magma statements ending in a `return` statement. The string may reference in-scope variables but cannot modify them (imported environment values are read-only). | Dynamic parse and evaluation at runtime |

---

## 1.12 Comments and Continuation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `//` | One-line comment: all text after `//` on the same line is ignored. | Language construct |
| `/* */` | Multi-line comment: all text between `/*` and `*/` is ignored. | Language construct |
| `\` | Line continuation: the backslash and the immediately following newline are ignored, allowing a logical line to span multiple physical lines. | Language construct |

---

## 1.13 Timing

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Cputime()` | Return the CPU time (real number, default precision) used since the start of the Magma session. On MS-DOS, returns real (wall-clock) time instead. | OS process CPU time query |
| `Cputime(t)` | Return the CPU time elapsed since time `t`. Time starts at 0.0 at session start. | OS process CPU time query |
| `Realtime()` | Return the absolute real (wall-clock) time as a real number: seconds since 00:00:00 GMT, 1 January 1970. On MS-DOS, returns time since session start. | OS wall-clock query |
| `Realtime(t)` | Return the real time elapsed since time `t`. | OS wall-clock query |
| `ClockCycles()` | Return the number of CPU clock cycles since Magma's startup (matches real/wall-clock time, not process time). Returns 0 if unsupported on the current processor. | CPU cycle counter (e.g. RDTSC) |
| `time statement;` | Execute `statement` and print the CPU time taken when it completes. | Wraps `Cputime()` around statement |
| `vtime flag: statement;` | If the verbose flag `flag` (set via `SetVerbose`) has level ≥ 1, execute `statement` and print time taken; if level = 0, execute without printing. | Conditional timing output |
| `vtime flag, n: statement;` | As above, but activates printing when `flag` level ≥ `n`. | Conditional timing output |

---

## 1.14 Types, Category Names, and Structures

Magma has two levels of type granularity: *coarse-grained* types (type `Cat`, e.g., `RngUPol`, `FldFin`) and *extended types* (type `ECat`, e.g., `RngUPol[RngInt]`, `Map[RngInt, FldRat]`). Extended types add a parameter and can interact with normal types in all ways.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Type(x)` / `Category(x)` | Return the (coarse-grained) type/category name of any object `x`. | — |
| `ExtendedType(x)` / `ExtendedCategory(x)` | Return the extended type/category name of any object `x`. | — |
| `ISA(T, U)` | Given types or extended types `T` and `U`, return whether `T ISA U`, i.e., whether objects of type `T` inherit properties of type `U`. Example: `ISA(RngInt, Rng)` is `true`. | Type hierarchy traversal |
| `MakeType(S)` | Given a string `S` naming a type, return the actual type object. Useful when an intrinsic name shadows the type symbol. | — |
| `ElementType(S)` | Given a structure `S`, return the type of elements of `S`. Example: `ElementType(IntegerRing())` returns `RngIntElt`. | — |
| `CoveringStructure(S, T)` | Given structures `S` and `T`, return a covering structure `C` such that both `S` and `T` embed into `C`. Raises an error if none exists. | Structure coercion lattice |
| `ExistsCoveringStructure(S, T)` | Given structures `S` and `T`, return whether a covering structure exists and, if so, return it. | Structure coercion lattice |

---

## 1.15 Random Object Generation

Pseudo-random number generation uses Marsaglia's Monster generator [Mar00] (period ≈ 10^8859, passes all Diehard tests [Mar95]), combined with MD5 hashing since V2.13. State is represented as a seed/step pair.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetSeed(s, c)` | Procedure. Reset the random number generator to initial seed `s` (0 ≤ s < 2^32) and advance to step `c` (0 ≤ c < 2^64). Equivalent to the command-line flag `-Sn` (with `c = 0`). | Monster generator initialisation **[Mar00]** |
| `SetSeed(s)` | As above with `c = 0`. | Monster generator initialisation **[Mar00]** |
| `GetSeed()` | Return the initial seed `s` and the current step `c`. Complement to `SetSeed`. Allows saving and restoring the generator state. | — |
| `Random(S)` | Return a random element of a finite set or structure `S`. | Monster generator **[Mar00]** + MD5 |
| `Random(a, b)` | Return a random integer in the closed interval [a, b] (requires a ≤ b). | Monster generator **[Mar00]** + MD5 |
| `Random(b)` | Return a random integer in [0, b] (b a non-negative integer). Calling `Random(1)` is recommended for generating random bits. | Monster generator **[Mar00]** + MD5 |

---

## 1.16 Miscellaneous

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsIntrinsic(S)` | Given a string `S`, return `true` iff an intrinsic with name `S` exists in the current Magma version. If true, also return the intrinsic itself. | — |

---

## 1.17 Bibliography

| Key | Reference |
|-----|-----------|
| **[Mar95]** | G. Marsaglia. *DIEHARD: a battery of tests of randomness.* URL: http://stat.fsu.edu/pub/diehard/, 1995. |
| **[Mar00]** | G. Marsaglia. *The Monster, a random number generator with period 10^2857 times as long as the previously touted longest-period one.* Preprint, 2000. |

---

## Quick Reference — Constructs and Functions by Category

| Category | Constructs / Functions |
|----------|----------------------|
| Session control | `<Ctrl>-C`, `quit;`, `<Ctrl>-D`, `<Ctrl>-\` |
| Assignment (simple) | `x := e;`, `x1, ..., xn := e;`, `:= e;`, `assigned x` |
| Assignment (indexed) | `x[e1]...[en] := e;`, `x[e1,...,en] := e;` |
| Assignment (generator) | `E<x1,...,xn> := e;`, `E<[x]> := e;`, `AssignNames(~S, [...])` |
| Assignment (mutation) | `x o:= e;` |
| Assignment (deletion) | `delete x` |
| Boolean creation | `Booleans()`, `true`, `false`, `Random(B)` |
| Boolean operators | `and`, `or`, `xor`, `not` |
| Equality operators | `eq`, `ne`, `cmpeq`, `cmpne` |
| Coercion | `S ! x`, `IsCoercible(S, x)` |
| Local binding | `e1 where id is e2`, `e1 where id := e2` |
| Conditional statements | `if...then...else...end if;`, `elif`, `case...when...end case;` |
| Conditional expressions | `bool select e1 else e2`, `case< expr | left:right, ..., default:def >` |
| Error objects | `Error(x)`, `e'Position`, `e'Traceback`, `e'Object`, `e'Type` |
| Error raising | `error e,...;`, `error if bool, e,...;`, `assert bool;`, `assert2 bool;`, `assert3 bool;` |
| Error catching | `try...catch e...end try;` |
| Iteration (definite) | `for i := e1 to e2 by e3 do...end for;`, `for x in S do...end for;` |
| Iteration (indefinite) | `while bool do...end while;`, `repeat...until bool;` |
| Iteration (early exit) | `continue;`, `continue id;`, `break;`, `break id;` |
| Runtime evaluation | `eval expression` |
| Comments / continuation | `//`, `/* */`, `\` |
| Timing | `Cputime()`, `Cputime(t)`, `Realtime()`, `Realtime(t)`, `ClockCycles()`, `time statement;`, `vtime flag: statement;`, `vtime flag, n: statement;` |
| Type introspection | `Type(x)`, `Category(x)`, `ExtendedType(x)`, `ExtendedCategory(x)`, `ISA(T, U)`, `MakeType(S)`, `ElementType(S)`, `CoveringStructure(S, T)`, `ExistsCoveringStructure(S, T)` |
| Random generation | `SetSeed(s, c)`, `SetSeed(s)`, `GetSeed()`, `Random(S)`, `Random(a, b)`, `Random(b)` |
| Miscellaneous | `IsIntrinsic(S)` |
