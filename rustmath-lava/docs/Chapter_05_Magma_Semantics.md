# Chapter 5 — Magma Semantics

**Handbook part:** I — THE MAGMA LANGUAGE
**Handbook pages:** 117–134 (PDF pages 250–267)

---

## Scope and overview

This chapter describes the semantics of Magma — how expressions are evaluated, how
identifiers are treated, and how the language is structured — in a deliberately informal
way. It is both easy and essential reading. The chapter is descriptive rather than
prescriptive: it explains *how* Magma works, with little attempt to justify design
decisions. Early sections may gloss over details for the sake of learnability; full
explanations are provided as the chapter progresses.

Magma is characterised by the following properties:

- **Imperative** — values can be assigned and re-assigned to identifiers.
- **Call by value** — arguments to a function are fully evaluated before the function is
  applied (with the exceptions of `select`, `and`, and `or`, which use call-by-name /
  short-circuit evaluation).
- **Statically scoped** — free identifiers in a function expression are replaced by their
  values at the time the function expression is evaluated, not at the time the function
  is called.
- **Dynamically typed** — there is no need to declare identifier types; type violations
  are only detected when the offending code is actually executed.
- **Essentially functional** — functions are first-class values: they may be passed as
  arguments, returned from other functions, and assigned to identifiers exactly like any
  other value.

---

## 5.1 Introduction

This section situates the chapter, noting that it covers evaluation, identifier treatment,
and the key semantic properties of the language. No intrinsics.

---

## 5.2 Terminology

Establishes the vocabulary used throughout the chapter. Key definitions:

| Term | Meaning |
|------|---------|
| **expression** | A textual entity (source-code text). |
| **value** | A run-time object denoted by an expression (e.g., `1+2` and `3` are different expressions for the same value). |
| **function expression** | An expression of the form `function ... end function` (statement form) or `func< ... | ... >` (expression form). |
| **function value** | The run-time value denoted by a function expression; written `FUNC( ... : ... )` to distinguish from expressions. |
| **formal arguments** | Identifiers listed between the brackets after the `function` keyword (statement form) or before the `|` (expression form). |
| **arguments** | Expressions supplied between brackets at the call site. |
| **body** | Statements after the formal arguments (statement form) or the expression after `|` (expression form). |

An identifier is said to *occur inside* a function expression when it appears textually
anywhere in the body.

---

## 5.3 Assignment

An assignment associates an identifier with a value. A collection of such associations is
called a **context**.

| Situation | Effect |
|-----------|--------|
| Identifier not previously assigned | Added to current context (identifier is *declared*). |
| Identifier previously assigned | Value in context is updated (*re-assignment*). |

**Critical point:** re-assigning `a` does *not* retroactively change the value of any
other identifier that was computed from `a`. Each identifier's value is fixed at the
moment of its own assignment (static binding). Example: after `a := 6; b := a+7;
a := 0;` the context is `[(a,0),(b,13)]`, not `[(a,0),(b,7)]`.

---

## 5.4 Uninitialized Identifiers

Before executing a piece of code, Magma performs a semantic well-formedness check to
ensure identifiers are declared before use. Attempting to evaluate an undeclared
identifier raises an error. The checks are not exhaustive, however.

**Rule:** The right-hand side of `:=` is checked for well-formedness *before* the
left-hand side identifiers are considered declared. Thus `a := a;` is an error if `a`
was not previously declared (the `a` on the right is undeclared at that point).

---

## 5.5 Evaluation in Magma

Evaluation is the process of computing a value from an expression. Magma's evaluation
has two aspects: *when* arguments are evaluated (call by value) and *how* expressions
are reduced (identifier substitution followed by simplification).

### 5.5.1 Call by Value Evaluation

All arguments to a function are evaluated **before** the function is applied.
Operators `+`, `*`, etc. are treated as infix functions and their arguments are
likewise evaluated first. **Exceptions:** the operators `select`, `and`, and `or`
use call-by-name (short-circuit) evaluation — arguments are evaluated only as needed.
For example, `false and (4/0 eq 6)` yields `false` without evaluating `4/0`.

### 5.5.2 Magma's Evaluation Process

Expression evaluation proceeds in two steps:

1. Replace each free identifier in the expression by its value in the current context
   (all substitutions are conceptually simultaneous).
2. Simplify the resulting value to its canonical form.

**Example:** Given context `[(a,6),(b,7)]`, evaluating `c := a+b` replaces `a` → `6`,
`b` → `7`, giving `6+7`, then simplifies to `13`.

### 5.5.3 Function Expressions

Function expressions are evaluated by exactly the same substitution process as any other
expression. Free identifiers inside a function expression's body are replaced by their
current context values at the time the function expression is evaluated.

**Example:** With context `[(a,6),(b,7),(c,13)]`, evaluating `d := func< n | a+b+c+n >`
substitutes `a`, `b`, `c` immediately to yield `FUNC(n : 26+n)`. Subsequent changes to
`a`, `b`, or `c` do **not** affect `d`.

This behaviour — capturing context values at definition time — is what makes Magma
*statically scoped*.

### 5.5.4 Function Values Assigned to Identifiers

Identifiers whose values are function values are treated identically to identifiers whose
values are integers, matrices, or any other type. When such an identifier appears inside
a new function expression, it is substituted (step 1) by its function value, which may
then be simplified (step 2). Operators such as `+` are identifiers assigned function
families in the initial context and are substituted in the same way.

### 5.5.5 Recursion and Mutual Recursion

Because a function expression has no name, and because assignment rules exclude
left-hand-side identifiers from the right-hand side's scope, self-reference requires a
special mechanism:

| Construct | Purpose |
|-----------|---------|
| `$$` | Pseudo-identifier standing for the function value of the immediately enclosing function expression. Used to write recursive functions without naming them. |
| `forward f, g;` | Declares one or more identifiers as *forward references*, permitting mutual recursion: a function may call another function whose definition has not yet been evaluated. |

### 5.5.6 Function Application

Once arguments have been evaluated (call by value), applying a function value to actual
arguments proceeds by substitution:

1. Replace each formal argument in the function body by the corresponding actual argument.
2. Simplify the function body to its canonical form.

### 5.5.7 The Initial Context

When Magma starts, it has an *initial context* containing assignments of all built-in
function families to their identifiers (e.g., `+` → addition, `*` → multiplication,
etc.). Users interact with and can modify this context at the top level.

---

## 5.6 Scope

Static scoping ensures that a function value captures the values of free identifiers at
definition time. Without local declarations, an identifier appearing inside a function
body that also exists in the outer context would be substituted immediately during
evaluation of the function expression — often not the intended behaviour.

### 5.6.1 Local Declarations

A `local` declaration inside a function body designates an identifier as a *new*
identifier distinct from any outer-context identifier with the same name, whose scope is
confined to the enclosing function.

```
local temp;
```

Assignments to `local temp` inside the function do not affect the outer `temp`.

### 5.6.2 The 'first use' Rule

Explicit `local` declarations are optional: Magma implicitly treats an identifier as
local if its *first textual use* in the function body is on the left-hand side of `:=`.
"First textual use" means the first occurrence in source order, regardless of whether
that occurrence is inside a branch that is never executed at runtime.

### 5.6.3 Identifier Classes

Every identifier belongs to exactly one of three classes:

| Class | Identifiers included | May be assigned? |
|-------|----------------------|------------------|
| **Value identifier** | All loop identifiers; `$$`; any identifier whose first use inside a function body is as a value (not on the LHS of `:=` and not as an actual reference argument). | No — effectively a constant; treated as a placeholder for substitution. |
| **Variable identifier** | All identifiers declared `local` (explicitly or via the first-use rule). | Yes. |
| **Reference identifier** | Identifiers passed as actual reference arguments (prefixed with `~`); see §5.8. | Via the reference mechanism. |

### 5.6.4 The Evaluation Process Revisited

The substitution step is refined: only **free** identifiers are replaced, where an
identifier is *free* if it is a value identifier that is not a formal argument, a loop
identifier, or `$$`. This prevents Magma from substituting formal arguments with outer
context values during evaluation of a function expression.

### 5.6.5 The 'single use' Rule

An identifier may belong to only one class within a given function body. The class is
determined by its first textual use (with the right-hand side of `:=` examined before
the left-hand side). An identifier cannot be a value identifier on one occurrence and a
variable identifier on another within the same function.

---

## 5.7 Procedure Expressions

Procedures are the mechanism for changing the context *in place* rather than by
returning a new value. They complement functions when the functional style (compute and
re-assign) is unnatural or space-inefficient.

| Property | Detail |
|----------|--------|
| **Syntax (definition)** | `procedure( x, ~y ) ... end procedure;` |
| **Syntax (call)** | `p(a, ~b);` |
| **Value notation** | `PROC(x,~y : body)` analogous to `FUNC`. |
| **Reference argument** | Formal argument prefixed with `~`; the procedure may assign to it, changing the corresponding actual argument in the calling context. |
| **Value argument** | Formal argument without `~`; obeys standard substitution semantics. |
| **First-class status** | Procedures are first-class values: assignable, passable as arguments, returnable from functions. |
| **Scope** | Same rules as functions. |
| **Forward declarations** | `forward p;` supported for mutual recursion. |
| **No return value** | Procedures do not return values; their effect is entirely through reference arguments. |

---

## 5.8 Reference Arguments

When a procedure is called with an actual reference argument (e.g., `~b`), Magma
records the *name* `b`. If the corresponding formal reference argument is assigned inside
the procedure body, Magma locates the pair `(b, ...)` in the calling context and updates
its value. Reference arguments are thus synonyms for the corresponding pair in the
calling context. Value arguments, by contrast, extract only the *value* from the context;
the name is discarded.

---

## 5.9 Dynamic Typing

Magma determines types at runtime, not at parse or definition time.

| Consequence | Detail |
|-------------|--------|
| No type declarations | Identifier types need not be (and cannot be) declared. |
| Polymorphic operators | `+` in `func< a,b | a+b >` denotes a *family* of addition functions; the correct one is selected when the types of `a` and `b` are known at call time. |
| Late type-error detection | A type error in a branch that is never executed is never raised. Only reaching the offending line at runtime triggers the error. |
| Initial context | Contains assignments of built-in *function families* (not single function values) to operators like `+`, `*`, etc. |

---

## 5.10 Traps for Young Players

Two common sources of confusion arising from Magma's static scoping and call-by-value
semantics.

### 5.10.1 Trap 1

Operators are identifiers. Re-assigning an operator identifier changes its meaning for
all subsequent uses.

**Example:** After `'+' := '-';`, evaluating `1 + 2` yields `-1` because `+` now maps
to the subtraction function.

### 5.10.2 Trap 2

Because function values capture free identifiers at definition time, redefining a
function `f` does *not* affect any already-defined function `g` that used `f`: `g`
captured the *value* of `f` (the old function value), not a reference to the name `f`.
`g` must be re-evaluated (re-assigned) to pick up the new `f`.

**Example:** After `f := func< n | n+1 >; g := func< m | m + f(m) >;`, redefining
`f := func< n | n+2 >;` leaves `g` unchanged; `g(6)` still returns `13`.

---

## 5.11 Appendix A: Precedence

Operator precedence table (decreasing binding strength, top to bottom; associativity
indicated in the right column).

| Operator(s) | Associativity |
|-------------|---------------|
| `'` `''` | left |
| `(` | left |
| `[` | left |
| `assigned` | right |
| `~` | non |
| `#` | non |
| `&*` `&+` `&and` `&cat` `&join` `&meet` `&or` | non |
| `$` `$$` | non |
| `.` | left |
| `@` `@@` | left |
| `!` `!!` | right |
| `^` | right |
| unary `-` | right |
| `cat` | left |
| `*` `/` `div` `mod` | left |
| `+` `-` | left |
| `meet` | left |
| `sdiff` | left |
| `diff` | left |
| `join` | left |
| `adj` `in` `notadj` `notin` `notsubset` `subset` | non |
| `cmpeq` `cmpne` `eq` `ge` `gt` `le` `lt` `ne` | left |
| `not` | right |
| `and` | left |
| `or` `xor` | left |
| `^^` | non |
| `?` `else` `select` | right |
| `->` | left |
| `=` | left |
| `:=` `is` `where` | left |

---

## 5.12 Appendix B: Reserved Words

The following identifiers are reserved in the Magma language and cannot be used as
user-defined identifier names.

| | | | | |
|--|--|--|--|--|
| `adj` | `elif` | `is` | `require` | |
| `and` | `else` | `join` | `requirege` | |
| `assert` | `end` | `le` | `requirerange` | |
| `assert2` | `eq` | `load` | `restore` | |
| `assert3` | `error` | `local` | `return` | |
| `assigned` | `eval` | `lt` | `save` | |
| `break` | `exists` | `meet` | `sdiff` | |
| `by` | `exit` | `mod` | `select` | |
| `case` | `false` | `ne` | `subset` | |
| `cat` | `for` | `not` | `then` | |
| `catch` | `forall` | `notadj` | `time` | |
| `clear` | `forward` | `notin` | `to` | |
| `cmpeq` | `fprintf` | `notsubset` | `true` | |
| `cmpne` | `freeze` | `or` | `try` | |
| `continue` | `function` | `print` | `until` | |
| `declare` | `ge` | `printf` | `vprint` | |
| `default` | `gt` | `procedure` | `vprintf` | |
| `delete` | `if` | `quit` | `vtime` | |
| `diff` | `iload` | `random` | `when` | |
| `div` | `import` | `read` | `where` | |
| `do` | `in` | `readi` | `while` | |
| `intrinsic` | `repeat` | `xor` | | |

---

## Quick Reference

| Concept / construct | Sections | Key rule or behaviour |
|---------------------|----------|-----------------------|
| Assignment and context | 5.3 | Right-hand side evaluated before left-hand side is declared; no retroactive propagation. |
| Uninitialized identifiers | 5.4 | Checked before execution; checks are not exhaustive. |
| Call by value | 5.5.1 | Arguments evaluated before function application; `select`/`and`/`or` are short-circuit exceptions. |
| Identifier substitution | 5.5.2 | Evaluation = simultaneous substitution of free identifiers + simplification. |
| Static scoping of closures | 5.5.3 | Free identifiers in function expressions replaced at definition time. |
| Recursion (`$$`) | 5.5.5 | `$$` refers to the enclosing function value; enables anonymous recursion. |
| Mutual recursion (`forward`) | 5.5.5 | `forward f, g;` declares forward references before the function expressions are supplied. |
| Local declarations | 5.6.1 | `local x;` inside a function body confines `x` to that function. |
| First-use rule | 5.6.2 | Implicit `local` if first textual use is on the LHS of `:=`. |
| Identifier classes | 5.6.3 | Value / variable / reference; single class per identifier per function body. |
| Procedure expressions | 5.7 | `procedure( x, ~y ) ... end procedure;`; change context via reference arguments; no return value. |
| Reference arguments (`~`) | 5.8 | Actual reference argument's *name* remembered; assignment in callee updates caller's context. |
| Dynamic typing | 5.9 | Operators denote function families; type dispatch at runtime; type errors raised only on execution. |
| Operator re-assignment (Trap 1) | 5.10.1 | Assigning to `'+'` changes the meaning of `+` globally. |
| Value capture (Trap 2) | 5.10.2 | Redefining `f` does not update `g` if `g` already captured `f`'s value. |
| Precedence table | 5.11 | See §5.11 for full table (high to low). |
| Reserved words | 5.12 | See §5.12 for the complete list. |
