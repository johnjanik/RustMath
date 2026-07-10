# Chapter 159 — Linear Programming

**Handbook part:** XXIII — Optimization
**Handbook pages:** 5285–5291 (PDF pages 5416–5487)

---

## Scope and overview

A **Linear Program** in `n` variables `x₁, …, xₙ` with `m` constraints of the form
`Σⱼ aⱼxⱼ ≤ c` (where each relation may instead be `=` or `≥`) is represented in matrix form
as `A · x (REL) c`, with `A` an `m × n` coefficient matrix and `(REL)` a componentwise
relation (each entry `=`, `≤`, or `≥`). There is an additional **implicit constraint** that
all variables are nonnegative (`xᵢ ≥ 0`). The goal is to find a solution `(xᵢ)` that maximises
or minimises the linear objective function `Σᵢ oᵢxᵢ`.

Magma provides **two methods** for solving LP problems:

1. **Explicit LP solving functions** (§159.2) — set up suitable constraint matrices (LHS,
   relations, RHS, objective) and call a single solver function. These cover real, integer,
   and zero/one solutions, maximising or minimising.
2. **The LP process object** (§159.3–159.4) — create an instance of the `LP` process
   (category `LP`) via `LPProcess`, add constraints and set options (bounds, objective,
   integrality, max/min) incrementally, then call `Solution` to solve.

All functions that actually solve an LP return a **solution vector** together with an
**integer state code** (supplied by the `lp_solve` library):

| Code | Meaning |
|------|---------|
| 0 | Optimal Solution |
| 1 | Failure |
| 2 | Infeasible problem |
| 3 | Unbounded problem |
| 4 | Failure |

Magma supports LP problems over **Integer, Rational, and Real** rings. For Integer and Real
problems the solutions are provided as Integer and Real vectors respectively; for problems
provided over the Rationals the solution is a Real vector.

Linear programming in Magma is implemented using the **lp_solve** library written by Michel
Berkelaar (michel@ics.ele.tue.nl); the library source is at
`ftp://ftp.ics.ele.tue.nl/pub/lp_solve/`. Further reference: **[Naz87]**, **[Chv83]**,
**[OH68]**, **[NW88]**.

---

## 159.1 Introduction

Introductory section only (problem formulation, solution state codes, and supported rings, as
summarised in the overview above). No intrinsics are defined here.

---

## 159.2 Explicit LP Solving Functions

Each explicit LP solving function takes **four arguments** representing an LP problem in `n`
variables with `m` constraints:

1. `LHS` — an `m × n` matrix, the left-hand-side coefficients of the `m` constraints.
2. `relations` — an `m × 1` matrix over the same ring as `LHS`, the relation for each
   constraint: a **positive** entry means `≥`, a **zero** entry means `=`, and a **negative**
   entry means `≤`.
3. `RHS` — an `m × 1` matrix over the same ring as `LHS`, the right-hand-side values of the
   `m` constraints.
4. `objective` — a `1 × n` matrix over the same ring as `LHS`, the coefficients of the
   objective function to be optimised.

Each function returns a solution vector and an integer state code (see the state-code table in
the overview).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `MaximalSolution(LHS, relations, RHS, objective)` | The vector maximising the LP problem, plus an integer describing the state of the solution. | lp_solve LP solver **[Naz87, Chv83, OH68]**. |
| `MinimalSolution(LHS, relations, RHS, objective)` | The vector minimising the LP problem, plus an integer describing the state of the solution. | lp_solve LP solver **[Naz87, Chv83, OH68]**. |
| `MaximalIntegerSolution(LHS, relations, RHS, objective)` | The **integer** vector maximising the LP problem, plus an integer state code. | lp_solve integer/branch-and-bound solver **[NW88]**. |
| `MinimalIntegerSolution(LHS, relations, RHS, objective)` | The **integer** vector minimising the LP problem, plus an integer state code. | lp_solve integer/branch-and-bound solver **[NW88]**. |
| `MaximalZeroOneSolution(LHS, relations, RHS, objective)` | The vector with each entry either **zero or one** maximising the LP problem, plus an integer state code. | lp_solve 0/1 (binary) integer solver **[NW88]**. |
| `MinimalZeroOneSolution(LHS, relations, RHS, objective)` | The vector with each entry either **zero or one** minimising the LP problem, plus an integer state code. | lp_solve 0/1 (binary) integer solver **[NW88]**. |

*Worked examples: H159E1 (maximise `F(x,y) = 8x + 2y` subject to `10x + 21y ≤ 156`,
`2x + y ≤ 22` over the real field, building the LHS/RHS/relation/objective matrices and calling
`MaximalSolution`); H159E2 (maximise a 7-variable objective under a single knapsack-style
constraint, comparing `MaximalSolution`, `MaximalIntegerSolution`, and
`MaximalZeroOneSolution`).*

---

## 159.3 Creation of LP objects

| Intrinsic | Description |
|-----------|-------------|
| `LPProcess(R, n)` | A Linear Program over the ring `R` in `n` variables. Returns an LP process object of category `LP`. The new LP defaults to minimising the zero objective, with no constraints, no bounds, and no variables solved in integers. |

*Worked example: H159E3 (creating an LP over the real field in 2 variables and printing it,
showing the default state — minimising `[0 0]`, no constraints, no bounds, no integer
variables).*

---

## 159.4 Operations on LP objects

Functions for building up and interrogating an LP process `L` created by `LPProcess`.

| Intrinsic | Description |
|-----------|-------------|
| `AddConstraints(L, lhs, rhs)` | Add constraints to the LP problem `L`. All added constraints share the same relation, given by parameter `Rel` (type `MonStgElt`, default `"eq"`): `"eq"` for strict equality, `"le"` for less-or-equal, or `"ge"` for greater-or-equal. Constraints have the form `Σⱼ lhsᵢⱼ Rel rhsᵢ₁`, where `lhs` and `rhs` are as described in §159.2. |
| `NumberOfConstraints(L)` | The number of constraints in the LP problem `L`. |
| `NumberOfVariables(L)` | The number of variables in the LP problem `L`. |
| `EvaluateAt(L, p)` | Evaluate the objective function of the LP problem `L` at the point `p` given by a matrix. |
| `Constraint(L, n)` | The LHS, RHS and relation (`−1` for `≤`, `0` for `=`, `1` for `≥`) of the `n`-th constraint of the LP problem `L`. |
| `IntegerSolutionVariables(L)` | Sequence of indices of the variables in the LP problem `L` to be solved in integers. |
| `ObjectiveFunction(L)` | The objective function of the LP problem `L`. |
| `IsMaximisingFunction(L)` | Returns `true` if the LP problem `L` is set to maximise its objective function, `false` if set to minimise. |
| `RemoveConstraint(L, n)` | Remove the `n`-th constraint from the LP problem `L`. |
| `SetIntegerSolutionVariables(L, I, m)` | Set the variables of the LP problem `L` indexed by elements of the sequence `I` to be solved in integers if `m` is `true`, or in the usual ring if `m` is `false`. |
| `SetLowerBound(L, n, b)` | Set the lower bound on the `n`-th variable in the LP problem `L` to `b`. Note: all LP problems carry an implicit constraint that all variables are `≥ 0`; this is overridden if a lower bound is specified here (e.g. a lower bound of `−5` works as expected), but the lower bound cannot currently be completely removed. |
| `SetMaximiseFunction(L, m)` | Set the LP problem `L` to maximise its objective function if `m` is `true`, or to minimise it if `m` is `false`. |
| `SetObjectiveFunction(L, F)` | Set the objective function of the LP problem `L` to the matrix `F`. |
| `SetUpperBound(L, n, b)` | Set the upper bound on the `n`-th variable in the LP problem `L` to `b`. |
| `Solution(L)` | Solve the LP problem `L`; returns a point representing an optimal solution, and an integer representing the state of the solution. |
| `UnsetBounds(L)` | Remove any bounds on all variables in the LP problem `L`. Note: this reactivates the implicit constraint that all variables are `≥ 0`. |

*Worked example: H159E4 (maximise `F(x,y) = 3x + 13y` subject to `2x + 9y ≤ 40`,
`11x − 8y ≤ 82` using an LP process — `SetObjectiveFunction`, `AddConstraints` with
`Rel := "le"`, `SetMaximiseFunction`, then `Solution`; subsequently adding bounds on `y` with
`SetUpperBound`/`SetLowerBound`, switching variables to integers via
`SetIntegerSolutionVariables`, and removing a constraint with `RemoveConstraint`).*

---

## 159.5 Bibliography (canonical references)

| Key | Reference |
|-----|-----------|
| **[Chv83]** | V. Chvátal. *Linear Programming.* W. H. Freeman and Company, 1983. |
| **[Naz87]** | John Lawrence Nazareth. *Computer Solution of Linear Programs.* Oxford University Press, 1987. |
| **[NW88]** | G. L. Nemhauser and Laurence A. Wolsey. *Integer and Combinatorial Optimization.* John Wiley & Sons, Inc., 1988. |
| **[OH68]** | W. Orchard-Hays. *Advanced linear-programming computing techniques.* McGraw-Hill, 1968. |

---

### Algorithm-to-function quick reference

| Algorithm / theory | Functions |
|--------------------|-----------|
| lp_solve continuous LP solver (simplex) **[Naz87, Chv83, OH68]** | `MaximalSolution`, `MinimalSolution`, `Solution` (continuous variables) |
| lp_solve integer / branch-and-bound solver **[NW88]** | `MaximalIntegerSolution`, `MinimalIntegerSolution`, `Solution` (with integer variables) |
| lp_solve 0/1 (binary) integer solver **[NW88]** | `MaximalZeroOneSolution`, `MinimalZeroOneSolution` |
| LP process construction and incremental setup | `LPProcess`, `AddConstraints`, `SetObjectiveFunction`, `SetMaximiseFunction`, `SetIntegerSolutionVariables`, `SetLowerBound`, `SetUpperBound`, `UnsetBounds`, `RemoveConstraint` |
| LP process interrogation | `NumberOfConstraints`, `NumberOfVariables`, `EvaluateAt`, `Constraint`, `IntegerSolutionVariables`, `ObjectiveFunction`, `IsMaximisingFunction` |
