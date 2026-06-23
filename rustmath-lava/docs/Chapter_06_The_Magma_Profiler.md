# Chapter 6 — The Magma Profiler

**Handbook part:** I — THE MAGMA LANGUAGE
**Handbook pages:** 137–143 (PDF pages 270–276)

---

## Scope and overview

One of the most important aspects of the development cycle is optimization. It is often the
case that during the implementation of an algorithm, a programmer makes erroneous assumptions
about its run-time behavior. These errors can lead to performance which differs in surprising
ways from the expected output. The unfortunate tendency of programmers to optimize code before
establishing run-time bottlenecks tends to exacerbate the problem.

Experienced programmers will thus often be heard repeating the famous mantra "Premature
optimization is the root of all evil", coined by Sir Charles A. R. Hoare, the inventor of the
Quicksort algorithm. Instead of optimizing during the initial implementation, it is generally
better to perform an analysis of the run-time behaviour of the complete program, to determine
what are the actual bottlenecks. In order to assist in this task, Magma provides a profiler,
which gives the programmer a detailed breakdown of the time spent in a program.

The Magma profiler records timing information for each function, procedure, map, and intrinsic
call made by a program. When the profiler is switched on, upon the entry and exit to each such
call the current system clock time is recorded. This information is then stored in a **call
graph**, which can be viewed in various ways.

The call graph is a directed graph, with the nodes representing the functions that were called
during the program's execution. There is an edge in the call graph from a function x to a
function y if y was called during the execution of x. Thus, recursive calls will result in
cycles in the call graph. Each node has an associated label record with fields `Name` (function
name), `Time` (total time spent in the function), and `Count` (number of times called). Each
edge ⟨x, y⟩ also has an associated label record with fields `Time` (total time spent in y when
called from x) and `Count` (total number of times y was called by x).

---

## 6.1 Introduction

This section motivates the profiler and describes its conceptual model; it contains no
intrinsics.

---

## 6.2 Profiler Basics

Basic controls for enabling, resetting, and extracting the call graph from the profiler.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetProfile(b)` | Turns profiling on (if `b` is `true`) or off (if `b` is `false`). Profiling information is stored cumulatively, which means that in the middle of a profiling run the profiler can be switched off during sections for which profiling information is not wanted. At startup the profiler is off. Turning the profiler on will slow down the execution of the program slightly. | — |
| `ProfileReset()` | Clear out all information currently recorded by the profiler. It is generally a good idea to do this after the call graph has been obtained, so that future profiling runs in the same Magma session begin with a clean slate. | — |
| `ProfileGraph()` | Get the call graph based upon the information recorded up to this point by the profiler. Returns an error if the profiler has not yet been turned on. The call graph is a directed graph; nodes represent called functions; edges point from caller to callee; recursive calls produce cycles. Node labels are records with fields `Name`, `Time`, `Count`; edge labels are records with fields `Time`, `Count`. | — |

*Worked examples: H6E1 (basic profiler use with a recursive Fibonacci implementation; illustrates `SetProfile`, `ProfileGraph`, and manual inspection of vertex/edge labels).*

---

## 6.3 Exploring the Call Graph

### 6.3.1 Internal Reports

The profiler contains report generators that present the call graph in a tabular, more
intuitive way. All reports share a common set of columns:

- **Index:** the numeric identifier for the function in the vertex list of the call graph.
- **Name:** the name of the function; followed by an asterisk if a recursive call was made through it.
- **Time:** the time spent in the function (exact meaning varies by report).
- **Count:** the number of times the function was called (exact meaning varies by report).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ProfilePrintByTotalCount(G)` | Print the list of functions in the call graph `G`, sorted in descending order by the total number of times they were called. The `Time` and `Count` fields give the total time and total call count. Parameter `Percentage` (`BoolElt`, default `false`): if `true`, `Time` and `Count` are shown as percentages of the total value. Parameter `Max` (`RngIntElt`, default −1): if non-negative, only the first `Max` entries are displayed. | — |
| `ProfilePrintByTotalTime(G)` | Print the list of functions in the call graph `G`, sorted in descending order by the total time spent in them. Apart from the sort order, behaviour is identical to `ProfilePrintByTotalCount`. Parameters: `Percentage` (`BoolElt`, default `false`), `Max` (`RngIntElt`, default −1). | — |
| `ProfilePrintChildrenByCount(G, n)` | Given a vertex `n` in the call graph `G`, print the list of functions called by function `n`, sorted in descending order by the number of times they were called by `n`. The `Time` and `Count` fields give the time spent during calls by `n` and the number of times each function was called by `n`. Parameter `Percentage` (`BoolElt`, default `false`): if `true`, fields are percentages of the total value. Parameter `Max` (`RngIntElt`, default −1): if non-negative, only the first `Max` entries are displayed. | — |
| `ProfilePrintChildrenByTime(G, n)` | Given a vertex `n` in the call graph `G`, print the list of functions called by `n`, sorted in descending order by the time spent during calls by `n`. Apart from the sort order, behaviour is identical to `ProfilePrintChildrenByCount`. Parameters: `Percentage` (`BoolElt`, default `false`), `Max` (`RngIntElt`, default −1). | — |

*Worked examples: H6E2 (continuing from H6E1; uses `ProfilePrintByTotalTime` and `ProfilePrintChildrenByTime` to inspect the Fibonacci call graph, showing recursive-call asterisk notation).*

### 6.3.2 HTML Reports

While the internal reports are useful for casual inspection of a profile run, for detailed
examination a text-based interface has serious limitations. Magma's profiler also supports the
generation of HTML reports of the profile run. The HTML report can be loaded in any web
browser. If Javascript is enabled, the tables in the report can be dynamically sorted by any
field by clicking on the column heading; clicking multiple times alternates between ascending
and descending sorts.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ProfileHTMLOutput(G, prefix)` | Given a call graph `G`, an HTML report is generated using the file prefix `prefix`. The index file of the report will be `"prefix.html"`, and exactly n additional files will be generated with the given filename prefix, where n is the number of functions in the call graph. | — |

---

## 6.4 Recursion and the Profiler

Recursive calls can cause some difficulty with profiler results. The profiler takes care to
ensure that double-counting does not occur, but this can lead to unintuitive results. When
printing the children of a recursive function with `Percentage`, the sum of the `Time` column
can exceed 100% because some time is "double counted": the total time for the first call to a
recursive function includes the time for the recursive call, which is also counted separately.
This behaviour is by design and is a consequence of the profiler's strategy for avoiding
double-counting of recursive time in the global totals. Inspecting the edge labels of the call
graph directly (via `Label(E![u,v])`) reveals the per-call-site breakdown and makes the
accounting transparent.

This section contains no additional intrinsics beyond those already listed.

*Worked examples: H6E3 (a `recursive` procedure that calls `delay` and itself; demonstrates apparent >100% totals in `ProfilePrintChildrenByTime(:Percentage)` and explains the double-counting of recursive edge time via explicit `Label` inspection).*

---

### Algorithm-to-function quick reference

| Algorithm / behaviour | Functions |
|-----------------------|-----------|
| Enable/disable profiling; cumulative timing of all function/procedure/map/intrinsic calls | `SetProfile` |
| Clear accumulated profiling data | `ProfileReset` |
| Build the call-graph data structure from accumulated profiling data | `ProfileGraph` |
| Tabular report sorted by total call count | `ProfilePrintByTotalCount` |
| Tabular report sorted by total time | `ProfilePrintByTotalTime` |
| Per-caller tabular report sorted by call count | `ProfilePrintChildrenByCount` |
| Per-caller tabular report sorted by time | `ProfilePrintChildrenByTime` |
| Dynamically sortable HTML report (one file per function + index) | `ProfileHTMLOutput` |
