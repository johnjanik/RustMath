# Chapter 7 — Debugging Magma Code

**Handbook part:** I — THE MAGMA LANGUAGE
**Handbook pages:** 147–149 (PDF pages 280–282)

---

## Scope and overview

Magma includes a built-in command-line debugger to facilitate the debugging of complex
pieces of Magma code. The debugger is explicitly described as a prototype and can cause
Magma to crash. It is activated on error via `SetDebugOnError`, after which Magma breaks
into an interactive GDB-style prompt whenever a runtime error occurs.

The debugger syntax is modelled on the GNU GDB debugger for C programs. It operates on
a notion of **frames** — numbered stack entries, each corresponding to one function or
procedure invocation, together with all local variable definitions at that call site. Users
can navigate between frames, list source code, and evaluate arbitrary Magma expressions
in the context of any frame.

---

## 7.1 Introduction

The single intrinsic in this section enables or disables automatic entry into the debugger
on error.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetDebugOnError(f)` | If `f` is `true`, Magma will break into the debugger upon an error. If `f` is `false`, the debugger is not invoked on error. | — |

---

## 7.2 Using the Debugger

When the debugger is enabled and a runtime error occurs, Magma breaks into the
command-line debugger. The debugger commands below are modelled on the GNU GDB debugger
for C programs; acceptable abbreviations are shown in parentheses.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `backtrace` (`bt`) | Print out the stack of function and procedure calls, from the top level to the point at which the error occurred. Each line in the trace gives a single **frame**: the function or procedure that was called plus all local variable definitions for that frame. Each frame is numbered so that it can be referenced in other debugger commands. | — |
| `frame` (`f`) `n` | Change the current frame to the frame numbered `n` (the list of frames is obtained using `backtrace`). The current frame is used by other debugger commands — such as `print` — to determine the context within which expressions are evaluated. The default current frame is the top-most frame. | — |
| `list` (`l`) `[n]` | Print a source code listing for the current context (set by the `frame` command). If `n` is specified, print `n` lines of source code; the default value is 10. | — |
| `print` (`p`) `expr` | Evaluate the expression `expr` in the current context (set by the `frame` command). Has semantics identical to evaluating `eval "expr"` at the current point in the program. | — |
| `help` (`h`) | Print brief help on usage. | — |
| `quit` (`q`) | Quit the debugger and return to the Magma session. | — |

*Worked examples: H7E1 (a sample debugger session tracing a division-by-zero error in a
recursive function; demonstrates `SetDebugOnError`, `backtrace`, `frame`, and `print`).*

---

### Algorithm-to-function quick reference

| Behaviour / concept | Commands / functions |
|---------------------|----------------------|
| Enable debugger on error | `SetDebugOnError` |
| Inspect the call stack | `backtrace` (`bt`) |
| Navigate stack frames | `frame` (`f`) |
| View source context | `list` (`l`) |
| Evaluate expressions in context | `print` (`p`) |
| Inline help | `help` (`h`) |
| Exit debugger | `quit` (`q`) |
