# Chapter 3 — Input and Output

**Handbook part:** I — THE MAGMA LANGUAGE
**Handbook pages:** 65–91 (PDF pages 196–225)

---

## Scope and overview

Chapter 3 covers every facility Magma provides for communication between the Magma
process and its environment. The topics progress from the innermost text layer outward:

1. **Character strings** — creation, manipulation, conversion between strings and integers
   (and character codes), Boolean and ordering predicates, and string-parsing with delimiters
   and regular expressions.

2. **Printing** — the `print`, `printf`, and `fprintf` statements; verbose printing via `vprint`
   and `vprintf` conditioned on named flags; automatic printing and the previous-value buffer
   (`$1`, `$2`, …); indentation control; printing to files and to strings; and output redirection.

3. **External files** — opening files (wrapping the C `fopen`/`popen` interface), character-
   and line-level I/O (`Getc`, `Gets`, `Put`, `Puts`, `Seek`, `Tell`, `Flush`, …), reading an
   entire file as a string or binary string.

4. **Pipes** — creating a pipe to an external process (`POpen`, `Pipe`) and reading/writing
   characters or byte sequences over it.

5. **Sockets** — TCP client and server sockets (`Socket`, `WaitForConnection`), socket
   properties, and socket I/O. Available on UNIX systems only.

6. **Interactive input** — `read` / `readi` statements for runtime string and integer input.

7. **Session management** — `load`/`iload` for reading program files; `save`/`restore` for
   workspace snapshots; `SetLogFile`/`UnsetLogFile` for session logging; memory usage
   queries; system calls (including shell escape, alarm, directory operations); and `Tempname`
   for unique temporary filenames.

No algorithmic references appear in this chapter — the intrinsics are language/OS-level
constructs whose behaviour is characterised by their semantics rather than by a named algorithm.

---

## 3.1 Introduction

This section gives a prose orientation to the chapter. No intrinsics are listed.

---

## 3.2 Character Strings

Magma provides two string types. **Character strings** consist of printable keyboard characters
enclosed in double quotes; escape sequences `\"`, `\\`, `\n`, `\r`, `\t` are supported. The
double-quote `"` delimits a literal; `\` is the escape character. **Binary strings** can hold
arbitrary byte values (0–255) and are more space-efficient than sequences of integers for raw
binary data; they cannot be constructed from literals but must be created from character strings
or from binary file reads.

### 3.2.1 Representation of Strings

No intrinsics are listed in this subsection; the text describes the literal syntax.

### 3.2.2 Creation of Strings

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `"abc"` | Create a character string from a sequence of keyboard characters enclosed in double quotes. | — |
| `BinaryString(s)` / `BString(s)` | Create a binary string from the character string `s`. | — |
| `s cat t` / `s * t` | Concatenate strings `s` and `t`. | — |
| `s cat:= t` / `s *:= t` | Modification-concatenation: concatenate `s` and `t` in place, storing the result in `s`. | — |
| `&cat s` / `&* s` | Given an enumerated sequence `s` of strings, return the concatenation of all strings in `s`. | — |
| `s ^ n` | Form the `n`-fold concatenation of string `s` (for `n ≥ 0`; `n = 0` gives the empty string). | — |
| `s[i]` | Returns the substring of `s` consisting of the `i`-th character (as a length-1 string). | — |
| `s[i]` | Returns the numeric value representing the `i`-th character of `s`. (Overloaded: context determines which form is used.) | — |
| `ElementToSequence(s)` / `Eltseq(s)` | Returns the sequence of characters of `s` (as length-1 strings). | — |
| `ElementToSequence(s)` / `Eltseq(s)` | Returns the sequence of numeric values representing the characters of `s`. (Overloaded for binary strings.) | — |
| `Substring(s, n, k)` | Return the substring of `s` of length `k` starting at position `n`. | — |

*Worked examples: H3E1 (string concatenation, repetition, character indexing, `IntegerToString`, `Position`, `IsSubsequence` via sequence conversion, lexicographic comparison).*

### 3.2.3 Integer-Valued Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `#s` | The length of string `s`. | — |
| `Index(s, t)` / `Position(s, t)` | Returns the position `p` (with `0 < p ≤ #s`) in string `s` at which the substring `t` first occurs, or `0` if `t` is not a substring of `s`. The empty string always returns position `1`. | — |

### 3.2.4 Character Conversion

To perform more sophisticated string operations, one may convert the string into a sequence
and use the sequence facilities described in the next part of the manual.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `StringToCode(s)` | Returns the code number (ASCII on most UNIX machines) of the first character of string `s`. | — |
| `CodeToString(n)` | Returns a length-1 string corresponding to code number `n` (system-dependent; ASCII on most UNIX machines). | — |
| `StringToInteger(s)` | Returns the integer corresponding to the string of decimal digits `s`. All non-space characters must be digits (0–9) except optionally a leading `+` or `−`. Leading zeros are omitted. | — |
| `StringToInteger(s, b)` | Returns the integer corresponding to the string `s` of digits written in base `b`. All non-space characters must be digits less than `b` (using `A`, `B`, … for 10, 11, …) except optionally a leading `+` or `−`. | — |
| `StringToIntegerSequence(s)` | Returns the sequence of integers corresponding to the string `s` of space-separated decimal numbers. Each number may optionally begin with `+` or `−`. Leading zeros are omitted. | — |
| `IntegerToString(n)` | Convert the integer `n` into a string of decimal digits; negative `n` gives a leading `−`. | — |
| `IntegerToString(n, b)` | Convert integer `n` into a string of digits in base `b` (base must be in range 2–36); negative `n` gives a leading `−`. | — |

### 3.2.5 Boolean Functions

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `s eq t` | Returns `true` if and only if strings `s` and `t` are identical (blanks are significant). | — |
| `s ne t` | Returns `true` if and only if strings `s` and `t` are distinct (blanks are significant). | — |
| `s in t` | Returns `true` if and only if `s` appears as a contiguous substring of `t`. The empty string is contained in every string. | — |
| `s notin t` | Returns `true` if and only if `s` does not appear as a contiguous substring of `t`. | — |
| `s lt t` | Returns `true` if `s` is lexicographically less than `t` (ordering by ASCII code). | — |
| `s le t` | Returns `true` if `s` is lexicographically less than or equal to `t` (ordering by ASCII code). | — |
| `s gt t` | Returns `true` if `s` is lexicographically greater than `t` (ordering by ASCII code). | — |
| `s ge t` | Returns `true` if `s` is lexicographically greater than or equal to `t` (ordering by ASCII code). | — |

### 3.2.6 Parsing Strings

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Split(S, D)` / `Split(S)` | Given string `S` and a string `D` of separator characters, return the sequence of substrings obtained by splitting `S` at any character in `D`. If `D` is omitted, `S` is split on newlines only. To split on whitespace use `" \t\n"` for `D`. An empty field is included if `S` starts with a separator; no splitting occurs if `D` is empty. | — |
| `Regexp(R, S)` | Given a regular expression string `R` and string `S`, return whether `S` matches `R`. If so, also returns the matched substring and the sequence of substrings matched by parenthesised sub-expressions (numbered left-to-right). Based on the Henry Spencer V8 regexp reimplementation; syntax and interpretation of `|`, `*`, `+`, `?`, `^`, `$`, `[]`, `\` matches UNIX `egrep`. | — |

*Worked examples: H3E2 (elementary uses of `Split`); H3E3 (elementary uses of `Regexp`, including extraction of time fields from a date string).*

---

## 3.3 Printing

### 3.3.1 The print-Statement

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `print expression;` | Print the value of the expression. | — |
| `print expression, ..., expression;` | Print the values of a comma-separated list of expressions. | — |
| `print expression: parameters;` | Print the value of the expression at the specified print level. Levels: `Default` (same as no level), `Minimal`, `Maximal`, `Magma` (produces valid Magma input where possible). | — |

### 3.3.2 The printf and fprintf Statements

The `printf` statement prints values under control of a format string containing plain characters
and `%`-conversion specifications. Supported specifications: `%o` / `%O` (object, using default
or explicit print level), `%m` (Magma mode, equivalent to `%O` with `"Magma"`), `%h`
(hexadecimal, integers only). Field width is specified by an integer immediately after `%`:
positive = right-justified, negative = left-justified, `*` = width from next argument. No
trailing newline is printed (use `\n` in the format string).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `printf format, expression, ..., expression;` | Print values of the expressions under control of the format string `format`. | — |
| `fprintf file, format, expression, ..., expression;` | Print formatted output to `file`, which must be either a filename string (opened for appending, tilde expansion performed) or a file object opened for writing. Otherwise identical to `printf`; no trailing newline is added. | — |

*Worked examples: H3E4 (right-/left-justified and `*`-width fields); H3E5 (multiple arguments, percent literal, floating-point, `%O` with `"Magma"`); H3E6 (`fprintf` to a pipe to sort output).*

### 3.3.3 Verbose Printing (vprint, vprintf)

The following statements print output conditionally on whether a named verbose flag is active at
a sufficient level (see `SetVerbose`). Useful in package code.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `vprint flag: expression, ..., expression;` | If verbose flag `flag` is at level ≥ 1, print the expressions (as per `print`). | — |
| `vprint flag, n: expression, ..., expression;` | If verbose flag `flag` is at level ≥ `n`, print the expressions (as per `print`). | — |
| `vprintf flag: format, expression, ..., expression;` | If verbose flag `flag` is at level ≥ 1, print using `format` (as per `printf`). | — |
| `vprintf flag, n: format, expression, ..., expression;` | If verbose flag `flag` is at level ≥ `n`, print using `format` (as per `printf`). | — |

### 3.3.4 Automatic Printing

Magma allows automatic printing of expressions: a statement consisting of an expression (or
list of expressions) alone is taken as shorthand for the `print` statement. Rules:

- (a) Any single non-call-form expression followed by `;` is printed as if `print` preceded it.
- (b) A single call-form followed by `;` is dispatched as a procedure call if the first matching
  signature is procedural; otherwise its results are printed.
- (c) A comma-separated list of any expressions is printed; call-forms are treated as function
  calls only.
- (d) A print-level modifier (e.g. `: Magma`) may follow an expression list; call-forms are
  treated as function calls only.
- (e) Any printed list is placed in the previous-value buffer: `$1` is the last printed list,
  `$2` the one before, etc.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ShowPrevious()` | Show all previous values stored in the buffer. Does not alter the buffer. | — |
| `ShowPrevious(i)` | Show the `i`-th previous value stored. Does not alter the buffer. | — |
| `ClearPrevious()` | Clear all previous values stored in the buffer. | — |
| `SetPreviousSize(n)` | Set the maximum size of the previous-value buffer (default: 3). | — |
| `GetPreviousSize()` | Return the current maximum size of the previous-value buffer. | — |

*Worked examples: H3E7 (illustrating rules a–e, including `Quotrem`, `SetVerbose`, and `$1` retrieval).*

### 3.3.5 Indentation

Magma maintains a global indentation level controlling how many leading spaces are prepended
to each output line. The level is reset to 0 each time the top-level prompt is printed. Useful
for formatting recursive verbose output. The number of spaces per indentation level is
controlled by `SetIndent` and queried by `GetIndent` (default: 4 spaces per level).

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IndentPush()` | Increase (push) the indentation level by 1. | — |
| `IndentPop()` | Decrease (pop) the indentation level by 1. Error if the level is already 0. | — |

### 3.3.6 Printing to a File

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `PrintFile(F, x)` / `Write(F, x)` | Print `x` to the file named by string `F`. If the file exists, output is appended unless `Overwrite := true`, in which case the file is overwritten. Parameter: `Overwrite` (BoolElt, default `false`). | — |
| `WriteBinary(F, s)` | Write the binary string `s` to the file named by string `F`. If the file exists, output is appended unless `Overwrite := true`. Parameter: `Overwrite` (BoolElt, default `false`). | — |
| `PrintFile(F, x, L)` / `Write(F, x, L)` | Print `x` in print level `L` to the file named by string `F`. `L` must be one of `"Default"`, `"Minimal"`, `"Maximal"`, or `"Magma"`. Appends unless `Overwrite := true`. Parameter: `Overwrite` (BoolElt, default `false`). | — |
| `PrintFileMagma(F, x)` | Print `x` in Magma format to the file named by string `F`. Appends unless `Overwrite := true`. Parameter: `Overwrite` (BoolElt, default `false`). | — |

### 3.3.7 Printing to a String

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Sprint(x)` | Given any Magma object `x`, return a string containing the output obtained when `x` is printed at default level. | — |
| `Sprint(x, L)` | Given a Magma object `x` and print level string `L`, return the string obtained when `x` is printed at level `L`. | — |
| `Sprintf(F, ...)` | Given a format string `F` and corresponding arguments (as for `printf`), return the string resulting from formatted printing of `F` and the arguments. | — |

*Worked examples: H3E8 (`Sprintf` with field-width specifications to produce aligned strings).*

### 3.3.8 Redirecting Output

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetOutputFile(F)` | Redirect all Magma output to the file named by string `F`. Using `SetOutputFile(F: Overwrite := true)` empties `F` before writing. Parameter: `Overwrite` (BoolElt, default `false`). | — |
| `UnsetOutputFile()` | Close the output file and direct output back to standard output. | — |
| `HasOutputFile()` | If Magma currently has an output or log file `F`, return `true` and `F`; otherwise return `false`. | — |

---

## 3.4 External Files

Magma provides a file object type wrapping the standard C library file interface. Most standard
C library functions (`fseek`, `rewind`, `fflush`, etc.) are available as Magma intrinsics. A
file is closed by deleting the file object (using `delete` or by reassignment); there is no
explicit `Fclose`. This ensures the file is not closed while multiple references exist. The naming
follows Perl-style conventions.

### 3.4.1 Opening Files

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Open(S, T)` | Given filename string `S` and type indicator `T`, open the file named by `S` and return a Magma file object. Tilde expansion is performed on `S`. `T` has the same interpretation as for the C function `fopen()` (e.g. `"r"` for reading, `"w"` for writing). On Windows, include `"b"` in `T` for binary mode. | — |

### 3.4.2 Operations on File Objects

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Flush(F)` | Flush the buffer of file `F`. | — |
| `Tell(F)` | Return the offset in bytes of the file pointer within file `F`. | — |
| `Seek(F, o, p)` | Perform `fseek(F, o, p)`: move the file pointer of `F` to offset `o` relative to position `p` (0 = beginning, 1 = current, 2 = end). | — |
| `Rewind(F)` | Perform `rewind(F)`: move the file pointer of `F` to the beginning. | — |
| `Put(F, S)` | Write the characters of string `S` to file `F`. | — |
| `Puts(F, S)` | Write the characters of string `S` followed by a newline character to file `F`. | — |
| `Getc(F)` | Get and return one character from file `F` as a string. At end of file, a special EOF marker string is returned; test with `IsEof`. | — |
| `Gets(F)` | Get and return one line from file `F` as a string (newline removed). At end of file, returns the EOF marker string; test with `IsEof`. | — |
| `IsEof(S)` | Given a string `S`, return whether `S` is the special EOF marker. | — |
| `Ungetc(F, c)` | Given a length-1 string `c` and file `F`, perform `ungetc(c, F)`: push character `c` back into the input buffer of `F`. | — |

*Worked examples: H3E9 (line-counting function using `Open`, `Gets`, `IsEof`).*

### 3.4.3 Reading a Complete File

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Read(F)` | Return the entire contents of the text file named by string `F` as a string. | — |
| `ReadBinary(F)` | Return the entire contents of the file named by string `F` as a binary string. | — |

*Worked examples: H3E10 (using `Read` to import output from a C program; `System` to compile and run it; `StringToIntegerSequence` to parse the result).*

---

## 3.5 Pipes

Pipes allow Magma to communicate with newly-created external processes. Currently pipes are
only available on UNIX systems. The Magma I/O module is undergoing revision; current pipe
facilities are a mix of old and new methods.

When a read request is made on a pipe, the available data is returned immediately. If no data
is currently available, the process waits until some becomes available (or the pipe is closed).
Reads may return fewer characters than requested.

### 3.5.1 Pipe Creation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `POpen(C, T)` | Given a shell command line `C` and type indicator `T`, open a pipe between the Magma process and the command. Uses the C library function `popen()`: `"r"` to read from the command's stdout, `"w"` to write to the command's stdin. Returns a File object; file I/O functions (not the pipe-specific ones below) must be used with it. | — |
| `Pipe(C, S)` | Given a shell command `C` and input string `S`, create a pipe to `C`, send `S` into its standard input, and return `C`'s standard output as a string. `S` should end with a newline if it forms a single line. | — |

*Worked examples: H3E11 (using `POpen` and `Regexp` to extract current time from the UNIX `date` command).*

### 3.5.2 Operations on Pipes

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Read(P : parameters)` | Wait for data from pipe `P` and return it as a string. Parameter: `Max` (RngIntElt, default 0 = unlimited) — if positive, at most `Max` characters are returned. Returns the EOF marker string if the pipe has been closed. | — |
| `ReadBytes(P : parameters)` | Wait for data from pipe `P` and return it as a sequence of bytes (integers in range 0–255). Parameter: `Max` (RngIntElt, default 0 = unlimited). Returns the empty sequence if the pipe has been closed. | — |
| `Write(P, s)` | Write the characters of string `s` to pipe `P`. | — |
| `WriteBytes(P, Q)` | Write the bytes in byte sequence `Q` to pipe `P`. Each element of `Q` must be an integer in range 0–255. | — |

---

## 3.6 Sockets

Sockets establish communication channels between machines on the same network (TCP only in
Magma). Currently available on UNIX systems only. Data may not be instantly available;
reads take longer than file I/O. Reads return only currently available data; less than the
requested amount may be returned.

A socket is identified by a (host, port) pair. Port numbers below 1 024 are usually reserved
for system use. Two kinds of sockets exist: **client sockets** (initiate a connection) and
**server sockets** (wait for connections). Once a connection is established the distinction
becomes irrelevant for I/O.

### 3.6.1 Socket Creation

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Socket(H, P : parameters)` | Attempt to create a client socket connected to port `P` of host `H`. Parameters: `LocalHost` (MonStgElt, default none), `LocalPort` (RngIntElt, default 0) — the local binding values, usually left to the OS. | — |
| `Socket( : parameters)` | Attempt to create a server socket on the current machine. Parameters: `LocalHost` (MonStgElt, default none), `LocalPort` (RngIntElt, default 0) — if not set, values are chosen by the OS. | — |
| `WaitForConnection(S)` | For a server socket `S`: wait for a connection attempt and return a new (non-server) socket for communicating with the connecting client. `S` remains open for further connections. | — |

### 3.6.2 Socket Properties

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SocketInformation(S)` | Return identifying information for socket `S` as a pair of `<host, port>` tuples: the first is local, the second is remote. The remote tuple is undefined for server sockets. | — |

### 3.6.3 Socket Predicates

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `IsServerSocket(S)` | Return whether `S` is a server socket. | — |

### 3.6.4 Socket I/O

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Read(S : parameters)` | Wait for data from socket `S` and return it as a string. Parameter: `Max` (RngIntElt, default 0 = unlimited). Returns the EOF marker string if the socket has been closed. | — |
| `ReadBytes(S : parameters)` | Wait for data from socket `S` and return it as a sequence of bytes (integers in range 0–255). Parameter: `Max` (RngIntElt, default 0 = unlimited). Returns the empty sequence if the socket has been closed. | — |
| `Write(S, s)` | Write the characters of string `s` to socket `S`. | — |
| `WriteBytes(S, Q)` | Write the bytes in byte sequence `Q` to socket `S`. Each element must be an integer in range 0–255. | — |

*Worked examples: H3E12 (two Magma processes communicating via client/server sockets on `localhost`; `WaitForConnection`, `SocketInformation`, `Write`, `Read`, `IsEof`).*

---

## 3.7 Interactive Input

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `read identifier;` | Assign to `identifier` the string of characters entered on the following line at runtime. | — |
| `read identifier, prompt;` | As above; first prints `prompt` (a string) to solicit input. | — |
| `readi identifier;` | Assign to `identifier` the literal integer entered on the following line at runtime. | — |
| `readi identifier, prompt;` | As above; first prints `prompt` (a string) to solicit input. | — |

---

## 3.8 Loading a Program File

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `load "filename";` | Read and execute the file named by the string as Magma input. Tilde expansion of filenames is allowed. | — |
| `iload "filename";` | Interactive load: read the file named by the string as Magma input. As each line is read it is displayed; the user may skip, edit, or execute it. Tilde expansion is allowed. | — |

---

## 3.9 Saving and Restoring Workspaces

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `save "filename";` | Copy all information in the current Magma workspace to the file named by the string. The workspace is left intact. | — |
| `restore "filename";` | Copy a previously stored workspace from the named file into memory, replacing the current workspace. The computation can then continue from the point of the corresponding `save`. | — |

---

## 3.10 Logging a Session

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetLogFile(F)` | Set the log file to the file named by string `F`: all input and output is also sent to this file. If a log file is already in use, it is closed and `F` is used instead. Using `SetLogFile(F: Overwrite := true)` empties `F` before logging. Parameter: `Overwrite` (BoolElt, default `false`). | — |
| `UnsetLogFile()` | Stop logging Magma's output. | — |
| `SetEchoInput(b)` | Set to `true` or `false` whether input from external files should also be sent to standard output. | — |

---

## 3.11 Memory Usage

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `GetMemoryUsage()` | Return the current memory usage of Magma in bytes (process data size, excluding executable code). | — |
| `GetMaximumMemoryUsage()` | Return the maximum memory usage of Magma in bytes since the last reset (see `ResetMaximumMemoryUsage`). | — |
| `ResetMaximumMemoryUsage()` | Reset the recorded maximum memory usage to the current memory usage. | — |

---

## 3.12 System Calls

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Alarm(s)` | On UNIX systems: send the signal `SIGALRM` to the Magma process after `s` seconds, causing self-termination after the specified period. | — |
| `ChangeDirectory(s)` | Change to the directory specified by string `s`. Tilde expansion is allowed. | — |
| `GetCurrentDirectory()` | Return the current directory as a string. | — |
| `Getpid()` | Return Magma's process ID (value of the Unix C system call `getpid()`). | — |
| `Getuid()` | Return the user ID (value of the Unix C system call `getuid()`). | — |
| `System(C)` | Execute the system command specified by string `C` (via the C function `system()`). Returns the command's return value as an integer. On most Unix systems, divide by 256 to obtain the exit value. See also `Pipe`. | — |
| `%! shell-command` | Execute the given command in the Unix shell and return to Magma. Unlike `System`, this shell escape takes place entirely outside Magma and does not appear in Magma's history. | — |

---

## 3.13 Creating Names

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `Tempname(P)` | Given a prefix string `P`, return a unique temporary filename derived from `P` (using the C library function `mktemp()`). | — |

---

## 3.14 Bibliography

No bibliography is present in this chapter. All intrinsics in Chapter 3 are language and
operating-system-level constructs; no algorithmic references are cited.

---

### Algorithm-to-function quick reference

| Construct / category | Functions / statements |
|----------------------|------------------------|
| String literals and binary strings | `"abc"`, `BinaryString`, `BString` |
| String concatenation and repetition | `cat`, `*`, `cat:=`, `*:=`, `&cat`, `&*`, `^` |
| String indexing and subsequences | `s[i]`, `ElementToSequence`, `Eltseq`, `Substring` |
| String length and search | `#s`, `Index`, `Position` |
| Character/integer conversion | `StringToCode`, `CodeToString`, `StringToInteger`, `StringToIntegerSequence`, `IntegerToString` |
| String comparison predicates | `eq`, `ne`, `in`, `notin`, `lt`, `le`, `gt`, `ge` |
| String parsing | `Split`, `Regexp` |
| Basic output | `print`, `printf`, `fprintf` |
| Verbose/conditional output | `vprint`, `vprintf` |
| Previous-value buffer | `ShowPrevious`, `ClearPrevious`, `SetPreviousSize`, `GetPreviousSize`, `$1`, `$2`, … |
| Indentation | `IndentPush`, `IndentPop` |
| Output to files | `PrintFile`, `Write`, `WriteBinary`, `PrintFileMagma` |
| Output to strings | `Sprint`, `Sprintf` |
| Output redirection | `SetOutputFile`, `UnsetOutputFile`, `HasOutputFile` |
| File object I/O | `Open`, `Flush`, `Tell`, `Seek`, `Rewind`, `Put`, `Puts`, `Getc`, `Gets`, `IsEof`, `Ungetc` |
| Reading complete files | `Read`, `ReadBinary` |
| Pipe creation | `POpen`, `Pipe` |
| Pipe I/O | `Read`, `ReadBytes`, `Write`, `WriteBytes` |
| Socket creation | `Socket`, `WaitForConnection` |
| Socket properties and predicates | `SocketInformation`, `IsServerSocket` |
| Socket I/O | `Read`, `ReadBytes`, `Write`, `WriteBytes` |
| Interactive input | `read`, `readi` |
| Program file loading | `load`, `iload` |
| Workspace management | `save`, `restore` |
| Session logging | `SetLogFile`, `UnsetLogFile`, `SetEchoInput` |
| Memory usage | `GetMemoryUsage`, `GetMaximumMemoryUsage`, `ResetMaximumMemoryUsage` |
| System calls | `Alarm`, `ChangeDirectory`, `GetCurrentDirectory`, `Getpid`, `Getuid`, `System`, `%!` |
| Temporary filenames | `Tempname` |
