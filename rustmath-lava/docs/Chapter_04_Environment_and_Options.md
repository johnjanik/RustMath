# Chapter 4 — Environment and Options

**Handbook part:** I — THE MAGMA LANGUAGE
**Handbook pages:** 95–113 (PDF pages 226–247)

---

## Scope and overview

Chapter 4 describes the environmental features of Magma together with options that can be
specified at start-up on the command line or within Magma via `Set-` procedures. It covers:

1. **Command-line options** — flags that control start-up behaviour (banner suppression, startup
   file overrides, seed initialisation, workspace restoration, etc.) and the mechanism for
   passing variable assignments into Magma directly from the shell.

2. **Environment variables** — OS-level variables (e.g. `MAGMA_PATH`, `MAGMA_STARTUP_FILE`)
   that configure search paths, memory limits, library locations, help directories, and the
   temporary-file directory.

3. **Set/Get intrinsics** — a large family of paired `Set-` / `Get-` procedures and functions
   that query and control runtime parameters: IO formatting, memory, threading, logging,
   random-number seeds, verbosity, prompt appearance, and more.

4. **Verbose levels** — a uniform mechanism for enabling and querying per-module diagnostic
   output at multiple severity levels, including five user-defined slots.

5. **Information procedures** — procedures that print current state: memory usage, assigned
   identifiers, function traceback, intrinsic signatures, available categories.

6. **History system** — `%`-prefixed commands for listing, recalling, editing, and re-executing
   lines from the history buffer.

7. **Line editor** — Emacs and VI style key bindings for interactive input; toggled via
   `SetViMode`.

8. **Help system** — the `?` and `??` operators, external browser/program integration for
   online documentation.

No mathematical algorithms are involved; this is a pure language/environment chapter. The
"Algorithm" column below records behavioural semantics or is left `—` where not applicable.

---

## 4.1 Introduction

The chapter introduces the environmental features of Magma. No intrinsics are defined in
this section; it serves as an overview of what Sections 4.2–4.9 cover.

---

## 4.2 Command Line Options

When starting Magma, various command-line options can be supplied, followed by a list of
files to be automatically loaded (named as normal arguments without a `-` prefix). For each
filename, a search is conducted starting in the current directory, then in directories
specified by `MAGMA_PATH`. A startup file (specified by `MAGMA_STARTUP_FILE`, or overridden
by `-s`, or cancelled by `-n`) is loaded before any file arguments.

Arguments of the form `var:=val` (where `var` is a valid identifier and there is no space
between `var` and `:=`) assign the string `val` to `var` at the point they are processed.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `magma -b` | Suppresses the opening banner and all introductory messages; also suppresses the final "total time" message. Useful when redirecting all output to a file. | — |
| `magma -c filename` | Compiles the given package source file and exits immediately. Rarely needed since packages are automatically compiled when attached. | — |
| `magma -d` | Dumps the licence for the current magmapassfile (expiry date and valid hostids) and exits. | — |
| `magma -n` | Cancels any startup file specified by `MAGMA_STARTUP_FILE` or by `-s`. | — |
| `magma -q name` | Operates Magma as a slave (with the given name) for the MPQS integer factorisation algorithm. | — |
| `magma -r workspace` | Automatically restores the given workspace file on start-up. | — |
| `magma -s filename` | Uses the given filename as the startup file, overriding `MAGMA_STARTUP_FILE`. Should not be used for loading ordinary files (list them as plain arguments instead). | — |
| `magma -S integer` | Sets the seed for pseudo-random number generation to the specified value (range 0 to 2³² − 1 inclusive). If omitted or not followed by a number, Magma selects the seed itself. | — |

*Worked example: H4E1 — launching `magma file1 x:=abc file2`: reads startup file, loads `file1`, assigns `x := "abc"`, loads `file2`, then gives the prompt.*

---

## 4.3 Environment Variables

Environment variables are set by an appropriate operating-system command and define search
paths and other run-time options. No Magma intrinsics are introduced here; the relevant
`Set-`/`Get-` procedures appear in Section 4.4.

| Variable | Description | Algorithm |
|----------|-------------|-----------|
| `MAGMA_STARTUP_FILE` | Name of the default start-up file. Overridden by `magma -s`. | — |
| `MAGMA_PATH` | Colon-separated list of directories searched when loading files (before library directories). | — |
| `MAGMA_MEMORY_LIMIT` | Limit (in bytes) on memory that may be used by a Magma session. | — |
| `MAGMA_LIBRARY_ROOT` | Root directory for the Magma libraries (absolute path). Also settable via `SetLibraryRoot` / `GetLibraryRoot`. | — |
| `MAGMA_LIBRARIES` | Colon-separated list of sub-directories of the library root directory. Also settable via `SetLibraries` / `GetLibraries`. | — |
| `MAGMA_SYSTEM_SPEC` | Magma system spec file containing system packages automatically attached at start-up. | — |
| `MAGMA_USER_SPEC` | Personal user spec file containing user packages automatically attached at start-up. | — |
| `MAGMA_HELP_DIR` | Root directory for the Magma help files. | — |
| `MAGMA_TEMP_DIR` | Optional directory Magma uses for temporary files. Defaults to `/tmp` on Unix-like systems or the system-wide temporary directory on Windows. | — |

---

## 4.4 Set and Get

The `Set-` procedures attach values to certain environment parameters; the `Get-` functions
return their current values.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetAssertions(b)` / `GetAssertions()` | Controls assertion checking. Values: 0 = no checking, 1 = normal (default), 2 = debug, 3 = extremely stringent. | — |
| `SetAutoColumns(b)` / `GetAutoColumns()` | If enabled (default `true`), the IO system uses `ioctl()` to determine the terminal width and automatically updates the `Columns` variable when the window size changes. If disabled, `Columns` is only changed by `SetColumns`. | — |
| `SetAutoCompact(b)` / `GetAutoCompact()` | Controls automatic memory compaction between top-level statements. Compaction removes fragmentation. In rare cases where compaction becomes very slow, setting to `false` may help (at the cost of higher memory use). Default: `true`. | — |
| `SetBeep(b)` / `GetBeep()` | Controls terminal beeps. Default: `true`. | — |
| `SetColumns(n)` / `GetColumns()` | Controls the number of columns used by the IO system (affects the line editor and output word-wrapping). Setting to 0 disables word wrap. Default: 80 (unless `SetAutoColumns(true)`). | — |
| `GetCurrentDirectory()` | Returns the current working directory as a string. (Use `ChangeDirectory(s)` to change it.) | — |
| `SetEchoInput(b)` / `GetEchoInput()` | Controls whether input from external files is also sent to standard output. | — |
| `GetEnvironmentValue(s)` / `GetEnv(s)` | Returns the value of the OS environment variable `s` as a string. | — |
| `SetHistorySize(n)` / `GetHistorySize()` | Controls the number of lines saved in the history buffer. Setting to 0 preserves no history. | — |
| `SetIgnorePrompt(b)` / `GetIgnorePrompt()` | If enabled, leading `>` characters (possibly separated by whitespace) are ignored by the history system when reading from a terminal, allowing pasting of previously prompted output. Default: `false`. | — |
| `SetIgnoreSpaces(b)` / `GetIgnoreSpaces()` | Controls whether spaces are ignored during history search in the line editor (when using `<Ctrl>-P` or `<Ctrl>-N` with a non-empty prefix). Default: `true`. | — |
| `SetIndent(n)` / `GetIndent()` | Controls the indentation level for formatted output. Default: 4. | — |
| `SetLibraries(s)` / `GetLibraries()` | Controls the Magma library directories via `MAGMA_LIBRARIES`. `SetLibraries` takes a colon-separated string of sub-directories within the library root; `GetLibraries` returns the current value. MAGMA_PATH directories are searched first. | — |
| `SetLibraryRoot(s)` / `GetLibraryRoot()` | Controls the root directory for the Magma libraries via `MAGMA_LIBRARY_ROOT`. `SetLibraryRoot` takes an absolute pathname; `GetLibraryRoot` returns the current value. | — |
| `SetLineEditor(b)` / `GetLineEditor()` | Controls the line editor. Default: `true`. | — |
| `SetLogFile(F)` / `UnsetLogFile()` | Sets the log file to the file specified by string `F`; all input and output are sent to this file as well as the terminal. If a log file is already open it is closed. Parameter `Overwrite` (BoolElt, default `false`): if `true`, the file is truncated before writing; otherwise output is appended. `UnsetLogFile()` stops logging. | — |
| `SetMemoryLimit(n)` / `GetMemoryLimit()` | Sets the limit (in bytes) on memory the memory manager may allocate. 0 means no limit. Default: 0. | — |
| `SetNthreads(n)` / `GetNthreads()` | Sets the number of threads for multi-threaded algorithms (when POSIX threads are enabled). Currently affects `MinimumWeight` (coding theory) and the F4 Gröbner basis algorithm for medium-sized primes (`Groebner`). | — |
| `SetOutputFile(F)` / `UnsetOutputFile()` | Starts/stops redirecting all Magma output to the file specified by string `F`. Parameter `Overwrite` (BoolElt, default `false`): if `true`, the file is truncated before writing. | — |
| `SetPath(s)` / `GetPath()` | Controls the file-search path: a colon-separated list of directories searched in order (`.` is implicitly prepended). Tilde expansion is applied to each directory. May be overridden by `MAGMA_PATH`. | — |
| `SetPrintLevel(l)` / `GetPrintLevel()` | Controls the global printing level: one of `"Minimal"`, `"Magma"`, `"Maximal"`, `"Default"`. Default: `"Default"`. | — |
| `SetPrompt(s)` / `GetPrompt()` | Controls the terminal prompt string. The following `%` escapes are expanded: `%%` → the character `%`; `%h` → the current history line number; `%S` → the full parser state stack (words like `"if"`, `"while"` for incomplete statements); `%s` → only the topmost parser state word. Default: `"%S> "`. | — |
| `SetQuitOnError(b)` | Sets whether Magma should completely quit on any error (syntax, runtime, etc.). Default: `false`. | — |
| `SetRows(n)` / `GetRows()` | Controls the number of rows per page in the IO system. If 0, paging is disabled; otherwise a prompt is given after `n` rows. Default: 0. | — |
| `GetTempDir()` | Returns the directory Magma uses for temporary files (influenced at startup by `MAGMA_TEMP_DIR`). | — |
| `SetTraceback(n)` / `GetTraceback()` | Controls whether Magma produces a traceback of user function calls before each error message. Default: `true`. | — |
| `SetSeed(s, c)` / `GetSeed()` | Controls the initialisation seed and step number for pseudo-random number generation. See the chapter on statements and expressions for details. | — |
| `GetVersion()` | Returns integers `x`, `y`, and `z` such that the current version of Magma is V`x`.`y`-`z`. | — |
| `SetViMode(b)` / `GetViMode()` | Controls the line editor style: `false` = Emacs (default), `true` = VI. | — |

---

## 4.5 Verbose Levels

Verbose printing provides information on computations performed within various Magma
modules. For each verbose flag the verbosity may have different levels; the default is level 0
for each flag. There are also 5 slots for user-defined verbose flags: `"User1"` through
`"User5"`, settable via `SetVerbose("Usern", true)`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetVerbose(s, i)` / `SetVerbose(s, b)` | Sets the verbose level for flag `s` to integer level `i`, or to 0 / 1 via boolean `b` (`false` = 0, `true` = 1). `s` must be a string naming a valid verbose flag. | — |
| `GetVerbose(s)` | Returns the current value of verbose flag `s` as an integer. | — |
| `IsVerbose(s)` | Returns whether the verbose flag `s` is non-zero. | — |
| `IsVerbose(s, l)` | Returns whether the verbose flag `s` has value greater than or equal to `l`. | — |
| `ListVerbose()` | Prints each verbose flag together with its maximum level. | — |
| `ClearVerbose()` | Sets the level of all verbose flags to 0. | — |

---

## 4.6 Other Information Procedures

The following procedures print information about the current state of Magma.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `ShowMemoryUsage()` | (Procedure.) Shows Magma's current memory usage. | — |
| `ShowIdentifiers()` | (Procedure.) Lists all identifiers that have been assigned to. | — |
| `ShowValues()` | (Procedure.) Lists all identifiers that have been assigned to, together with their values. | — |
| `Traceback()` | (Procedure.) Displays a traceback of the current Magma function invocations. | — |
| `ListSignatures(C)` | Lists all intrinsic functions, procedures, and operators that have objects from category `C` among their arguments or return values. Parameters: `Isa` (BoolElt, default `true` — also consider categories that `C` inherits from); `Search` (MonStgElt, default `"Both"`, valid values `"Both"`, `"Arguments"`, `"ReturnValues"`); `ShowSrc` (BoolElt, default `false` — show where package intrinsics are defined). | — |
| `ListSignatures(F, C)` | Given an intrinsic `F` and category `C`, lists all signatures of `F` that match category `C` among their arguments or return values. Same parameters as above. | — |
| `ListCategories()` / `ListTypes()` | Procedure to list the abbreviated names for all available categories in Magma. | — |

---

## 4.7 History

Magma provides a history system allowing recall and editing of previous lines. The history
system is invoked by typing commands beginning with the history character `%`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `%p` | Lists the full history buffer, each line preceded by its history line number. | — |
| `%pn` | Lists history line `n` in `%p` format. | — |
| `%pn1 n2` | Lists history lines in the range `n1` to `n2` in `%p` format. | — |
| `%P` | Lists the full history buffer without the initial line numbers. | — |
| `%Pn` | Lists history line `n` in `%P` format. | — |
| `%Pn1 n2` | Lists history lines in the range `n1` to `n2` in `%P` format. | — |
| `%s` | Lists the full history buffer, with an initial statement before each line that resets the random-number seed to the value it had just before that line was executed. | — |
| `%sn` | Prints history line `n` in `%s` format. | — |
| `%sn1 n2` | Prints history lines in the range `n1` to `n2` in `%s` format. | — |
| `%S` | As for `%s`, but the seed-reset statement is only printed if the seed changed since the previous time it was printed, and not if it would appear in the middle of a statement (i.e. if the last line did not end in a semicolon). | — |
| `%Sn` | Prints history line `n` in `%S` format. | — |
| `%Sn1 n2` | Prints history lines in the range `n1` to `n2` in `%S` format. | — |
| `%` | Re-enters the last line into the input stream. | — |
| `%n` | Re-enters the line specified by line number `n` into the input stream. | — |
| `%n1 n2` | Re-enters history lines in the range `n1` to `n2` into the input stream. | — |
| `%e` | Edits the last line using the editor named by the `EDITOR` environment variable (or `/bin/ed` if unset). If the file is unchanged after editing, nothing is done; otherwise the new contents are re-entered into the input stream. | — |
| `%en` | Edits the line specified by line number `n`. | — |
| `%en1 n2` | Edits history lines in the range `n1` to `n2`. | — |
| `%! shell-command` | Executes the given command in the Unix shell, then returns to Magma. | — |

---

## 4.8 The Magma Line Editor

Magma provides a line editor with both Emacs and VI style key bindings. VI style is enabled
with `SetViMode(true)` and reverted with `SetViMode(false)`. The default is Emacs style
(`SetViMode(false)`). Many key bindings are shared between modes because some VI users
prefer to retain certain Emacs keys (like `<Ctrl>-P`) in insert mode.

### 4.8.1 Key Bindings (Emacs and VI mode)

`<Ctrl>-key` means hold down the Control key and press `key`.

| Key | Description | Algorithm |
|-----|-------------|-----------|
| `<Return>` | Accept the current line and print a new line. Works in any mode. | — |
| `<Backspace>` / `<Delete>` | Delete the previous character. | — |
| `<Tab>` | Complete the word under or just after the cursor. If no unique completion exists, expand to the common prefix; a second Tab lists all possible completions. Completes system functions, procedures, parameters, reserved words, and user identifiers. | — |
| `<Ctrl>-A` | Move to the beginning of the line. | — |
| `<Ctrl>-B` | Move back a character. | — |
| `<Ctrl>-C` | Abort the current line and start a new line. | — |
| `<Ctrl>-D` | On an empty line: send EOF (exit at the top level). At end of line: list completions. Otherwise: delete the character under the cursor. | — |
| `<Ctrl>-E` | Move to the end of the line. | — |
| `<Ctrl>-F` | Move forward a character. | — |
| `<Ctrl>-H` | Same as `<Backspace>`. | — |
| `<Ctrl>-I` | Same as `<Tab>`. | — |
| `<Ctrl>-J` | Same as `<Return>`. | — |
| `<Ctrl>-K` | Delete all characters from the cursor to the end of the line. | — |
| `<Ctrl>-L` | Redraw the line on a new line (useful if the screen is disrupted by other programs). | — |
| `<Ctrl>-M` | Same as `<Return>`. | — |
| `<Ctrl>-N` | Go forward a line in the history buffer. If the cursor is not at the beginning, search forward to the first line starting with the same prefix (ignoring spaces if `SetIgnoreSpaces` is on). If used at a new line after recalling a preceding line, enters the next line after the recalled one. | — |
| `<Ctrl>-P` | Go back a line in the history buffer. If the cursor is not at the beginning, search backward to the first line starting with the same prefix (ignoring spaces if `SetIgnoreSpaces` is on). | — |
| `<Ctrl>-U` | Clear the whole of the current line. | — |
| `<Ctrl>-Vchar` | Insert the following character literally. | — |
| `<Ctrl>-W` | Delete the previous word. | — |
| `<Ctrl>-X` | Same as `<Ctrl>-U`. | — |
| `<Ctrl>-Y` | Insert the contents of the yank-buffer before the character under the cursor. | — |
| `<Ctrl>-Z` | Stop Magma. | — |
| `<Ctrl>-_` | Undo the last change. | — |
| `<Ctrl>-\` | Immediately quit Magma. | — |

On most systems the arrow keys also have the obvious meaning.

### 4.8.2 Key Bindings in Emacs Mode Only

`Mkey` means press the Meta key (currently only the Esc key) and then `key`.

| Key | Description | Algorithm |
|-----|-------------|-----------|
| `Mb` / `MB` | Move back a word. | — |
| `Mf` / `MF` | Move forward a word. | — |

### 4.8.3 Key Bindings in VI Mode Only

In VI mode the line editor has two sub-modes: insert mode (non-control characters are
inserted at the cursor position) and command mode (entered by pressing Esc). Command mode
accepts the following range specifiers, optionally preceded by a repeat count:

| Range | Description | Algorithm |
|-------|-------------|-----------|
| `0` | Move to the beginning of the line. | — |
| `$` | Move to the end of the line. | — |
| `<Ctrl>-space` | Move to the first non-space character of the line. | — |
| `%` | Move to the matching bracket (bracket characters: `(`, `)`, `[`, `]`, `{`, `}`, `<`, `>`). | — |
| `;` | Move to the next character (used with `F`, `f`, `T`, `t`). | — |
| `,` | Move to the previous character (used with `F`, `f`, `T`, `t`). | — |
| `B` | Move back a space-separated word. | — |
| `b` | Move back a word. | — |
| `E` | Move forward to the end of the space-separated word. | — |
| `e` | Move forward to the end of the word. | — |
| `Fchar` | Move back to the first occurrence of `char`. | — |
| `fchar` | Move forward to the first occurrence of `char`. | — |
| `h` / `H` | Move back a character. | — |
| `l` / `L` | Move forward a character. | — |
| `Tchar` | Move back to just after the first occurrence of `char`. | — |
| `tchar` | Move forward to just before the first occurrence of `char`. | — |
| `w` | Move forward a space-separated word. | — |
| `W` | Move forward a word. | — |

The following keys are available in command mode:

| Key | Description | Algorithm |
|-----|-------------|-----------|
| `A` | Move to end of line and switch to insert mode. | — |
| `a` | Move forward one character (unless already at end) and switch to insert mode. | — |
| `C` | Delete to end of line and switch to insert mode. | — |
| `crange` | Delete to the specified range and switch to insert mode. | — |
| `D` | Delete to end of line. | — |
| `drange` | Delete to the specified range. | — |
| `I` | Move to the first non-space character and switch to insert mode. | — |
| `i` | Switch to insert mode. | — |
| `j` | Go forward a line in the history buffer (same as `<Ctrl>-N`). | — |
| `k` | Go back a line in the history buffer (same as `<Ctrl>-P`). | — |
| `P` | Insert the contents of the yank-buffer before the character under the cursor. | — |
| `p` | Insert the contents of the yank-buffer before the character after the cursor. | — |
| `R` | Enter over-type mode: typed characters replace old characters without insertion. Esc returns to command mode. | — |
| `rchar` | Replace the character under the cursor with `char`. | — |
| `S` | Delete the whole line and switch to insert mode. | — |
| `s` | Delete the current character and switch to insert mode. | — |
| `U` / `u` | Undo the last change. | — |
| `X` | Delete the character to the left of the cursor. | — |
| `x` | Delete the character under the cursor. | — |
| `Y` | Copy the whole line into the yank-buffer. | — |
| `yrange` | Copy all characters from the cursor to the specified range into the yank-buffer. | — |

---

## 4.9 The Magma Help System

Magma provides extensive online help facilities. The command `magmahelp` launches a browser
on the main documentation page. Typing an intrinsic name alone (e.g. `FundamentalUnit;`)
lists all its signatures. The `?` operator retrieves detailed documentation — either via the
internal text-based help browser or via a configured external browser / program. Setting
`SetVerbose("Help", true)` causes Magma to display the exact command used and the return
value obtained. The internal help browser is entered with `??`.

| Intrinsic | Description | Algorithm |
|-----------|-------------|-----------|
| `SetHelpExternalBrowser(S, T)` / `SetHelpExternalBrowser(S)` | Defines the external browser command to use when `SetHelpUseExternalBrowser(true)` is in effect. The string `S` must be a valid command taking exactly one `%s` argument (replaced by a URL). When two strings are provided, `T` is a fallback (e.g. try an already-running browser, then start a new one). | — |
| `SetHelpUseExternalBrowser(b)` | Tells Magma to use (or stop using) the external browser. If both `SetHelpUseExternalSystem` and `SetHelpUseExternalBrowser` are set to `true`, the most recent assignment takes effect. | — |
| `SetHelpExternalSystem(s)` | Tells Magma to use a user-defined external program for help. The string must contain exactly one `%s` (replaced by the argument to `?`); the result must be a valid command. | — |
| `SetHelpUseExternalSystem(b)` | Tells Magma to use (or stop using) the external help system. If both `SetHelpUseExternalSystem` and `SetHelpUseExternalBrowser` are set to `true`, the most recent assignment takes effect. | — |
| `GetHelpExternalBrowser()` | Returns the currently configured browser command strings. | — |
| `GetHelpExternalSystem()` | Returns the currently configured external help system command string. | — |
| `GetHelpUseExternal()` | Returns two values: the current setting from `SetHelpUseExternalBrowser` and the current setting from `SetHelpUseExternalSystem`. | — |

### 4.9.1 Internal Help Browser

The internal help browser is entered with `??` at the Magma prompt. It provides a powerful
text-based interface to the Magma documentation. No additional intrinsics beyond `??` are
defined in this subsection.

---

## Algorithm-to-Function Quick Reference

| Category / Feature | Functions / Constructs |
|--------------------|----------------------|
| Command-line start-up control | `magma -b`, `magma -c`, `magma -d`, `magma -n`, `magma -q`, `magma -r`, `magma -s`, `magma -S` |
| OS environment variables | `MAGMA_STARTUP_FILE`, `MAGMA_PATH`, `MAGMA_MEMORY_LIMIT`, `MAGMA_LIBRARY_ROOT`, `MAGMA_LIBRARIES`, `MAGMA_SYSTEM_SPEC`, `MAGMA_USER_SPEC`, `MAGMA_HELP_DIR`, `MAGMA_TEMP_DIR` |
| IO formatting (columns, rows, indent, print level) | `SetColumns`, `GetColumns`, `SetAutoColumns`, `GetAutoColumns`, `SetRows`, `GetRows`, `SetIndent`, `GetIndent`, `SetPrintLevel`, `GetPrintLevel` |
| Memory management | `SetMemoryLimit`, `GetMemoryLimit`, `SetAutoCompact`, `GetAutoCompact`, `ShowMemoryUsage` |
| Threading | `SetNthreads`, `GetNthreads` |
| Logging and output redirection | `SetLogFile`, `UnsetLogFile`, `SetOutputFile`, `UnsetOutputFile` |
| Library and path configuration | `SetPath`, `GetPath`, `SetLibraries`, `GetLibraries`, `SetLibraryRoot`, `GetLibraryRoot` |
| Random-number seeds | `SetSeed`, `GetSeed`, `magma -S` |
| Prompt and line editor | `SetPrompt`, `GetPrompt`, `SetViMode`, `GetViMode`, `SetLineEditor`, `GetLineEditor`, `SetBeep`, `GetBeep` |
| Assertions | `SetAssertions`, `GetAssertions` |
| Input/output behaviour | `SetEchoInput`, `GetEchoInput`, `SetIgnorePrompt`, `GetIgnorePrompt`, `SetIgnoreSpaces`, `GetIgnoreSpaces`, `SetQuitOnError` |
| Directory and environment queries | `GetCurrentDirectory`, `GetEnvironmentValue`, `GetEnv`, `GetTempDir`, `GetVersion` |
| History settings | `SetHistorySize`, `GetHistorySize` |
| Traceback | `SetTraceback`, `GetTraceback`, `Traceback` |
| Verbose flags | `SetVerbose`, `GetVerbose`, `IsVerbose`, `ListVerbose`, `ClearVerbose` |
| Introspection | `ShowIdentifiers`, `ShowValues`, `ListSignatures`, `ListCategories`, `ListTypes` |
| History recall and editing | `%p`, `%pn`, `%pn1 n2`, `%P`, `%Pn`, `%Pn1 n2`, `%s`, `%sn`, `%sn1 n2`, `%S`, `%Sn`, `%Sn1 n2`, `%`, `%n`, `%n1 n2`, `%e`, `%en`, `%en1 n2`, `%! shell-command` |
| Line editor (shared keys) | `<Return>`, `<Backspace>`, `<Delete>`, `<Tab>`, `<Ctrl>-A/B/C/D/E/F/H/I/J/K/L/M/N/P/U/V/W/X/Y/Z`, `<Ctrl>-_`, `<Ctrl>-\` |
| Line editor (Emacs only) | `Mb`, `MB`, `Mf`, `MF` |
| Line editor (VI only — ranges) | `0`, `$`, `<Ctrl>-space`, `%`, `;`, `,`, `B`, `b`, `E`, `e`, `Fchar`, `fchar`, `h`, `H`, `l`, `L`, `Tchar`, `tchar`, `w`, `W` |
| Line editor (VI only — commands) | `A`, `a`, `C`, `crange`, `D`, `drange`, `I`, `i`, `j`, `k`, `P`, `p`, `R`, `rchar`, `S`, `s`, `U`, `u`, `X`, `x`, `Y`, `yrange` |
| Help system | `SetHelpExternalBrowser`, `SetHelpUseExternalBrowser`, `SetHelpExternalSystem`, `SetHelpUseExternalSystem`, `GetHelpExternalBrowser`, `GetHelpExternalSystem`, `GetHelpUseExternal` |
