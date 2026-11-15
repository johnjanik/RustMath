# S1-symbolic, revision 7rt1: red team 2 (input structure and recursion)

Design attacked: `fu2/design/S1-symbolic.md`, revision 7 red-team round 1 (all of it, with §2.5,
§2.11, §2.12, §2.14, §2.15, §2.16 and §6 step 8 in detail). Prototype: `EXP/rev7rt1/proto9/`
(EXP = `fu2/design/S1-symbolic`), its binaries `bin/split` and `bin/split_rel` (HEAD's parser), and
one probe of mine built against the same HEAD rlibs.

My probe is `EXP/redteam2-structure/src/rt2.rs` (`build.sh debug|release src/rt2.rs rt2`, binaries
`bin/rt2` and `bin/rt2_rel`). It links proto9's `api.rs`, `konst.rs`, `ser.rs`, `ball.rs` and
`display.rs` unchanged. In front of them it has a small parser that follows §2.12: sign runs, the
`* /` and `+ -` chains by loops, `a/b` an operator with the literal fold, `exponent := ('-' | '+')*
power`, unsigned literals of any size. So text reaches the engine in the structure the port's
parser gives it (`HEADPARSE=1` uses HEAD's parser instead). Modes: `parse`, `engine` (parse on a
1 GiB thread, the entry point on a STACK thread), `both` (one thread), `head` (HEAD's `taylor`).
`ENTRY=se|lim` as in `split.rs`; `PRINT`, `EQ_TOO`, `GPCHECK` and `GPFORM` print the result, compare
it with a clone, or write it for gp.

Scripts in `EXP/redteam2-structure/`: `r.py` (one run with a 4 GiB address-space cap and a 60 s
timeout), `bs.py` (binary search of the deepest input handled), `gen.py` (the shape generators),
`gpcheck.py` (differential check against gp), `randgen.py` (random inputs in the §2.12 grammar).
Inputs are in `inputs/`, outputs in `out/`. Budget 1e8 (`TAYLOR_WORK`, or `LIMIT_SERIES_WORK` on the
limit path). "Debug 2 MB" is the debug build on a 2 MiB thread, "release 8 MB" the release build on
an 8 MiB thread.

## H1 (over the bar, (c)): mixed nesting overflows the engine before the parser on a 2 MB debug thread

**What happens.** §2.5 compares the parser and the engine per parenthesis only on shapes that nest
one construct (calls, negations, right-deep sums, `1/(1 + ...)`, `((x*2+1)*2+1)`). Between two
parentheses, §2.12's grammar has up to six levels: a `+ -` chain, a `* /` chain, a negation, a
tower, a negated exponent, then the atom. HEAD's nom parser pays about the same stack per
parenthesis whichever of these levels the text uses, because every parenthesis goes through all of
its rules. The engine pays one dispatcher frame and one chain frame for each level that is present.
The debug frames are large: `konst_of_inner` 2328 bytes, `konst_of` 568, `konst_pow_chain` 1784,
`konst_product_chain` 1224, `konst_sum_chain` 936, `expand_product` 2008, `expand_sum` 1592,
`expand_pow_chain` 1576, `expand` 1192 (`sub rsp` of `bin/rt2`). A text that uses every level in each
parenthesis costs the engine 15.5 to 18.9 KB per parenthesis in debug. HEAD's parser costs 9.7 KB.

The gdb backtrace of the overflow repeats `konst_of_inner, konst_of, konst_pow_chain, konst_of_inner,
konst_of, konst_of_inner, konst_of, konst_product_chain, konst_of_inner, konst_of, konst_sum_chain`:
eleven frames per parenthesis.

**Inputs** (`gen.py`; deepest handled by `bs.py`, then the first failure):

| shape | point, order | parser reads, debug 2 MB | engine handles, debug 2 MB | parser reads, release 8 MB | engine handles, release 8 MB | HEAD's taylor, debug 2 MB |
|---|---|---|---|---|---|---|
| `mix5cs`: `x + -(-(...(1)^2*3+4)...)^2*3+4` (constant, `from_expr`) | 0, 1 | 216 (HEAD's parser) | 132 | 4123 | 5133 | 45 to 49 |
| `mix5`: `-(-(...(x)^x*x+2*x)...)^x*x+2*x` (series, `expand`) | 1, 1 | 216 | 173 | 4123 | 5454 | 40 to 59 |
| `mix4`: `((...(x)^x*x+x)...)^x*x+x` | 1, 1 | 216 | 192 | 4123 | 6159 | |
| `mix5cdiv`: as `mix5cs` with `/3` | 0, 1 | 216 | 132 | | | |
| `mix6c`: `x + -2^-(-2^-(...(1))*3+4)*3+4` (needs §2.12's `^-`) | 0, 1 | 266 (my §2.12 parser) | 111 | 4438 | 4436 | |
| `mix6`: `-x^-(-x^-(...(x))*x+2*x)*x+2*x` | 1, 1 | 266 | 168 | 4438 | 5690 | |

Reproductions (each aborts with "fatal runtime error: stack overflow", rc -6):

- `EXP/rev7rt1/bin/split both inputs/mix5cs_133.txt 0 1 2097152` (1200 bytes, the prototype's own
  driver and HEAD's parser on one 2 MB debug thread): abort after 7 ms, VmHWM about 6 MB. At 132
  levels (1191 bytes) it is `Ok` in 3.5 ms with 1.5e5 units. `split parse inputs/mix5cs_216.txt`
  parses on the same thread size. `ENTRY=se` and `ENTRY=lim` abort the same way at 150 and 200
  levels.
- `split both inputs/mix5_180.txt 1 1 2097152` (1981 bytes) and `inputs/mix4_200.txt` (1601 bytes):
  abort; both parse on that thread.
- `bin/rt2 both inputs/mix6c_112.txt 0 1 2097152` (1123 bytes, my §2.12 parser and the engine on one
  thread): abort after 5 ms, through all three entry points; 111 levels are `Ok`.
- The REPL's own thread (64 MiB, `kernel.rs:45`), debug: `split both inputs/mix5cs_4400.txt 0 1
  67108864` (39.6 KB) parses (`split parse`, 47 ms) and then aborts the process. 4000 levels (36 KB)
  are `Ok`. So one REPL line of 40 KB kills a debug kernel.

**Correctness** where it works: `gpcheck.py inputs/mixcases.txt` with `GPFORM=1` gives maxdiff 0
against gp for `mix5cs`, `mix5`, `mix6c` and `mix6` at 2, 3 and 5 levels (for example `x+-(-(1)^2*3+4)^2*3+4`
at 0 is `1 + x`, gp `x + 1`).

**Over the bar.** Yes, (c): a text of 1.1 to 2 KB, which the parser reads on the thread, overflows a
2 MB debug thread through `taylor_with_budget`, `series_expansion` and the limit path. It is not a
regression: HEAD's `taylor` aborts earlier on every one of these shapes (at 45 to 60 levels). On
8 MB release the parser stops first, by a margin of 1.24 for `mix5cs` with HEAD's parser and of 1.0
for `mix6c` with a §2.12 parser (4438 against 4436).

**Sections.** §2.5 ("Parsed inputs stop in the parser first", "per parenthesis or call it uses 1.6
to 8 times the stack of the engine's recursion", "On the thread that parsed an input
`taylor_with_budget` cannot overflow first", the depth table), §2.12 ("only parentheses and calls
cost the engine stack, and less per level than they cost the parser"), §2.16 fact 3 ("the parser
stops first on the thread that parsed a text") and the rows `from_expr`, `expand`, §6 step 8, §7
item 5. Step 8 cannot catch this: it measures single-construct shapes only. Its remedy also cannot
apply: a parser nesting limit "below the engine's limit on a 2 MB debug thread and not below HEAD's
parser limit on that shape" does not exist for `mix5cs` (engine 132, HEAD's parser 216).

**Fix that closes the class.** The engine's stack per parenthesis must not depend on how many
grammar levels a parenthesis contains. Two ways:

1. (Closes the class.) Make `expand` and `from_expr` iterative over the input: one explicit work
   stack of pending nodes and one stack of partial values, with the chains, towers and argument
   lists as records on the heap. Then the engine has no recursion over the input, fact 3 holds for
   every shape, and the parser is the only recursion over a text. This takes back part of §7 item 5
   for the engine (not for `PartialEq` and `substitute`).
2. (Weaker.) Shrink the recursive path so that the six-level combination costs less than the
   parser's 9.7 KB per parenthesis in debug and 2 KB in release, with a margin of at least 1.5: one
   small dispatcher for `konst_of` and `konst_of_inner` with every arm, the `Unary` arm included, in a
   non-inlined function, and the chain and tower functions split so that the frame that stays on
   the stack holds only the item vector. Frame sizes depend on the compiler, so this needs the
   mixed shapes as tests.

Either way, add the six shapes above to the table of §2.5, to `deep_api_inputs` and to the list of
§6 step 8, on 2 MB debug and 8 MB release and on the REPL's 64 MiB debug thread, and correct the
claims listed under "Sections".

## m1 (below the bar, a Display round trip): a negated fraction under `^` or `/` prints text that §2.12 reads as another value

**What happens.** §2.11 parenthesizes a fraction that is itself an operand of `Pow` or the right
operand of `Div`, and the operand of a negation when it is negative or a negation. It does not cover
a negation whose operand is a positive fraction, `Neg(Rational(p/q))`. That node prints `-p/q`.
Under `^` or as a divisor, §2.12 reads `-p/q` as `(-p)` and then `/q` outside it: `(pi ^ -2/3)`
becomes `pi^(-2)/3`, and `(3 / -1/2)` becomes `3/(-1)/2`. HEAD's parser read `-2/3` as one literal,
so HEAD's identical text read back correctly. Kernels made from input nodes keep the input's
subtree (§2.2), so a result prints such a node whenever the input has one.

**Inputs** (`bin/rt2_rel engine`, release, 8 MB, each about 0.05 ms and 2000 units):

| input, point, order | result text | that text re-parsed (my §2.12 parser, and gp) | correct value |
|---|---|---|---|
| `x*pi^-(2/3)`, 1, 1 | `((pi ^ -2/3) + ((pi ^ -2/3) * (x - 1)))` | `(1/3 / (pi ^ 2)) + ...`, gp `Pi^-2/3` = 0.0338 | `pi^(-2/3)` = 0.4662 |
| `x + atan(3/-(1/2))`, 1, 1 | `((1 + arctan((3 / -1/2))) + (x - 1))` | `arctan(-3/2)`, gp -0.9828 | `atan(-6)` = -1.4056 |
| `x + y^-(1/2)`, 0, 1 | `((y ^ -1/2) + x)` | `(1/2 / y)` | `y^(-1/2)` |
| `x + atan(2^-(1/3))`, 0, 1 | `(arctan((2 ^ -1/3)) + x)` | `arctan(1/6)` | `atan(2^(-1/3))` |

The engine's values are right: `gpcheck.py inputs/disp.txt` gives maxdiff 0 with `GPFORM=1` (an
unambiguous gp form of the result) and 0.42 to 0.51 when gp reads the Display text. The random check
below found this class in 5 of 74 results and no other.

**Over the bar.** No: the computed `Expr` is right and only its text misleads. It is a regression of
the round trip that §2.11 and the round-trip tests of §4.3 promise, on text that HEAD prints the
same way and reads back correctly.

**Fix that closes the class.** `rule_neg` also parenthesizes an operand that is a `Rational` with
denominator above 1, whatever its sign: `-(2/3)`. Then a negated fraction never exposes its `/`, and
`(pi ^ -(2/3))` and `(3 / -(1/2))` read back as printed. By operator and operand, this is the only
pair §2.11 misses: `+`, `-` and `*` give the same value either way, and every binary node is in
parentheses. Add the four inputs above to the round-trip tests of §4.3, and a round trip (Display,
parse, equal value) over random trees in the §2.12 grammar.

## m2 (below the bar, a false claim): results as deep as a left-deep chain inside a call pass `PartialEq`'s limit

§2.15 and §7 item 5 say that `PartialEq` handles the deepest result of any text the parser reads on
the same thread, and they name the tower at 2 (depth 2878 on 2 MB debug) as the deepest. But a
kernel made from an input node takes that node as its expression, and the parser reads a `+ -` chain
by a loop. So `x + sin(1+1+...+1)` with 20 000 ones (40 KB) gives a result of depth 20 003 (`bin/rt2
both inputs/sinsum_20000.txt 0 1 2097152`: `Ok` in 77 ms). `EQ_TOO=1` on the same 2 MB debug thread
aborts; 5000 ones compare in 0.9 ms. The input is as deep as the result, and HEAD's `taylor` aborts
already at 5000 ones, so this is no regression. Fix: state the bound as "the input's depth, or
2 D + 2 for a tower at a point other than 0 and 1, whichever is larger", and drop the claim that
`PartialEq` handles every result of a parsed text.

## m3 (below the bar, an honest error where HEAD answers): an exact monomial to a power above i64

`pow_int` takes the exact-monomial branch before the branch for `n vb >= P`. So `x^N` with
`N vb` above `i64::MAX` is `Unsupported("a series exponent overflows i64")`, although the series is
`O(h^P)` and the next branch returns that. `x^(2^100)+x`, `x^(2^63)+x`, `(2*x)^(2^100)+x` and
`x^(2^100)*sin(x)+x` at 0, order 1, are refused in 0.05 ms. HEAD returns
`((0 ^ N) + (((N * (0 ^ (N - 1))) + 1) * x))`, whose value is x. `x^(2^62)+x` and
`(x+x^2)^(2^100)+x` give x. Fix: in `pow_int`, test `vb > 0 && n vb >= P` before the exact-monomial
branch (the result is `O(h^P)` when `n vb` does not fit), so the overflow error remains only for a
pole of that order. The value of `exp(x)^(2^100)` is right but keeps a kernel `1 ^ N`: `kpow_const`
could return 1 for the base 1 before the "too large" branch.

## Families tried that held

- **Text without parentheses, in the §2.12 structure** (my parser, parse on a 1 GiB thread, release
  8 MB): `x^-x^-...^-x` at 2, order 0, 30 000 carets (90 KB): `Ok` in 0.12 s, 6.1e7 units, result
  depth 90 001 (3 per caret), 57 MB. `x^x^...^x` with 50 000 carets (100 KB) at 2, order 0: `Ok` in
  0.19 s, 9.2e7 units, depth 100 002, 77 MB; at 1, order 3: budget after 0.11 s. `(1+x)^...` with
  15 000 carets at 0: `Ok` in 0.09 s. `sin(x)^...` with 12 000 carets at 1: budget after 0.10 s.
  `x^x^...` with 5000 carets at 1, order 100: budget after 0.98 s (1e8 units per second).
- **Nested kernels**: `x + sqrt(1+sqrt(1+...(1)))` and `(1)/(x + sqrt(1+...))` with 1000 to 10 000
  levels: each level's zero test walks the whole nest, so the work is quadratic, and the budget
  refuses after 0.6 s with at most 31 MB.
- **Many radicals** (release): a product of 9000 square roots of primes (107 KB) `Ok` in 55 ms; the
  inverse of a sum of 8000 square roots 0.13 s; the inverse of a product of 8000, 0.13 s; 5000
  radicals of indices 2 to 64, 51 ms; `sqrt(p)^(2k+1)` for 6000 primes, budget after 0.66 s with
  41 MB; `(sqrt(2)+...+sqrt(p_300))^2` kept as a kernel in 4 ms; `((2)^(1/2))^(1/2)...` 150 levels
  (0.4 ms, one kernel per level).
- **Correctness of chains and towers**: 31 hand cases (`inputs/cases1.txt`: towers with negations
  between levels, constant towers `2^-2^-2^-2`, `* /` chains with constant, series and zero
  divisors, `x*0/x`, `(1)/x/x/x*x^2` at 1, sign runs) and 74 random inputs in the §2.12 grammar
  (`randgen.py 1 600`, `out/rand1c.out`): every coefficient equals gp to 60 digits in the
  unambiguous form, except two values of size 1e120 and 1e214 that agree to 76 relative digits.
- **Engine recursion other than over the input**: by code reading and a scan for self-calls, only
  `kpow_const` (once), `pow_int` (once), `Ball::unary` (once), `balanced_arc` and `bal_depth`
  (log n) recurse. The pre-pass, the ball walks, the key writer and both writers of §2.11 use
  explicit stacks.
- **Caps (code reading)**: the factor cap is only needed in `mono_mul` (`inverse` and `pow_u` keep
  or shorten a monomial, `indet_groups` splits); the term cap is checked before each push in
  `merge2`, `sum_many` and `mul`; the kernel cap before creation; the depth cap before the key. The
  cached depth of a kernel bounds the depth of the expression it builds (the input node's depth, or
  1 + the arguments' rendered depth). The tower and chain loops walk their spine and charge
  `KONST_NODE` per spine node right after it, so on a DAG at most one spine is walked past the
  budget.
- **Huge literals**: exponents `2^62`, `2^100` (non-monomial bases), `L` in `x^(1/L)`-style
  rationals and in kernel keys stay within the budget; the parser fold of `L1/L2` is one gcd.
- **Display of the other operand contexts**: every operator and operand pair other than m1 reads
  back with the same value (each binary node is in parentheses, and sign runs and the literal fold
  give the same value under `+`, `-` and `*`).
- **Not reproducible here**: the `Drop` of §2.14 and the `simplify` of §2.10 are not in the
  prototype (it links HEAD's `Expr` and HEAD's `simplify`). By code reading, `Drop::drop` returns
  before the glue drops the fields, so the design's drop never uses more stack than HEAD's glue;
  and the engine's renderings do not put one child twice in a node, so a kernel chain in a result is
  uniquely owned when its last owner drops it.

## Files

All under `EXP/redteam2-structure/`: `src/rt2.rs` (the probe), `build.sh`, `r.py`, `bs.py`,
`gen.py`, `gpcheck.py`, `randgen.py`; `inputs/mix5cs_133.txt`, `mix5cs_4400.txt`, `mix5_180.txt`,
`mix4_200.txt`, `mix6c_112.txt`, `mixcases.txt`, `disp.txt`, `cases1.txt`, `rand1.txt`,
`sinsum_20000.txt` and the others named above; `out/rand1c.out`. The binaries in `bin/` are rebuilt
by `build.sh` in a few seconds.
