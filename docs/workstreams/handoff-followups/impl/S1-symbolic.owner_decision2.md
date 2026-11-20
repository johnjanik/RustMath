# Owner decision on S1, 2026-11-16 (second decision)

After six review rounds of the test-first implementation, the resources reviewer kept finding
consumers of a result (simplify in rounds 4 and 5; expand, differentiate and the REPL's diff in
round 6) whose cost on a result is about (number of terms) x (their cost on the input's kernel
source). The owner accepted the recommendation (option 1), verbatim from the proposal:

"Narrow it, then finish. The 20 s / 512 MB / stack limit covers S1's own entry points (taylor,
series expansion, limits), plus printing, dropping and re-parsing a result. Reviewers have found
these holding. For simplify, expand, diff and integrate of a result, the docs state the measured
scaling. I'd remove the comments that currently claim those are bounded. Then fix the cheap minor
items, update the trackers, commit and push, and open the S1 PR. A separate unit follows to make
those functions share repeated subexpressions instead of rebuilding them, which is the real fix.
It changes rustmath-symbolic code outside S1, so it deserves its own design."

The three branches (WP-F, Steiner, and this branch with the work-in-progress S1 commit 4763f31)
were merged into main by the owner (main 28c4b69).
