# Owner decision on S1 (2026-11-15), forwarded from the owner's other session (verbatim)

S1: fix and implement, but change the acceptance test
This is based on the park record on `claude/gracious-gates-e6b6l7`, not on the 5,259-line design itself.
Paper review has stopped converging, so I agree with moving to implementation. The catch is the proposed acceptance test: the 2,000 random inputs against PARI/GP passed while both blockers were present. Review 9 found B1 and B2 by reasoning, not through that harness. Used alone, it would accept the code with the blockers still in it.
What I would ask for:

* A singular-point matrix alongside the random harness. For each inverse function and each of its branch points, test two inputs: an argument that simplifies to that constant, and a non-constant argument with that constant term. Each has one expected outcome: a value for the first, a typed refusal for the second.
* Refusal instead of a scope cut. Both blockers come from expanding at a branch point. An exact-constant argument should just be evaluated. A non-constant argument at a branch point should return "not a Laurent series", never "division by zero". That keeps the inverse functions everywhere they are regular, so option 2's wholesale removal is not needed.
* Freeze the design. No revision 9. Turn B1, B2 and the red team's three remaining holes into regression tests before any fix is written.

One thing to weigh: S1 is in `rustmath-symbolic`, and nothing in the M23 work packages depends on it. If the M23 results are the priority, leaving it parked costs them nothing.
