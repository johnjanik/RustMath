# Build transitive_24.jsonl from GAP's TransGrp library (no LMFDB access needed).
#
# Usage (from this directory; needs GAP with the TransGrp package, e.g.
#   Ubuntu/Debian: apt install gap-core gap-libs gap-transgrp
#   FreeBSD:       pkg install gap):
#     gap -q build_transitive_24.g
# Writes transitive_24.jsonl, one JSON object per group 24Tt (t = 1..25000):
#   {"label":"24Tt","t":t,"order":"<n>","parity":1|-1,"prim":0|1,"solv":0|1,
#    "gens":[[[a,b,...],[c,d,...]], ...]}
# gens are generators in 1-indexed disjoint-cycle notation (non-trivial cycles only).
# GAP and LMFDB share the Hulpke numbering for degree 24, so t matches LMFDB's 24Tt;
# GAP's generators may differ from LMFDB's but generate the same permutation group.
# parity follows LMFDB: 1 if G <= A_24, -1 otherwise.

OUT := "transitive_24.jsonl";
out := OutputTextFile(OUT, false);
SetPrintFormattingStatus(out, false);  # no GAP line wrapping: one JSON object per line

JsonList := function(l)
    return Concatenation("[", JoinStringsWithSeparator(List(l, String), ","), "]");
end;

Bit := function(b)
    if b then return 1; else return 0; fi;
end;

for t in [1..NrTransitiveGroups(24)] do
    G := TransitiveGroup(24, t);
    gens := Filtered(GeneratorsOfGroup(G), g -> g <> ());
    gjson := List(gens, g -> JsonList(List(Filtered(Cycles(g, [1..24]), c -> Length(c) > 1),
                                           JsonList)));
    if ForAll(gens, g -> SignPerm(g) = 1) then parity := 1; else parity := -1; fi;
    PrintTo(out, "{\"label\":\"24T", String(t), "\",\"t\":", String(t),
             ",\"order\":\"", String(Size(G)), "\",\"parity\":", String(parity),
             ",\"prim\":", String(Bit(IsPrimitive(G, [1..24]))),
             ",\"solv\":", String(Bit(IsSolvableGroup(G))),
             ",\"gens\":", JsonList(gjson), "}\n");
    if t mod 2500 = 0 then Print(t, "\n"); fi;
od;
CloseStream(out);
QUIT;
