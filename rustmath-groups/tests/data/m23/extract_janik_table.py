"""Write janik_table.json from the owner's m23golay/nielsen.py (JANIK_TABLE only).

JANIK_TABLE is the owner's transcription of Table 1 of his own manuscript [Janik26]
(J. A. Janik, "Real Structure and the Galois Fixed Point in the Nielsen Class (2,23A,23B)
of M23", 2026). Nothing else is taken: LELAY_X (Le Lay's representatives, CC BY-NC-SA) and
HJLPPZ_FILE (the HJLPPZ triples, no licence) are not read.

Run:  python3 -I extract_janik_table.py <checkout of johnjanik/M23 at 2d11038> <out.json>
The module source is parsed with ast; the owner's code is never imported or run.
Cycles stay 1-based, as printed in the source.
"""
import ast, hashlib, json, sys

src_path = sys.argv[1] + "/m23golay/nielsen.py"
src = open(src_path, "rb").read()
tree = ast.parse(src)


def cycles_of_call(node):
    # _c(1, 13, ...) -> [[1, 13, ...]];  from_cycles([(a, b), ...]) -> [[a, b], ...]
    if node.func.id == "_c":
        return [[ast.literal_eval(a) for a in node.args]]
    assert node.func.id == "from_cycles"
    return [list(t) for t in ast.literal_eval(node.args[0])]


out = {
    "source": "johnjanik/M23 m23golay/nielsen.py, JANIK_TABLE ([Janik26, Table 1])",
    "source_commit": "2d11038",
    "source_sha256": hashlib.sha256(src).hexdigest(),
    "reading": "rows (g2, g3) with first entry the published g1; g1 g2 g3 = 1",
}
for stmt in tree.body:
    if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
        if stmt.targets[0].id == "JANIK_TABLE":
            out["janik_table_g2_g3"] = [[cycles_of_call(e) for e in row.elts] for row in stmt.value.elts]
assert len(out["janik_table_g2_g3"]) == 7
json.dump(out, open(sys.argv[2], "w"), indent=0)
