#!/usr/bin/env python3
"""Build the MAGMA intrinsic tracker from the handbook chapter digests.

Inputs
  rustmath-lava/docs/Chapter_NN_*.md      per-chapter digests (the spec)
  docs/reference/magma_coverage.md         chapter -> home crate(s)
  PROJECT_TRACKER.md                        algorithm inventory (file:line -> status emoji)
  docs/magma/status_overrides.csv           hand-verified statuses (preserved across runs)
  all *.rs under rustmath-*/                fn definitions for name matching

Outputs (docs/magma/)
  intrinsics.csv     one row per intrinsic table row in the digests
  bibliography.csv   one row per bibliography key, with the chapters citing it
  STATUS.md          per-part / per-chapter roll-up

Status lifecycle (column `status`):
  auto:   missing    no Rust fn with the snake_case name anywhere
          candidate  a same-named fn exists (home crate preferred); NOT verified
  manual: stub | partial | done | verified | facade | broken | n/a
          set in status_overrides.csv (id,status,rust_path,note); overrides always win.

Run from the repo root:  python3 tools/magma_tracker/build.py
Standard library only.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIGESTS = ROOT / "rustmath-lava" / "docs"
OUT = ROOT / "docs" / "magma"
COVERAGE = ROOT / "docs" / "reference" / "magma_coverage.md"
INVENTORY = ROOT / "PROJECT_TRACKER.md"
OVERRIDES = OUT / "status_overrides.csv"

# Parts I-II (ch 1-16) document the Magma language/runtime (REPL, aggregates, records, I/O).
# They are tracked as scope "language": the front-end (rustmath-lava) concern, not the math port.
LANGUAGE_MAX_CHAPTER = 16

MANUAL_STATUSES = {"stub", "partial", "done", "verified", "facade", "broken", "n/a", "missing", "candidate"}

# --------------------------------------------------------------------------- helpers


def split_row(line: str) -> list[str]:
    """Split a markdown table row on '|' that are neither escaped nor inside backticks."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, cur, in_code, i = [], [], False, 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur.append("|")
            i += 2
            continue
        if c == "`":
            in_code = not in_code
        if c == "|" and not in_code:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    cells.append("".join(cur).strip())
    return cells


NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
KEY_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9+&'\-]*\d*[a-z]?(?:\+)?)\]")
SKIP_NAMES = {"ext", "sub", "quo", "rec", "hom", "map", "iso", "case", "for", "if", "while",
              "func", "proc", "intrinsic", "not", "and", "or", "in", "eq", "ne", "lt", "gt"}


def intrinsic_names(cell: str) -> list[str]:
    """MAGMA intrinsic names in an Intrinsic cell (first identifier of each backtick span)."""
    names = []
    for span in re.findall(r"`([^`]*)`", cell) or [cell]:
        span = span.strip()
        m = NAME_RE.match(span)
        if not m:
            continue
        name = m.group(0)
        # `R . i` style or constructor syntax `ext< ... >`: keep the operator word too
        if name.lower() in SKIP_NAMES:
            names.append(name)
            continue
        names.append(name)
    out = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def snake(name: str) -> str:
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def slug(text: str) -> str:
    s = re.sub(r"`", "", text)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    return s[:60] or "row"


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- inputs


def load_coverage() -> dict[int, dict]:
    cov = {}
    if not COVERAGE.exists():
        return cov
    for line in COVERAGE.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = split_row(line)
        if len(cells) < 5 or not cells[0].strip().isdigit():
            continue
        ch = int(cells[0])
        crates = re.findall(r"rustmath-[a-z0-9\-]+", cells[3])
        cov[ch] = {"coverage": cells[1], "effort": cells[2], "home": cells[3], "crates": crates}
    return cov


def load_inventory() -> dict[str, str]:
    """file:line -> status emoji from PROJECT_TRACKER's Algorithm Inventory tables."""
    inv = {}
    if not INVENTORY.exists():
        return inv
    for line in INVENTORY.read_text().splitlines():
        if not line.startswith("| "):
            continue
        cells = split_row(line)
        if len(cells) < 3:
            continue
        for loc in re.findall(r"`(rustmath-[^`:]+\.rs):(\d+)`", cells[2]):
            inv[f"{loc[0]}:{loc[1]}"] = cells[0].strip()
    return inv


INV_EMOJI = {"🟢": "done", "⏳": "partial", "🛑": "facade", "🔴": "broken", "⬜": "missing"}


def index_rust_fns() -> dict[str, list[tuple[str, str, int]]]:
    """snake name -> [(crate, path, line)] for every `fn name` in rustmath-* sources."""
    idx: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    fn_re = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([a-z_][a-z0-9_]*)")
    for crate_dir in sorted(ROOT.glob("rustmath-*")):
        if not crate_dir.is_dir():
            continue
        for path in crate_dir.rglob("*.rs"):
            if "target" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            # crate name: rustmath-lava/crates/lava-core -> lava-core
            parts = rel.split("/")
            crate = parts[2] if parts[0] == "rustmath-lava" and len(parts) > 2 and parts[1] == "crates" else parts[0]
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for n, line in enumerate(lines, 1):
                m = fn_re.match(line)
                if m:
                    idx[m.group(1)].append((crate, rel, n))
    return idx


def load_overrides() -> dict[str, dict]:
    ov = {}
    if OVERRIDES.exists():
        with OVERRIDES.open() as f:
            for row in csv.DictReader(f):
                if row.get("id"):
                    ov[row["id"]] = row
    return ov


# --------------------------------------------------------------------------- digest parsing


def parse_digest(path: Path):
    text = path.read_text()
    m = re.match(r"Chapter_(\d+)_", path.name)
    ch = int(m.group(1))
    title_m = re.search(r"^# Chapter \d+\s*[—-]\s*(.+)$", text, re.M)
    title = clean(title_m.group(1)) if title_m else path.stem
    part_m = re.search(r"^\*\*Handbook part:\*\*\s*(.+)$", text, re.M)
    part = clean(part_m.group(1)) if part_m else ""
    pages_m = re.search(r"^\*\*Handbook pages:\*\*\s*(.+)$", text, re.M)
    pages = clean(pages_m.group(1)) if pages_m else ""

    rows, bib = [], []
    section = subsection = ""
    header: list[str] | None = None
    in_bib = False
    for line in text.splitlines():
        h = re.match(r"^(#{2,4})\s+(.*)$", line)
        if h:
            level, head = len(h.group(1)), clean(h.group(2))
            if level == 2:
                section, subsection = head, ""
                in_bib = bool(re.search(r"bibliograph|^references?$", head, re.I))
            else:
                subsection = head
            header = None
            continue
        if not line.startswith("|"):
            header = None
            continue
        cells = split_row(line)
        if all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c):
            continue  # separator row
        if header is None:
            header = [c.lower() for c in cells]
            continue
        if header and header[0] == "key" and "reference" in header:
            key = re.sub(r"[*\[\]]", "", cells[0]).strip()
            ref = clean(cells[1]) if len(cells) > 1 else ""
            if key:
                bib.append({"key": key, "reference": ref, "chapter": ch, "in_bib_section": in_bib})
            continue
        if header and header[0] in ("intrinsic", "intrinsics", "function", "signature"):
            sig = clean(cells[0])
            if not sig or "algorithm-to-function" in section.lower() or "quick reference" in section.lower():
                continue
            desc = clean(cells[1]) if len(cells) > 1 else ""
            algo = clean(cells[2]) if len(cells) > 2 and "algorithm" in header[2:3] else ""
            keys = sorted(set(KEY_RE.findall(desc + " " + algo)))
            rows.append({
                "chapter": ch, "part": part, "section": section, "subsection": subsection,
                "signature": sig, "names": intrinsic_names(sig), "description": desc,
                "algorithm": algo, "refs": keys,
            })
    return {"chapter": ch, "title": title, "part": part, "pages": pages, "rows": rows, "bib": bib}


# --------------------------------------------------------------------------- main


def main() -> int:
    coverage = load_coverage()
    inventory = load_inventory()
    overrides = load_overrides()
    fns = index_rust_fns()

    chapters = [parse_digest(p) for p in sorted(DIGESTS.glob("Chapter_*.md"),
                                                key=lambda p: int(re.match(r"Chapter_(\d+)", p.name).group(1)))]
    OUT.mkdir(parents=True, exist_ok=True)

    seen_ids: Counter = Counter()
    records = []
    for chap in chapters:
        ch = chap["chapter"]
        home = coverage.get(ch, {}).get("crates", [])
        for r in chap["rows"]:
            base = f"ch{ch:03d}.{slug(r['names'][0] if r['names'] else r['signature'])}"
            seen_ids[base] += 1
            rid = base if seen_ids[base] == 1 else f"{base}.{seen_ids[base]}"
            # candidate code match
            match_path, match_scope, inv_status = "", "", ""
            for name in r["names"]:
                if name.lower() in SKIP_NAMES:
                    continue
                hits = fns.get(snake(name), [])
                if not hits:
                    continue
                home_hits = [h for h in hits if h[0] in home]
                best = (home_hits or hits)[0]
                match_scope = "home" if home_hits else "elsewhere"
                match_path = f"{best[1]}:{best[2]}"
                for h in (home_hits or hits):
                    e = inventory.get(f"{h[1]}:{h[2]}")
                    if e:
                        inv_status = INV_EMOJI.get(e, e)
                        match_path = f"{h[1]}:{h[2]}"
                        break
                break
            status = "candidate" if match_path else "missing"
            note = ""
            ov = overrides.get(rid)
            if ov:
                if ov.get("status") in MANUAL_STATUSES:
                    status = ov["status"]
                if ov.get("rust_path"):
                    match_path = ov["rust_path"]
                note = ov.get("note", "")
            records.append({
                "id": rid, "scope": "language" if ch <= LANGUAGE_MAX_CHAPTER else "math", "chapter": ch, "chapter_title": chap["title"], "part": chap["part"],
                "section": r["section"], "subsection": r["subsection"],
                "intrinsic": ", ".join(r["names"]), "signature": r["signature"],
                "description": r["description"], "algorithm": r["algorithm"],
                "refs": " ".join(r["refs"]), "home_crates": " ".join(home),
                "status": status, "match_scope": match_scope, "rust_path": match_path,
                "inventory_status": inv_status, "verified_note": note,
            })

    fields = ["id", "scope", "chapter", "chapter_title", "part", "section", "subsection", "intrinsic",
              "signature", "description", "algorithm", "refs", "home_crates", "status",
              "match_scope", "rust_path", "inventory_status", "verified_note"]
    with (OUT / "intrinsics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    # bibliography: merge keys across chapters
    bib: dict[str, dict] = {}
    cited_by: dict[str, set] = defaultdict(set)
    for chap in chapters:
        for r in chap["rows"]:
            for k in r["refs"]:
                cited_by[k].add(chap["chapter"])
        for b in chap["bib"]:
            e = bib.setdefault(b["key"], {"key": b["key"], "reference": b["reference"], "chapters": set()})
            e["chapters"].add(b["chapter"])
            if len(b["reference"]) > len(e["reference"]):
                e["reference"] = b["reference"]
    with (OUT / "bibliography.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "chapters", "n_intrinsics_citing", "reference"])
        n_cite = Counter(k for rec in records for k in rec["refs"].split())
        for k in sorted(bib, key=str.lower):
            w.writerow([k, " ".join(str(c) for c in sorted(bib[k]["chapters"])), n_cite.get(k, 0),
                        bib[k]["reference"]])

    write_status(chapters, records, coverage)
    print(f"{len(records)} intrinsic rows, {len(bib)} bibliography keys -> {OUT.relative_to(ROOT)}",
          file=sys.stderr)
    return 0


def write_status(chapters, records, coverage):
    by_ch = defaultdict(Counter)
    for r in records:
        by_ch[r["chapter"]][r["status"]] += 1
        by_ch[r["chapter"]]["_total"] += 1
        if r["match_scope"] == "home":
            by_ch[r["chapter"]]["_home"] += 1
        if r["inventory_status"] in ("facade", "broken"):
            by_ch[r["chapter"]]["_bad"] += 1
    tot = Counter()
    for ch, c in by_ch.items():
        if ch > LANGUAGE_MAX_CHAPTER:
            tot.update(c)

    def pct(a, b):
        return f"{100 * a / b:.0f}%" if b else "-"

    lines = [
        "# MAGMA port status (generated)",
        "",
        "_Generated by `tools/magma_tracker/build.py` from the handbook digests in `rustmath-lava/docs/`._",
        "_Do not edit by hand; record verified statuses in `status_overrides.csv` and regenerate._",
        "",
        "`candidate` means a Rust function with the snake_case name exists. It is a lead for review, **not**",
        "evidence the intrinsic is implemented: many names (`Degree`, `Order`) match unrelated functions.",
        "Only `verified` (tested against reference values) counts as ported.",
        "",
        "## Totals (mathematics scope: chapters 17-159)",
        "",
        "Chapters 1-16 (the Magma language and runtime) are listed below but excluded from the totals.",
        "",
        "| Rows | verified | done | partial | stub | facade/broken | candidate | missing |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|",
        f"| {tot['_total']} | {tot['verified']} | {tot['done']} | {tot['partial']} | {tot['stub']} | "
        f"{tot['facade'] + tot['broken']} | {tot['candidate']} ({pct(tot['candidate'], tot['_total'])}) | "
        f"{tot['missing']} ({pct(tot['missing'], tot['_total'])}) |",
        "",
        "## By chapter",
        "",
        "| Ch | Title | Part | Rows | verified | candidate (home crate) | missing | Survey coverage | Home crate |",
        "|--:|---|---|--:|--:|--:|--:|---|---|",
    ]
    for chap in chapters:
        ch = chap["chapter"]
        c = by_ch.get(ch, Counter())
        part = chap["part"].split("—")[0].strip()
        cov = coverage.get(ch, {})
        lines.append(
            f"| {ch} | {chap['title']} | {part} | {c['_total']} | {c['verified']} | "
            f"{c['candidate']} ({c['_home']}) | {c['missing']} | {cov.get('coverage', '')} | "
            f"{cov.get('home', '').replace('|', '/')} |")
    (OUT / "STATUS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
