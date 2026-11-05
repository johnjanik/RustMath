#!/usr/bin/env python3
"""Build the reading list (papers/books to collect locally) for a chapter scope.

Reads docs/magma/bibliography.csv and docs/magma/intrinsics.csv (run build.py first) and writes
  docs/magma/reading_list_<name>.csv   machine-readable, one row per reference
  docs/magma/reading_list_<name>.md    grouped for humans (by priority, then chapter)

With --doi, each reference is looked up on Crossref (title match, cached in
docs/magma/doi_cache.json); a DOI is recorded only when the returned title matches closely.

Usage (repo root):
  python3 tools/magma_tracker/reading_list.py --name core --chapters 17-52,57-68 [--doi]
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAG = ROOT / "docs" / "magma"
CACHE = MAG / "doi_cache.json"
UA = "rustmath-reading-list/0.1 (https://github.com/johnjanik/RustMath)"


def parse_ranges(spec: str) -> set[int]:
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def split_authors(text: str) -> tuple[str, str]:
    """Split 'A. B. Name and C. Other. Title words. More' after the author list.

    The author list ends at the first '. ' whose preceding word is not an initial
    (a single letter, or a hyphenated pair like 'J.-F.')."""
    for m in re.finditer(r"\.\s+(?=[A-Z0-9\"'“(])", text):
        before = text[: m.start()]
        last = re.split(r"[\s]+", before.strip())[-1] if before.strip() else ""
        # initials: "J", "Th", "N.J.A", "J.-F" (letters left after dropping dots/hyphens)
        core = last.replace(".", "").replace("-", "")
        if re.fullmatch(r"[A-Z][a-z]?|[A-Z]{1,4}", core):
            continue
        return before.strip(), text[m.end():].strip()
    return "", text.strip()


VENUE_AFTER = re.compile(r"^\s*[,.]?\s*(\d|vol|volume|pages|pp|no\.|\(|[A-Z][a-z]+\s+\d)", re.I)


def split_reference(ref: str) -> dict:
    m = re.search(r"\*([^*]+)\*", ref)
    style_a = False
    if m:
        # sentinel "X" gives the author splitter a capitalized word after the final ". "
        _, between = split_authors(ref[: m.start()].rstrip() + " X")
        between = between[:-1].strip()
        # style A when nothing sentence-like sits between the author list and the italics
        style_a = len(between.split()) < 2
    if style_a:
        # style A: "Authors. *Title*. Venue..."
        title = m.group(1).strip().rstrip(".")
        authors = ref[: m.start()].strip().rstrip(".")
        rest = ref[m.end():].strip(" .,")
    else:
        # style B: "Authors. Title. *Journal*, vol:pages, year."  or no italics at all
        head = ref[: m.start()] if m else ref
        authors, tail = split_authors(head)
        title = tail.strip().rstrip(".")
        if not m:
            # no italics: title is the first sentence after the authors
            title = re.split(r"\.\s+(?=[A-Z])", tail, maxsplit=1)[0].strip().rstrip(".")
        rest = ref[len(authors):].replace(title, "", 1).strip(" .,") if title else ref
    years = re.findall(r"\b(1[89]\d\d|20\d\d)\b", ref)
    arxiv = re.search(r"arXiv[: ]\s*([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})|arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})",
                      ref, re.I)
    low = ref.lower()
    if re.search(r"ph\.?\s*d|thesis|dissertation|diplomarbeit|habilitation|master", low):
        kind = "thesis"
    elif arxiv or "preprint" in low or "submitted" in low or "in preparation" in low:
        kind = "preprint"
    elif re.search(r"magma|software|package|manual|handbook of functions", low) and "journal" not in low:
        kind = "software/manual"
    elif re.search(r"\b(press|springer|graduate texts|lecture notes|volume \d+ of|vol\. \d+ of|"
                   r"cambridge university|birkh|wiley|ams|crc|chapman|de gruyter|series)\b", low) \
            and not re.search(r"\*\*\d+\*\*|\bpp?\.|\d+\s*[–-]\s*\d+", rest):
        kind = "book"
    elif re.search(r"proceedings|proc\.|symposium|conference|issac|ants|lncs|lecture notes in comput",
                   low):
        kind = "proceedings"
    else:
        kind = "article"
    return {"authors": authors, "title": title, "venue": rest, "year": years[-1] if years else "",
            "arxiv": (arxiv.group(1) or arxiv.group(2)) if arxiv else "", "type": kind}


def crossref(ref: dict, cache: dict) -> tuple[str, float, str]:
    if not ref["title"]:
        return "", 0.0, ""
    key = norm(ref["title"]) + "|" + ref["year"]
    if key in cache:
        c = cache[key]
        return c["doi"], c["score"], c["title"]
    q = {"query.bibliographic": f"{ref['title']} {ref['authors']} {ref['year']}", "rows": "3",
         "select": "DOI,title,issued"}
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(q)
    best = ("", 0.0, "")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            items = json.load(r)["message"]["items"]
        for it in items:
            t = (it.get("title") or [""])[0]
            score = difflib.SequenceMatcher(None, norm(t), norm(ref["title"])).ratio()
            yr = str((it.get("issued", {}).get("date-parts") or [[None]])[0][0] or "")
            if ref["year"] and yr and abs(int(yr) - int(ref["year"])) > 2:
                score -= 0.15
            if score > best[1]:
                best = (it.get("DOI", ""), round(score, 3), t)
    except Exception as e:  # network or parse failure: leave blank, do not cache
        print(f"crossref failed for {ref['title'][:50]}: {e}", file=sys.stderr)
        if "429" in str(e):
            time.sleep(10)
        return "", 0.0, ""
    cache[key] = {"doi": best[0], "score": best[1], "title": best[2]}
    time.sleep(1.0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--chapters", required=True)
    ap.add_argument("--doi", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.90)
    a = ap.parse_args()
    scope = parse_ranges(a.chapters)

    titles = {}
    cites = defaultdict(lambda: defaultdict(int))   # key -> chapter -> n intrinsics citing
    with (MAG / "intrinsics.csv").open() as f:
        for r in csv.DictReader(f):
            ch = int(r["chapter"])
            titles[ch] = r["chapter_title"]
            if ch in scope:
                for k in r["refs"].split():
                    cites[k][ch] += 1

    rows = []
    with (MAG / "bibliography.csv").open() as f:
        for b in csv.DictReader(f):
            chs = {int(c) for c in b["chapters"].split()} & scope
            if not chs and b["key"] not in cites:
                continue
            chs |= set(cites[b["key"]])
            p = split_reference(b["reference"])
            n = sum(cites[b["key"]].values())
            rows.append({"key": b["key"], **p, "chapters": sorted(chs), "n_intrinsics": n,
                         "priority": "A" if n else "B", "reference": b["reference"]})

    # merge exact duplicates (same normalized title) under different keys
    merged: dict[str, dict] = {}
    for r in rows:
        t = norm(r["title"]) or r["key"]
        if t in merged:
            m = merged[t]
            m["key"] += " / " + r["key"]
            m["chapters"] = sorted(set(m["chapters"]) | set(r["chapters"]))
            m["n_intrinsics"] += r["n_intrinsics"]
            m["priority"] = "A" if m["n_intrinsics"] else "B"
        else:
            merged[t] = dict(r)
    rows = sorted(merged.values(), key=lambda r: (r["priority"], -r["n_intrinsics"], r["key"].lower()))

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for i, r in enumerate(rows):
        doi = score = ctitle = ""
        if a.doi and r["type"] in ("article", "proceedings", "book", "preprint"):
            doi, score, ctitle = crossref(r, cache)
            if i % 25 == 0:
                CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True))
        r["doi"] = doi if score and score >= a.min_score else ""
        r["doi_candidate"] = doi if doi and not r["doi"] else ""
        r["doi_score"] = score
    if a.doi:
        CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True))

    fields = ["priority", "key", "type", "authors", "title", "venue", "year", "chapters", "n_intrinsics",
              "doi", "doi_candidate", "doi_score", "arxiv", "reference"]
    with (MAG / f"reading_list_{a.name}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "chapters": " ".join(map(str, r["chapters"]))})

    write_md(a, rows, titles)
    n_doi = sum(1 for r in rows if r["doi"])
    print(f"{len(rows)} references ({sum(r['priority'] == 'A' for r in rows)} priority A), "
          f"{n_doi} with confident DOI", file=sys.stderr)
    return 0


def link(r):
    if r["doi"]:
        return f"[doi:{r['doi']}](https://doi.org/{r['doi']})"
    if r["arxiv"]:
        return f"[arXiv:{r['arxiv']}](https://arxiv.org/abs/{r['arxiv']})"
    if r["doi_candidate"]:
        return f"unconfirmed: [doi:{r['doi_candidate']}](https://doi.org/{r['doi_candidate']})"
    return ""


def write_md(a, rows, titles):
    by_type = defaultdict(int)
    for r in rows:
        by_type[r["type"]] += 1
    L = [f"# Reading list: `{a.name}` (chapters {a.chapters})", "",
         "_Generated by `tools/magma_tracker/reading_list.py` from the handbook chapter bibliographies._", "",
         f"{len(rows)} distinct references. Types: "
         + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])) + ".", "",
         "- **Priority A**: cited in the algorithm/description cell of at least one intrinsic in scope "
         "(these name the algorithm we must implement). Sorted by number of citing intrinsics.",
         "- **Priority B**: listed in a chapter bibliography in scope but not attached to a specific intrinsic.",
         "- DOIs come from Crossref title matching; `unconfirmed` ones scored below the threshold and need a "
         "human check. Books and theses mostly have no DOI.", "",
         "Suggested local layout: `<key>.pdf` per entry (first key if several), matching the `key` column of "
         "`reading_list_" + a.name + ".csv`.", ""]
    for pri, head in (("A", "Priority A: cited by specific intrinsics"), ("B", "Priority B: chapter bibliographies")):
        L += [f"## {head}", "", "| Key | Type | Reference | Chapters | Intrinsics | Link |",
              "|---|---|---|---|--:|---|"]
        for r in rows:
            if r["priority"] != pri:
                continue
            ref = r["reference"].replace("|", "/")
            L.append(f"| {r['key']} | {r['type']} | {ref} | {' '.join(map(str, r['chapters']))} | "
                     f"{r['n_intrinsics']} | {link(r)} |")
        L.append("")
    L += ["## Chapters in scope", "", "| Ch | Title |", "|--:|---|"]
    for ch in sorted(parse_ranges(a.chapters)):
        if ch in titles:
            L.append(f"| {ch} | {titles[ch]} |")
    (MAG / f"reading_list_{a.name}.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
