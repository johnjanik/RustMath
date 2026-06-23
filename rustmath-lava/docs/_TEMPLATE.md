# Per-Chapter Documentation — Format Specification

Every chapter document must follow the structure modelled by
`Chapter_38_Galois_Theory_of_Number_Fields.md` (the canonical example — read it first).

## File naming
`Chapter_<NN>_<Title_With_Underscores>.md`  (zero-pad NN to 2 digits, e.g. `Chapter_08_...`).
Use the chapter title from `chapter_map.json`, Title-Cased, spaces → underscores, drop
punctuation that is awkward in filenames (keep it readable).

## Required structure

1. **Title header** — `# Chapter <N> — <Title>` (Title-Cased, human-readable).
2. **Metadata block** — authors (from the handbook part/TOC if visible in the text, else omit),
   handbook part name, handbook printed page range, and PDF page range.
3. **Scope and overview** — 1–4 short paragraphs summarising what the chapter covers and the
   key mathematical objects / algorithmic approach. Capture any prose the chapter gives about
   *methods* and *limitations*.
4. **One section per handbook section** (`## N.k <Section title>`), and `### N.k.m` subsections
   where present. Under each:
   - A short intro paragraph if the chapter provides one (especially any algorithm description).
   - A **Markdown table** of intrinsics with columns: **Intrinsic | Description | Algorithm**.
     - *Intrinsic*: the signature exactly as printed (e.g. `GaloisGroup(f)`). Group overloaded
       signatures that share a description into one row.
     - *Description*: concise but complete — what it returns and the important parameters
       (name + meaning + default when given).
     - *Algorithm*: the method used, **with the bibliography key(s)** in bold (e.g.
       **[Sta73]**) when the text attributes one. If no algorithm is stated, write a short
       characterisation or `—`.
   - Note worked examples in italics: `*Worked examples: Hxx... (brief description).*`
5. **Bibliography section** (`## N.k Bibliography`) — a table `| Key | Reference |` transcribing
   every entry of the chapter's bibliography. This is the "canonical reference".
6. **Algorithm-to-function quick reference** — a closing table mapping each major
   algorithm/theory to the functions that use it.

## Fidelity rules
- Transcribe intrinsic names and signatures **exactly** (preserve capitalisation, arguments).
- Repair PDF artefacts: ligatures (ﬁ→fi, ﬀ→ff, ﬂ→fl), broken hyphenation across lines,
  accented names (Kl¨uners→Klüners, Geißler). Mathematical symbols may be rendered in plain
  text/Unicode (e.g. `Z`, `Q`, `α`, `≤`, `⊆`).
- Do **not** invent intrinsics or references that are not in the extracted text.
- If a chapter is purely introductory (few/no intrinsics), still produce overview +
  bibliography; tables may be small or omitted.
- Keep the algorithm column substantive — the whole point of this project is documenting
  *which algorithm each function uses* and its *canonical reference*.

## Extraction
Run: `python3 docs/extract_chapter.py <N>`  → raw page-delimited text of the chapter.
The first lines of each chapter (the "listing" pages) give the full intrinsic→page index;
use them to ensure no intrinsic is missed.
