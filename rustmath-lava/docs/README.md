# MAGMA Handbook of Functions — Per-Chapter Documentation

This directory holds one Markdown document per chapter of the *MAGMA Handbook of Functions*
(5488-page PDF). Each document records, for its chapter:

- **All modules / functions (intrinsics)** with their signatures and a concise description.
- **The algorithm(s)** used by each function, with the bibliography key(s) that document them.
- **A canonical reference list** (the chapter's bibliography), plus an algorithm-to-function
  quick-reference table.

## Naming convention

`Chapter_<NN>_<Title_With_Underscores>.md`

## Source-PDF page mapping

Handbook (printed) page numbers map to 0-based PDF page indices by a fixed offset in this
front-matter region:

> **PDF page index = handbook page number + 133**

(Verified: handbook p. 961 — start of Chapter 38 — is PDF page 1094.) The offset may shift
across the book if the front matter or part dividers change pagination; re-verify per part by
locating a known chapter heading. Chapter page ranges come from `../TOC.pdf` (the table of
contents with page numbers).

Extraction recipe (PyMuPDF / `fitz` is installed):

```python
import fitz
doc = fitz.open('MAGMA_handbook_of_functions.pdf')
text = "".join(doc[i].get_text() for i in range(start_pdf_page, end_pdf_page + 1))
```

## Progress

| Chapter | Title | Handbook pp. | PDF pp. (0-based) | Status |
|---------|-------|--------------|-------------------|--------|
| 38 | Galois Theory of Number Fields | 961–995 | 1094–1128 | ✅ Done |

(Chapters are being documented on request, starting with Chapter 38.)
