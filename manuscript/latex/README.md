# LaTeX submission package

`main.tex` is generated from `manuscript/draft.md` by `manuscript/convert_to_latex.py`. Do not
hand-edit `main.tex` -- edit `draft.md` and rerun the converter, or your changes will be
overwritten on the next regeneration.

## Regenerate and compile

```bash
uv run python manuscript/convert_to_latex.py
cd manuscript/latex
tectonic main.tex   # or: latexmk -pdf main.tex / pdflatex main.tex (twice)
```

`tectonic` (self-contained, fetches packages on first use) is what this package was built and
verified with. Since the 2026-11-10 revision this plain build is the public (arXiv) version: it
carries the real author block (`AUTHOR` in the converter) and links the public repository. 16 pages:
main text ends on page 10, references on page 11, appendices A-G after them.

arXiv source package (`dist/` is gitignored):

```bash
tar -czf dist/arxiv_source.tar.gz -C manuscript/latex main.tex \
  figures/elasticmla_main_results.pdf figures/elasticmla_risk_spectrum.pdf
```

## TMLR submission version

The venue is TMLR (decided 2026-10-27, see `notes/submission_readiness_roadmap.md`).

```bash
uv run python manuscript/convert_to_latex.py --style tmlr
cd manuscript/latex_tmlr && tectonic main.tex
```

(TMLR desk-rejected #13801 on 2026-11-08; `latex_tmlr/` is kept as the submitted snapshot and was
not regenerated from the revised draft, whose appendix links the non-anonymous repository -- remove
that link before reusing this style for any double-blind venue) writes the anonymized,
TMLR-formatted `manuscript/latex_tmlr/main.tex` from the same
`draft.md`. The official style files (`tmlr.sty`, `fancyhdr.sty`, `tmlr.bst`; Apache-2.0,
`LICENSE.tmlr-style`) are vendored there. Authors are hidden by the style until `[accepted]`.

Supplementary archive (anonymized code, tests, results, notes; deterministic, from git HEAD):

```bash
uv run python manuscript/build_supplement.py   # -> dist/elasticmla_supplement.zip (gitignored)
```

## Contents

- `main.tex` -- generated LaTeX source (title, abstract, all sections, tables, figures,
  bibliography).
- `figures/` -- the two figures actually cited in the paper (`elasticmla_main_results.pdf`,
  `elasticmla_risk_spectrum.pdf`), copied here by the converter so this directory is
  self-contained and does not depend on the rest of the repository to compile.

## Converter scope and known limitations

`manuscript/convert_to_latex.py` is a purpose-built converter for this document's specific
markdown conventions, not a general markdown-to-LaTeX tool. It handles: `#`/`##`/`###` headers,
GFM pipe tables (auto-wrapped in `\resizebox` when they have 7+ columns), `$...$`/`$$...$$` math
(passed through verbatim, since the source markdown was already written in LaTeX math syntax),
`` `code` `` spans (rendered as escaped `\texttt{}` with breakable slashes), `**bold**`/`*italic*`,
bracketed numeric citations (rendered as plain text, not `\cite`), and the paper's
`**Figure N.** ... Source: \`path\`.` caption-paragraph convention (rendered as a real
`figure` float with `\includegraphics`), a `**Table N.** ...` paragraph directly before a table
(becomes that table's `\caption`), and a `## Appendix` line after the references: every section
after it is typeset after the bibliography under `\appendix` (lettered A, B, ...). The title is read
from the draft's first `#` line.

It does **not** attempt general-purpose Markdown compatibility (nested lists, blockquotes,
footnotes, HTML, etc.) beyond what `draft.md` actually uses. If you add a new markdown construct
to `draft.md`, either avoid it or extend the converter to handle it before regenerating.
