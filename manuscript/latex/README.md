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
verified with: 13 pages, no LaTeX errors, only a handful of cosmetic under/overfull-hbox warnings.

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
`figure` float with `\includegraphics`).

It does **not** attempt general-purpose Markdown compatibility (nested lists, blockquotes,
footnotes, HTML, etc.) beyond what `draft.md` actually uses. If you add a new markdown construct
to `draft.md`, either avoid it or extend the converter to handle it before regenerating.
