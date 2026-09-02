#!/usr/bin/env python3
"""Convert manuscript/draft.md into a compilable LaTeX submission package.

Regenerates manuscript/latex/main.tex (and copies the two cited figures) from the
current manuscript/draft.md. Run, then compile with:

    tectonic manuscript/latex/main.tex

or any standard LaTeX toolchain (pdflatex/xelatex). This is a purpose-built
converter for this document's specific markdown conventions (headers, GFM
tables, $...$/$$...$$ math, `code` spans, **bold**/*italic*, bracketed
numeric citations, and "**Figure N.** ... Source: `path`." caption
paragraphs) rather than a general markdown-to-LaTeX tool.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "manuscript" / "draft.md"
LATEX_DIR = ROOT / "manuscript" / "latex"


def strip_number_prefix(s):
    return re.sub(r"^\d+(\.\d+)*\.?\s+", "", s).strip()


def code_repl(m):
    code = m.group(1)
    code = (
        code.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )
    code = code.replace("/", r"/\allowbreak ")
    return r"\texttt{" + code + "}"


def protect_and_convert(text):
    placeholders = []

    def stash(s):
        placeholders.append(s)
        return f"@@PH{len(placeholders) - 1}@@"

    text = re.sub(r"\$\$(.+?)\$\$", lambda m: stash("$$" + m.group(1) + "$$"), text, flags=re.S)
    text = re.sub(r"\$(.+?)\$", lambda m: stash("$" + m.group(1) + "$"), text, flags=re.S)
    text = re.sub(r"`([^`]+)`", lambda m: stash(code_repl(m)), text)

    text = text.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\textit{\1}", text)
    text = re.sub(r"(?<=\s)--(?=\s)", "---", text)

    for i, s in enumerate(placeholders):
        text = text.replace(f"@@PH{i}@@", s)
    return text


def convert_table(block_lines):
    rows = [l for l in block_lines if l.strip().startswith("|")]
    header, align_row, body_rows = rows[0], rows[1], rows[2:]

    def split_row(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]

    header_cells = split_row(header)
    align_cells = split_row(align_row)
    body = [split_row(r) for r in body_rows]
    n = len(header_cells)
    col_spec = []
    for a in align_cells:
        a = a.strip()
        if a.startswith(":") and a.endswith(":"):
            col_spec.append("c")
        elif a.endswith(":"):
            col_spec.append("r")
        elif a.startswith(":"):
            col_spec.append("l")
        else:
            col_spec.append("c" if "-" in a else "l")
    colspec = "".join(col_spec)
    wide = n >= 7
    out = [r"\begin{table}[t]", r"\centering"]
    if wide:
        out.append(r"\resizebox{\textwidth}{!}{%")
    out += [r"\small", r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    out.append(" & ".join(protect_and_convert(c) for c in header_cells) + r" \\")
    out.append(r"\midrule")
    for row in body:
        cells = [protect_and_convert(c) for c in row]
        cells = (cells + [""] * n)[:n]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if wide:
        out.append("}")
    out.append(r"\end{table}")
    return "\n".join(out)


def replace_tables(tex):
    parts = tex.split("@@TABLE@@")
    out = [parts[0]]
    for part in parts[1:]:
        table_src, rest = part.split("@@ENDTABLE@@", 1)
        out.append(convert_table(table_src.strip("\n").splitlines()))
        out.append(rest)
    return "".join(out)


def convert_body(md_text):
    lines = md_text.splitlines()
    out, i, n = [], 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if stripped == "":
            i += 1
            continue
        if stripped.startswith("### "):
            out.append(r"\subsection{" + protect_and_convert(strip_number_prefix(stripped[4:])) + "}")
            i += 1
            continue
        if stripped.startswith("## "):
            out.append(r"\section{" + protect_and_convert(strip_number_prefix(stripped[3:])) + "}")
            i += 1
            continue
        if stripped.startswith("$$"):
            if stripped.count("$$") >= 2 and len(stripped) > 2:
                math_src, i = stripped, i + 1
            else:
                block = [stripped]
                i += 1
                while i < n and "$$" not in lines[i]:
                    block.append(lines[i]); i += 1
                if i < n:
                    block.append(lines[i]); i += 1
                math_src = "\n".join(block)
            math_src = math_src.strip()
            math_src = math_src[2:] if math_src.startswith("$$") else math_src
            math_src = math_src[:-2] if math_src.endswith("$$") else math_src
            out.append(r"\[" + math_src.strip() + r"\]")
            continue
        if stripped.startswith("**Figure"):
            para = [stripped]
            i += 1
            while i < n and lines[i].strip() != "":
                para.append(lines[i].strip()); i += 1
            text = " ".join(para)
            m = re.match(r"\*\*Figure (\d+)\.\*\*\s*(.*)", text)
            fignum, rest = m.group(1), m.group(2)
            src_m = re.search(r"Source: `([^`]+)`\.?$", rest)
            src = src_m.group(1) if src_m else None
            caption_text = re.sub(r"\s*Source: `[^`]+`\.?$", "", rest).strip()
            fname = Path(src).stem if src else f"figure{fignum}"
            out += [
                r"\begin{figure}[t]", r"\centering",
                r"\includegraphics[width=\textwidth]{figures/" + fname + ".pdf}",
                r"\caption{" + protect_and_convert(caption_text) + "}",
                r"\label{fig:" + fname + "}", r"\end{figure}",
            ]
            continue
        if stripped.startswith("|"):
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            out += ["@@TABLE@@", "\n".join(block), "@@ENDTABLE@@"]
            continue
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n and (
                re.match(r"^\d+\.\s+", lines[i].strip())
                or (lines[i].strip() and items and lines[i].startswith("   "))
            ):
                if re.match(r"^\d+\.\s+", lines[i].strip()):
                    items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(r"\begin{enumerate}")
            out += [r"\item " + protect_and_convert(it) for it in items]
            out.append(r"\end{enumerate}")
            continue
        if stripped.startswith("- "):
            items = []
            while i < n and (
                lines[i].strip().startswith("- ")
                or (
                    lines[i].strip() and items
                    and not lines[i].strip().startswith(("#", "|", "**Figure", "$$", "-"))
                    and lines[i].startswith("  ")
                )
            ):
                if lines[i].strip().startswith("- "):
                    items.append(lines[i].strip()[2:])
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(r"\begin{itemize}")
            out += [r"\item " + protect_and_convert(it) for it in items]
            out.append(r"\end{itemize}")
            continue
        para = [stripped]
        i += 1
        while (
            i < n and lines[i].strip() != ""
            and not lines[i].strip().startswith(("#", "|", "**Figure", "$$", "- "))
            and not re.match(r"^\d+\.\s+", lines[i].strip())
        ):
            para.append(lines[i].strip()); i += 1
        out.append(protect_and_convert(" ".join(para)))
    return "\n\n".join(out)


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage{enumitem}
\usepackage{csquotes}
\MakeOuterQuote{"}
\emergencystretch=2.5em
\sloppy

\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}

\title{ElasticMLA: Context-Aware Token-Wise Latent Capacity Allocation for Multi-Head Latent Attention}
\author{ElasticMLA Project}
\date{}

\begin{document}
\maketitle

\begin{abstract}
"""


def main():
    mt = DRAFT.read_text()
    abs_start = mt.index("## Abstract")
    abs_end = mt.index("## 1. Introduction")
    abstract_text = mt[abs_start + len("## Abstract"):abs_end].strip()
    abs_paragraphs = [p.strip() for p in abstract_text.split("\n\n") if p.strip()]
    abstract_tex = "\n\n".join(protect_and_convert(" ".join(p.split())) for p in abs_paragraphs)

    idx_intro = mt.index("## 1. Introduction")
    body_tex = convert_body(mt[idx_intro:])
    body_tex = replace_tables(body_tex)
    ref_sec_idx = body_tex.index(r"\section{References}")
    body_tex_final = body_tex[:ref_sec_idx].rstrip()

    refs_md = mt[mt.index("## References"):]
    ref_entries = re.findall(r"\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)", refs_md, re.S)
    bib_items = [(num, protect_and_convert(" ".join(entry.strip().split()))) for num, entry in ref_entries]
    bib_tex = "\\begin{thebibliography}{9}\n\n" + "".join(
        f"\\bibitem{{ref{n}}} {e}\n\n" for n, e in bib_items
    ) + "\\end{thebibliography}\n"

    full_tex = PREAMBLE + abstract_tex + "\n\\end{abstract}\n\n" + body_tex_final + "\n\n" + bib_tex + "\n\\end{document}\n"

    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    (LATEX_DIR / "main.tex").write_text(full_tex)
    figures_dir = LATEX_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)
    for fname in ("elasticmla_main_results.pdf", "elasticmla_risk_spectrum.pdf"):
        shutil.copy(ROOT / "figures" / fname, figures_dir / fname)
    print(f"wrote {LATEX_DIR / 'main.tex'} ({len(full_tex)} chars)")


if __name__ == "__main__":
    main()
