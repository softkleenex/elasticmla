#!/usr/bin/env python3
"""Build the anonymized supplementary archive for the double-blind TMLR submission.

The archive holds the code, tests, committed result artifacts and lab notes that the
paper cites by path (`experiments/...`, `notes/...`), taken from the files git tracks
at HEAD -- never from the working tree, so uncommitted edits cannot leak in.
Left out: ``archive/`` (superseded, never cited), ``papers/``, ``manuscript/`` (the
paper is submitted separately) and ``CLAUDE.md`` (agent instructions).

Anonymization replaces each identifying token below. Every replacement changes that
file's bytes, so its SHA-256 no longer matches what other artifacts record about it;
the archive's README lists each such file with its original hash, and whether any
other file in the archive references that hash, so a reviewer can tell an
anonymization edit from a provenance failure. The build refuses to finish if any
identifying token survives anywhere in the archive.
"""
import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PREFIXES = ("archive/", "papers/", "manuscript/")
EXCLUDED_FILES = {"CLAUDE.md"}
REPLACEMENTS = {"softkleenex": "anonymous-author"}
# Checked after replacement; any hit aborts the build.
FORBIDDEN = ("softkleenex", "isangjae", "heyeun", "knu.ac.kr", "dltkdwo", "shangjae",
             "/Volumes/samsd", "/Users/")
ZIP_DATE = (2026, 1, 1, 0, 0, 0)


def tracked_files():
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                             capture_output=True).stdout.decode()
    return sorted(path for path in listing.split("\0") if path
                  and not path.startswith(EXCLUDED_PREFIXES) and path not in EXCLUDED_FILES)


def head_bytes(path):
    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True,
                          capture_output=True).stdout


def anonymize(data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("utf-8")


def find_forbidden(data):
    lowered = data.lower()
    return [token for token in FORBIDDEN if token.lower().encode() in lowered]


def readme(changed, contents):
    lines = [
        "# ElasticMLA -- anonymized supplementary material",
        "",
        "Code (`code/`, `experiments/*.py`), unit tests (`tests/`), every committed result",
        "artifact the paper cites (`experiments/**.json`, `*.log`), figures and the dated lab",
        "notebook (`notes/`; read `notes/README.md` first -- later notes correct earlier ones).",
        "",
        "    uv sync",
        "    uv run python -m unittest discover -s tests",
        "",
        "Checkpoints and tokenized data are not included (size); every result JSON records",
        "their SHA-256. `README.md` describes the pipeline.",
        "",
        "## Files changed by anonymization",
        "",
        "A cloud username was replaced by `anonymous-author` in the files below, which",
        "changes their SHA-256. Original hash, and whether another archived file records it:",
        "",
    ]
    for path, original in changed:
        referenced = [other for other, data in contents.items()
                      if other != path and original.encode() in data]
        note = f"referenced by {', '.join(referenced)}" if referenced else "not referenced"
        lines.append(f"- `{path}` -- original sha256 `{original}` ({note})")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "elasticmla_supplement.zip")
    args = parser.parse_args()

    contents, changed = {}, []
    for path in tracked_files():
        original = head_bytes(path)
        data = anonymize(original)
        if data != original:
            changed.append((path, hashlib.sha256(original).hexdigest()))
        contents[path] = data
    contents["SUPPLEMENT_README.md"] = readme(changed, contents).encode()

    leaks = {path: hits for path, data in contents.items() if (hits := find_forbidden(data))}
    if leaks:
        raise SystemExit(f"identifying tokens remain: {leaks}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(contents):
            info = zipfile.ZipInfo(f"elasticmla_supplement/{path}", ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, contents[path])
    size = args.output.stat().st_size
    print(f"wrote {args.output}: {len(contents)} files, {size / 2**20:.1f} MiB, "
          f"{len(changed)} anonymized, sha256 {hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
