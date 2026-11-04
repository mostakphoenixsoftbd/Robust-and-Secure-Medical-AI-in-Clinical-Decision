"""Check that every number in the manuscript comes from the pipeline's shared outputs.

    python check_numbers.py ../manuscript/methods.md          # Markdown (one or more files)
    python check_numbers.py manuscript.docx                    # Word (needs python-docx)

Sources of truth (in SHARE_DIR, filled by share_results.py): <task>_results_<task>.json (smoke runs ending
in _smoke never count), <task>_cohort_summary.json and <task>_training_log.json, plus numbers_allowlist.txt
next to this script for fixed design constants (budgets, attack steps, subset sizes, split, dataset
versions ...). Results never go there.

Ported from the sibling sepsis project (same author); only the source-file names differ.

A number passes if some source value, rounded to the decimals the manuscript shows, equals it; proportions
also match as percentages (0.8123 -> 81.2 or 81.23%). Skipped: the reference list, years, citation
brackets like [12] or [3-5], table/figure/section numbers, version strings like 2.1.0, integers up to
--min-int (default 10), the level in "95% CI", identifiers (DOI, ORCID, URL, e-mail), and placeholders
[[R: ...]] (these are listed separately: a manuscript with placeholders is not finished).

Writes check_numbers_report.csv in SHARE_DIR. Prints PASS and returns 0 only if nothing is flagged.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import config as C

ALLOWLIST_FILE = Path(__file__).with_name("numbers_allowlist.txt")
SOURCE_GLOBS = ("*results*.json", "*cohort_summary.json", "*training_log.json")
RETIRED: set[str] = set()          # numbers known to be wrong (none yet for this project)

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:st|nd|rd|th)?(?![\w])(\s?%)?")
HEADING_NUMBER = re.compile(r"^\s*#*\s*\d+(?:\.\d+)*\.?(?=\s+\S)")
VERSION = re.compile(r"\b\d+\.\d+\.\d+\b")
CITATION = re.compile(r"\[\s*\d+(?:\s*[,–—-]\s*\d+)*\s*\]")
PLACEHOLDER = re.compile(r"\[\[(?:R|P|CITE|CHECK):[^\]]*\]\]")   # any open item blocks PASS
LABEL_BEFORE = re.compile(r"(?:Table|Tables|Figure|Figures|Fig\.|Eq\.|Equation|Section|Step|Phase|"
                          r"Supplementary|Appendix|Algorithm|Prompt)\s*$", re.I)
CI_LEVEL = re.compile(r"\s?%\s*(?:CI|confidence)", re.I)
REFERENCES_HEADING = re.compile(r"^\s*#*\s*(references|bibliography|reference list)\s*$", re.I)
IDENTIFIER = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b|\b10\.\d{4,9}/\S+|https?://\S+|\S+@\S+|PMID:?\s*\d+", re.I)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def flatten_numbers(obj) -> list[float]:
    out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out += flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += flatten_numbers(v)
    return out


def source_paths(out_dir: Path) -> list[Path]:
    found = {p for g in SOURCE_GLOBS for p in out_dir.glob(g) if not p.stem.endswith("_smoke")}
    return sorted(found)


def load_sources(out_dir: Path, allowlist: Path) -> tuple[list[float], list[str]]:
    values, used = [], []
    for p in source_paths(out_dir):
        values += flatten_numbers(json.loads(p.read_text(encoding="utf-8")))
        used.append(p.name)
    if allowlist.exists():
        for line in allowlist.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                values.append(float(line.replace(",", "")))
        used.append(allowlist.name)
    return values, used


def matches(token: str, decimals: int, values: list[float]) -> bool:
    x = float(token)
    tol = 1e-9 + 0.5 * 10 ** -(decimals + 6)
    for v in values:
        if abs(round(v, decimals) - x) < tol or abs(round(abs(v), decimals) - x) < tol:
            return True
        if abs(v) <= 2 and abs(round(abs(v) * 100, decimals) - x) < tol:
            return True
    return False


# ---------------------------------------------------------------------------
# Manuscript
# ---------------------------------------------------------------------------
def markdown_text(path: Path) -> list[tuple[str, str]]:
    items = []
    in_code = False
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line.strip() or line.lstrip().startswith(">"):   # blockquotes = editing notes
            continue
        if REFERENCES_HEADING.match(line):
            break
        text = line
        if line.lstrip().startswith("#"):
            text = HEADING_NUMBER.sub(lambda m: " " * len(m.group()), text)
        items.append((f"{path.name} line {i}", text))
    return items


def docx_text(path: Path) -> list[tuple[str, str]]:
    import docx
    d = docx.Document(str(path))
    items = []
    for i, p in enumerate(d.paragraphs, 1):
        if REFERENCES_HEADING.match(p.text):
            break
        if p.text.strip():
            text = p.text
            if p.style.name.lower().startswith("heading") or (len(text) < 100 and not text.rstrip().endswith(".")):
                text = HEADING_NUMBER.sub(lambda m: " " * len(m.group()), text)
            items.append((f"paragraph {i}", text))
    for t, table in enumerate(d.tables, 1):
        for r, row in enumerate(table.rows, 1):
            seen = set()
            for c, cell in enumerate(row.cells, 1):
                if id(cell._tc) in seen or not cell.text.strip():
                    continue
                seen.add(id(cell._tc))
                items.append((f"table {t} row {r} col {c}", cell.text))
    return items


def manuscript_text(path: Path) -> list[tuple[str, str]]:
    return docx_text(path) if path.suffix.lower() == ".docx" else markdown_text(path)


def numbers_in(text: str, min_int: int):
    masked = PLACEHOLDER.sub(lambda m: " " * len(m.group()), text)
    for rx in (IDENTIFIER, VERSION, CITATION):
        masked = rx.sub(lambda m: " " * len(m.group()), masked)
    for m in NUM.finditer(masked):
        whole, frac, pct = m.group(1), m.group(2) or "", m.group(3) or ""
        if LABEL_BEFORE.search(masked[: m.start()]):
            continue
        if CI_LEVEL.match(masked[m.end(1):]):
            continue
        token = whole.replace(",", "") + frac
        decimals = len(frac) - 1 if frac else 0
        if not frac:
            n = int(token)
            if n <= min_int or (1900 <= n <= 2100 and not pct and "," not in whole):
                continue
        start = max(0, m.start() - 40)
        yield token, decimals, pct.strip(), text[start: m.end() + 40].replace("\n", " ")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manuscript", type=Path, nargs="+")
    ap.add_argument("--min-int", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=C.SHARE_DIR)
    ap.add_argument("--allowlist", type=Path, default=ALLOWLIST_FILE)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    values, used = load_sources(args.out_dir, args.allowlist)
    if not any("results" in u for u in used):
        print("note: no results.json found yet; every result number will be flagged")
    print(f"sources: {', '.join(used) or 'none'} ({len(values):,} values)")

    flagged, placeholders, checked = [], [], 0
    for doc in args.manuscript:
        for where, text in manuscript_text(doc):
            for ph in PLACEHOLDER.findall(text):
                placeholders.append((where, ph, "PLACEHOLDER", ph))
            for token, dec, pct, context in numbers_in(text, args.min_int):
                checked += 1
                shown = token + (pct and "%")
                if token in RETIRED:
                    flagged.append((where, shown, "RETIRED", context))
                elif not matches(token, dec, values):
                    flagged.append((where, shown, "NOT IN RESULTS", context))

    report = args.out_dir / "check_numbers_report.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["location", "number", "problem", "context"])
        w.writerows(flagged + placeholders)
    print(f"checked {checked:,} numbers; flagged {len(flagged):,}; placeholders {len(placeholders):,}")
    for where, shown, why, context in flagged[:40]:
        print(f"  [{why}] {shown:>10}  {where}: ...{context}...")
    if len(flagged) > 40:
        print(f"  ... {len(flagged) - 40:,} more in {report}")
    if not flagged and not placeholders:
        print("PASS: every number traces to the pipeline outputs.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
