"""One-time export: Knowledge/*.docx (canonical module EL sources) ->
Strategy/modules/*.txt (stable text inputs for the burner).

The docx files are the only copy of the exit-module PowerLanguage sources in
the repo; the burner needs plain text.  tests/test_module_sources.py re-runs
this extraction and diffs it against the committed .txt so they cannot drift.

Run:  py -m burner.tools.export_modules_from_docx
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import docx  # python-docx

REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE = REPO_ROOT / "Knowledge"
MODULES_DIR = REPO_ROOT / "Strategy" / "modules"

# docx stem == signal name == output .txt stem
MODULE_DOCX = [
    "SFJ_15Dworkshop_lesson4_ATRstop",
    "SFJ_15Dworkshop_lesson9_1_TrailingStop",
    "SFJ_15Dworkshop_lesson10_3_EntryBarsAfterExit",
    "SFJ_15Dworkshop_lesson11_3_high_volatility_exit",
    "QuantPass_PT_Exit",
    "RescueTeamExit",
    "_Crypto1MUSD",
]

_NORMALIZE = {
    " ": " ",   # non-breaking space
    "“": '"', "”": '"',   # curly double quotes
    "‘": "'", "’": "'",   # curly single quotes
    "–": "-", "—": "-",   # en/em dash
    "　": " ",   # ideographic space
}


def extract_el_text(docx_path: Path) -> str:
    d = docx.Document(str(docx_path))
    lines = [p.text for p in d.paragraphs]
    text = "\n".join(lines)
    for bad, good in _NORMALIZE.items():
        text = text.replace(bad, good)
    text = text.rstrip("\n") + "\n"
    non_ascii = sorted({c for c in text if ord(c) > 127})
    if non_ascii:
        raise ValueError(
            f"{docx_path.name}: non-ASCII chars survive normalization: "
            + ", ".join(f"U+{ord(c):04X}" for c in non_ascii))
    return text


def export_all(out_dir: Path = MODULES_DIR) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for stem in MODULE_DOCX:
        src = KNOWLEDGE / f"{stem}.docx"
        if not src.exists():
            raise FileNotFoundError(src)
        text = extract_el_text(src)
        dst = out_dir / f"{stem}.txt"
        dst.write_text(text, encoding="ascii", newline="\n")
        written[stem] = dst
        print(f"exported {src.name} -> {dst.relative_to(REPO_ROOT)} ({len(text)} chars)")
    return written


if __name__ == "__main__":
    export_all()
    sys.exit(0)
