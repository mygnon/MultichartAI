"""Final .txt text = AUTOGEN header + assembled body."""
from __future__ import annotations

from typing import List

from . import templates
from .reader import BurnSource


def render(strategy_id: str, body: str, src: BurnSource, main_source: str,
           template_version: str) -> str:
    if src.kept:
        modules_line = ", ".join(f"{m.label}={m.signal}" for m in src.kept)
    else:
        modules_line = "(none)"
    header = templates.HEADER_TMPL.substitute(
        strategy_id=strategy_id,
        source_state_sha256=src.state_sha256,
        template_version=template_version,
        main_source=main_source,
        modules_line=modules_line,
    )
    text = header + "\n" + body
    if not text.endswith("\n"):
        text += "\n"
    return text
