"""Strategy/modules/*.txt -> ModuleSource (verbatim EL + lexed declarations)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from . import el_lex
from .instruments import MODULE_REGISTRY, REPO_ROOT

MODULES_DIR = REPO_ROOT / "Strategy" / "modules"


@dataclass(frozen=True)
class ModuleSource:
    label: str
    signal: str
    el_text: str
    declared_inputs: Dict[str, str]  # name -> raw default text
    declared_vars: Dict[str, str]


def load_module(label: str, signal: str, modules_dir: Path = MODULES_DIR) -> ModuleSource:
    expected = MODULE_REGISTRY.get(label)
    if expected is not None and expected != signal:
        raise ValueError(f"{label}: state.json signal {signal!r} != registry {expected!r}")
    path = modules_dir / f"{signal}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"module source missing: {path} "
            f"(run: py -m burner.tools.export_modules_from_docx)")
    text = path.read_text(encoding="ascii")
    inputs, vars_ = el_lex.parse_declarations(text)
    if not inputs:
        raise ValueError(f"{path.name}: no input declarations found")
    return ModuleSource(label=label, signal=signal, el_text=text,
                       declared_inputs=inputs, declared_vars=vars_)
