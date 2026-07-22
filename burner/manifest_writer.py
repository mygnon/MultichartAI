"""Manifest (spec section 3) + burn_report.json.  Atomic writes (tmp+replace)."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .instruments import REPO_ROOT
from .reader import BurnSource


@dataclass
class BurnResult:
    inst: str
    strategy_id: str = ""
    status: str = "failed"          # burned | reused | failed
    errors: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)


def el_sha256(el_text: str) -> str:
    return hashlib.sha256(el_text.encode("utf-8")).hexdigest()


def build_manifest(src: BurnSource, strategy_id: str, el_text: str,
                   name: str, burned_at: str) -> Dict:
    try:
        rel_state = src.state_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel_state = str(src.state_path)
    return {
        "schema": 1,
        "strategy_id": strategy_id,
        "base_strategy": name,
        "source_variant": f"{name}_{src.ctx.variant_suffix}",
        "symbol": src.ctx.symbol,
        "symbol_class": src.ctx.symbol_class,
        "timeframe": strategy_id.rsplit("_", 2)[-2],
        "params": dict(src.main_params),
        "exit_modules": [
            {"id": m.label, "signal": m.signal, "params": dict(m.params)}
            for m in src.kept
        ],
        "stage4_final": dict(src.stage4_final),
        "dependencies": [],
        "oos": {
            "net_profit": src.winner.get("oos_np"),
            "mdd": src.winner.get("mdd_full"),
            "pass": src.winner.get("pass"),
        },
        "source_state_json": rel_state,
        "source_state_sha256": src.state_sha256,
        "el_sha256": el_sha256(el_text),
        "burned_at": burned_at,
    }


def manifest_core(manifest: Dict) -> Dict:
    return {k: manifest.get(k)
            for k in ("params", "exit_modules", "source_state_sha256")}


def _atomic_write(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_outputs(out_dir: Path, strategy_id: str, el_text: str,
                  manifest: Dict) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / f"{strategy_id}.txt"
    mf = out_dir / f"{strategy_id}.manifest.json"
    _atomic_write(txt, el_text)
    _atomic_write(mf, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return [str(txt), str(mf)]


def write_burn_report(out_dir: Path, name: str, burned_at: str,
                      results: List[BurnResult]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "base_strategy": name,
        "burned_at": burned_at,
        "results": [{"inst": r.inst, "strategy_id": r.strategy_id,
                     "status": r.status, "errors": r.errors, "files": r.files}
                    for r in results],
    }
    path = out_dir / "burn_report.json"
    _atomic_write(path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return path
