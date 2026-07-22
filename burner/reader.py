"""state.json -> BurnSource.  Strict: missing stages raise, no silent defaults."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .instruments import InstrumentCtx


class BurnInputError(Exception):
    pass


@dataclass(frozen=True)
class KeptModule:
    label: str                 # "M5"
    signal: str                # MC signal name
    params: Dict[str, float]   # stage3 optimal params


@dataclass(frozen=True)
class BurnSource:
    ctx: InstrumentCtx
    state_path: Path
    state_sha256: str
    main_params: Dict[str, float]      # stage2.main_champ verbatim
    winner: Dict                       # stage2.winner (manifest oos fields)
    kept: List[KeptModule] = field(default_factory=list)  # final_kept order
    stage4_final: Dict = field(default_factory=dict)      # np / midd / romad
    timeframe: str = "hourly"


def _stage4_final(stage4: Dict) -> Dict:
    """Deploy expectation = last valid KEEP step; empty final_kept -> baseline.
    (The last element of steps[] is NOT guaranteed to be a KEEP.)"""
    keeps = [s for s in stage4.get("steps", [])
             if s.get("decision") == "KEEP" and s.get("valid")]
    if keeps:
        last = keeps[-1]
        return {"net_profit": last["net_profit"],
                "max_intraday_drawdown": last["max_intraday_drawdown"],
                "romad": last["romad"]}
    base = stage4.get("baseline") or {}
    np_ = base.get("net_profit")
    midd = base.get("max_intraday_drawdown")
    if np_ is None or midd is None:
        raise BurnInputError("stage4 has neither a valid KEEP step nor a baseline")
    romad = round(np_ / abs(midd), 4) if midd else 0.0
    return {"net_profit": np_, "max_intraday_drawdown": midd, "romad": romad}


def load_burn_source(state_path: Path, ctx: InstrumentCtx, tf: str,
                     no_modules: bool = False) -> BurnSource:
    if not state_path.exists():
        raise BurnInputError(f"state.json not found: {state_path}")
    raw = state_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    state = json.loads(raw.decode("utf-8"))

    stage2 = state.get("stage2")
    if not stage2 or not stage2.get("main_champ"):
        raise BurnInputError(f"{state_path}: stage2.main_champ missing")
    main_params = dict(stage2["main_champ"])
    winner = stage2.get("winner")
    if not winner:
        raise BurnInputError(f"{state_path}: stage2.winner missing")

    kept: List[KeptModule] = []
    if no_modules:
        stage4 = state.get("stage4")
        if not stage4:
            raise BurnInputError(f"{state_path}: stage4 missing")
    else:
        stage3, stage4 = state.get("stage3"), state.get("stage4")
        if not stage3 or not stage4:
            raise BurnInputError(f"{state_path}: stage3/stage4 missing "
                                 f"(pipeline died mid-run?)")
        mods = stage3.get("modules") or {}
        for label in stage4.get("final_kept") or []:
            info = mods.get(label)
            if not info:
                raise BurnInputError(f"{state_path}: final_kept {label} "
                                     f"absent from stage3.modules")
            kept.append(KeptModule(label=label, signal=info["signal"],
                                   params=dict(info["params"])))

    return BurnSource(ctx=ctx, state_path=state_path, state_sha256=sha,
                      main_params=main_params, winner=dict(winner), kept=kept,
                      stage4_final=_stage4_final(stage4), timeframe=tf)
