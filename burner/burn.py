"""Burn orchestration: state.json -> validated .txt + manifest per instrument."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from . import TEMPLATE_VERSION, assembler, renderer, static_validator, versioning
from .instruments import (INSTRUMENTS, REPO_ROOT, TF_MAP, InstrumentCtx,
                          main_source_path, state_json_path)
from .manifest_writer import (BurnResult, build_manifest, manifest_core,
                              write_burn_report, write_outputs)
from .module_library import load_module
from .reader import BurnInputError, load_burn_source

BURNED_ROOT = REPO_ROOT / "burned"


def burn_one(name: str, strat_key: str, tf: str, ctx: InstrumentCtx,
             no_modules: bool = False, dry_run: bool = False,
             burned_root: Path = BURNED_ROOT,
             burned_at: Optional[str] = None) -> BurnResult:
    res = BurnResult(inst=ctx.inst)
    burned_at = burned_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tf_code = TF_MAP[tf]
    out_dir = burned_root / name
    try:
        src = load_burn_source(state_json_path(ctx.key, strat_key, tf), ctx, tf,
                               no_modules=no_modules)
        main_path = main_source_path(name, ctx)
        if not main_path.exists():
            raise BurnInputError(f"main strategy source missing: {main_path}")
        main_el = main_path.read_text(encoding="utf-8")
        mods = [load_module(km.label, km.signal) for km in src.kept]

        def render_at(n: int) -> str:
            sid = f"{name}_{ctx.inst}_{tf_code}_v{n}"
            body = assembler.assemble(src, main_el, mods, sid, TEMPLATE_VERSION)
            return renderer.render(sid, body, src,
                                   f"{name}_{ctx.variant_suffix}", TEMPLATE_VERSION)

        version, reused = versioning.resolve_version(
            out_dir, name, ctx.inst, tf_code, render_at,
            manifest_core({"params": dict(src.main_params),
                           "exit_modules": [{"id": m.label, "signal": m.signal,
                                             "params": dict(m.params)}
                                            for m in src.kept],
                           "source_state_sha256": src.state_sha256}))
        sid = f"{name}_{ctx.inst}_{tf_code}_v{version}"
        el_text = render_at(version)
        manifest = build_manifest(src, sid, el_text, name, burned_at)
        errors = static_validator.validate(el_text, manifest, name)
        res.strategy_id = sid
        if errors:
            res.status = "failed"
            res.errors = errors
            return res
        if reused:
            res.status = "reused"
            res.files = [str(out_dir / f"{sid}.txt"),
                         str(out_dir / f"{sid}.manifest.json")]
            return res
        if dry_run:
            res.status = "burned"
            res.errors = ["(dry-run: nothing written)"]
            return res
        res.files = write_outputs(out_dir, sid, el_text, manifest)
        res.status = "burned"
        return res
    except (BurnInputError, FileNotFoundError, ValueError,
            assembler.AssemblyError) as e:
        res.errors = [str(e)]
        return res


def burn_all(name: str, strat_key: str, tf: str, insts: List[str],
             no_modules: bool = False, dry_run: bool = False,
             burned_root: Path = BURNED_ROOT) -> List[BurnResult]:
    burned_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = []
    for key in insts:
        ctx = INSTRUMENTS[key]
        results.append(burn_one(name, strat_key, tf, ctx, no_modules=no_modules,
                                dry_run=dry_run, burned_root=burned_root,
                                burned_at=burned_at))
    if not dry_run:
        write_burn_report(burned_root / name, name, burned_at, results)
    return results
