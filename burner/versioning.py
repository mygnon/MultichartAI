"""v{n} resolution: idempotent reruns reuse the version, any content change
bumps it.  Old versions are never overwritten."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Tuple


def _existing_versions(out_dir: Path, name: str, inst: str, tf: str):
    pat = re.compile(re.escape(f"{name}_{inst}_{tf}_v") + r"(\d+)\.manifest\.json$")
    found = {}
    if out_dir.exists():
        for p in out_dir.iterdir():
            m = pat.match(p.name)
            if m:
                found[int(m.group(1))] = p
    return found


def resolve_version(out_dir: Path, name: str, inst: str, tf: str,
                    render_at: Callable[[int], str],
                    manifest_core: dict) -> Tuple[int, bool]:
    """(version, reused).  render_at(n) must return the full .txt content the
    burn would produce at version n.  manifest_core = the identity fields
    {params, exit_modules, source_state_sha256} of the candidate burn."""
    versions = _existing_versions(out_dir, name, inst, tf)
    if not versions:
        return 1, False
    latest = max(versions)
    txt_path = out_dir / f"{name}_{inst}_{tf}_v{latest}.txt"
    if txt_path.exists():
        try:
            old_manifest = json.loads(versions[latest].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old_manifest = {}
        old_core = {k: old_manifest.get(k)
                    for k in ("params", "exit_modules", "source_state_sha256")}
        if old_core == manifest_core and \
                txt_path.read_text(encoding="utf-8") == render_at(latest):
            return latest, True
    return latest + 1, False
