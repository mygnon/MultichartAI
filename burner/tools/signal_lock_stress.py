"""Signal-file lock stress tool (acceptance test 4 of the I/O fix).

Simulates a worst-case OMS reader that holds an exclusive handle on a signal
.json for hundreds of milliseconds in a loop.  The v2 emit block's MoveFileExA
retry (5x / 30ms) must wait out the lock and publish without ever raising an
EL runtime error (which would disarm AOE).

  py -m burner.tools.signal_lock_stress --file "Z:\\oms\\signals\\<id>.json" ^
        --hold-ms 300 --interval 1.0 --duration 3600

  py -m burner.tools.signal_lock_stress --scan-dir "Z:\\oms\\signals"
        # audit mode: verify the *.json glob convention finds no .tmp files
        # and report any orphaned .tmp litter
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import sys
import time
from pathlib import Path

GENERIC_READ = 0x80000000
OPEN_EXISTING = 3
INVALID_HANDLE = wt.HANDLE(-1).value

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def _open_exclusive(path: str):
    """Open with dwShareMode=0: blocks every other open/rename-onto of it."""
    h = _kernel32.CreateFileW(path, GENERIC_READ, 0, None, OPEN_EXISTING, 0, None)
    return None if h == INVALID_HANDLE else h


def stress(path: Path, hold_ms: int, interval: float, duration: float) -> int:
    end = time.monotonic() + duration
    cycles = held = missed = 0
    print(f"locking {path} for {hold_ms}ms every {interval}s "
          f"for {duration:.0f}s -- watch PT: AOE must stay armed, "
          f"json must keep updating")
    last_mtime = None
    while time.monotonic() < end:
        cycles += 1
        h = _open_exclusive(str(path))
        if h is None:
            missed += 1  # file mid-rename or absent this instant: fine
        else:
            held += 1
            time.sleep(hold_ms / 1000.0)
            _kernel32.CloseHandle(h)
        try:
            m = path.stat().st_mtime
            if last_mtime is not None and m != last_mtime:
                print(f"  [{time.strftime('%H:%M:%S')}] signal updated "
                      f"(mtime changed) after {cycles} lock cycles")
            last_mtime = m
        except OSError:
            pass
        time.sleep(interval)
    print(f"done: {cycles} cycles, held lock {held}x, open-missed {missed}x")
    return 0


def scan_dir(d: Path) -> int:
    """Audit the reader convention: *.json must not match tmp files, and
    orphaned .tmp litter indicates the writer's cleanup failed."""
    jsons = sorted(d.glob("*.json"))
    bad = [p for p in jsons if ".tmp" in p.name]
    tmps = sorted(d.glob("*.tmp"))
    print(f"{d}: {len(jsons)} *.json matched, {len(tmps)} *.tmp present")
    for p in bad:
        print(f"  BAD GLOB MATCH (reader would ingest a tmp): {p.name}")
    now = time.time()
    stale = [p for p in tmps if now - p.stat().st_mtime > 60]
    for p in stale:
        print(f"  ORPHANED tmp (>60s old, writer cleanup failed): {p.name}")
    for p in jsons[:5]:
        try:
            with open(p, encoding="utf-8") as f:  # open-read-close immediately
                json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  UNREADABLE/PARTIAL json: {p.name}: {e}")
    return 1 if (bad or stale) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="signal .json to lock repeatedly")
    ap.add_argument("--hold-ms", type=int, default=300)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=3600.0)
    ap.add_argument("--scan-dir", help="audit mode: check glob/tmp hygiene")
    args = ap.parse_args(argv)
    if args.scan_dir:
        return scan_dir(Path(args.scan_dir))
    if not args.file:
        ap.error("--file or --scan-dir required")
    return stress(Path(args.file), args.hold_ms, args.interval, args.duration)


if __name__ == "__main__":
    sys.exit(main())
