"""CLI.

  py -m burner burn --name DualAnchorBreakout --key dualanchor [--tf hourly]
                    [--inst all|btc,gc] [--no-modules] [--dry-run]
  py -m burner verify --name DualAnchorBreakout --key dualanchor [--inst ...]
                    [--dry-run] [--pipeline-script <path>] [--tolerance 0.005]
  py -m burner export-modules
"""
from __future__ import annotations

import argparse
import sys

from .instruments import INSTRUMENTS


def _insts(arg: str):
    if arg == "all":
        return list(INSTRUMENTS)
    keys = [k.strip().lower() for k in arg.split(",") if k.strip()]
    bad = [k for k in keys if k not in INSTRUMENTS]
    if bad:
        raise SystemExit(f"unknown instrument key(s): {bad}; "
                         f"valid: {', '.join(INSTRUMENTS)}")
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="burner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("burn", help="burn state.json -> merged EL + manifest")
    b.add_argument("--name", required=True, help="base strategy, e.g. DualAnchorBreakout")
    b.add_argument("--key", required=True, help="pipeline key, e.g. dualanchor")
    b.add_argument("--tf", default="hourly", choices=["hourly", "daily", "240"])
    b.add_argument("--inst", default="all", help="all | comma list (btc,gc,...)")
    b.add_argument("--no-modules", action="store_true",
                   help="burn the bare main only (ignore stage3/4 kept modules)")
    b.add_argument("--dry-run", action="store_true", help="validate, write nothing")

    v = sub.add_parser("verify", help="equivalence verification (A/B, MC64)")
    v.add_argument("--name", required=True)
    v.add_argument("--key", required=True)
    v.add_argument("--tf", default="hourly", choices=["hourly", "daily", "240"])
    v.add_argument("--inst", default="all")
    v.add_argument("--dry-run", action="store_true",
                   help="emit equivalence_checklist.json only (no MC64)")
    v.add_argument("--pipeline-script", default=None,
                   help="pipeline .py to scrape WORKSPACE/date-range constants from")
    v.add_argument("--tolerance", type=float, default=0.005)

    sub.add_parser("export-modules", help="Knowledge/*.docx -> Strategy/modules/*.txt")

    args = ap.parse_args(argv)

    if args.cmd == "export-modules":
        from .tools.export_modules_from_docx import export_all
        export_all()
        return 0

    if args.cmd == "burn":
        from .burn import burn_all
        results = burn_all(args.name, args.key, args.tf, _insts(args.inst),
                           no_modules=args.no_modules, dry_run=args.dry_run)
        failed = 0
        for r in results:
            mark = {"burned": "OK ", "reused": "== ", "failed": "FAIL"}[r.status]
            print(f"[{mark}] {r.inst:<4} {r.strategy_id or '-'}")
            for e in r.errors:
                print(f"        {e}")
            failed += r.status == "failed"
        print(f"\n{len(results) - failed}/{len(results)} instruments burned"
              + (" (dry-run)" if args.dry_run else ""))
        return 1 if failed else 0

    if args.cmd == "verify":
        from .equivalence import run_verify
        return run_verify(args.name, args.key, args.tf, _insts(args.inst),
                          dry_run=args.dry_run,
                          pipeline_script=args.pipeline_script,
                          tolerance=args.tolerance)
    return 2


if __name__ == "__main__":
    sys.exit(main())
