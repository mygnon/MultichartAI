"""Post-render, pre-compile gates (spec 4.3).  Any failure -> no artifacts."""
from __future__ import annotations

import re
from typing import Dict, List

from . import el_lex

STRATEGY_ID_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9]+)_(?P<inst>BTC|ETH|BNB|TXF|NQ|GC)_"
    r"(?P<tf>H1|D1|M240)_v(?P<ver>\d+)$")

# External PL functions the burned output must NOT depend on (spec: zero
# external dependencies).  If one of these appears it must be a declared var
# (the inlined-variable idiom), never an undeclared function call.
KNOWN_EXTERNAL_FUNCS = {"_crypto1musd"}


def validate(el_text: str, manifest: Dict, strategy_name: str) -> List[str]:
    errors: List[str] = []

    # -- gate 1: structural sanity ------------------------------------------
    if el_text.startswith("﻿"):
        errors.append("BOM at start of file")
    if not el_text.endswith("\n"):
        errors.append("file does not end with a newline")
    try:
        toks = el_lex.lex(el_text)
    except ValueError as e:
        return errors + [f"lex failure: {e}"]
    depth = 0
    for t in toks:
        if t.kind == "other" and t.text in "{}":
            errors.append(f"stray {t.text!r} at offset {t.start} (unbalanced comment)")
        if t.kind == "other" and t.text == "(":
            depth += 1
        elif t.kind == "other" and t.text == ")":
            depth -= 1
            if depth < 0:
                errors.append(f"unbalanced ')' at offset {t.start}")
                depth = 0
    if depth != 0:
        errors.append(f"unbalanced '(' x{depth}")

    # -- gate 2: round-trip read-back (float-exact) -------------------------
    inputs, vars_ = el_lex.parse_declarations(el_text)
    inputs_ci = {k.lower(): v for k, v in inputs.items()}
    expected: Dict[str, float] = dict(manifest.get("params") or {})
    for em in manifest.get("exit_modules") or []:
        prefix = em["id"].lower() + "_"
        for p, v in (em.get("params") or {}).items():
            expected[prefix + p] = v
    for name, want in expected.items():
        raw = inputs_ci.get(name.lower())
        if raw is None:
            errors.append(f"round-trip: input {name!r} not declared in output")
            continue
        try:
            got = float(raw)
        except ValueError:
            errors.append(f"round-trip: input {name!r} default {raw!r} not numeric")
            continue
        if got != float(want):
            errors.append(f"round-trip: input {name!r} default {got} != manifest {want}")

    # -- gate 3: case-insensitive uniqueness --------------------------------
    def _dups(names):
        seen, dups = set(), []
        for n in names:
            k = n.lower()
            if k in seen:
                dups.append(n)
            seen.add(k)
        return dups

    for kind, names in (("input", list(inputs)), ("var", list(vars_)),
                        ("order name", el_lex.find_order_names(el_text))):
        for d in _dups(names):
            errors.append(f"duplicate {kind} (case-insensitive): {d!r}")
    both = {n.lower() for n in inputs} & {n.lower() for n in vars_}
    for n in sorted(both):
        errors.append(f"name declared as both input and var: {n!r}")

    # -- gate 4: external-function dependency check -------------------------
    declared = {n.lower() for n in inputs} | {n.lower() for n in vars_}
    for t in toks:
        if t.kind == "ident" and t.text.lower() in KNOWN_EXTERNAL_FUNCS \
                and t.text.lower() not in declared:
            errors.append(f"external PL function referenced but not declared "
                          f"as a var: {t.text!r} at offset {t.start}")
            break

    # -- gate 5: strategy_id format -----------------------------------------
    sid = manifest.get("strategy_id", "")
    m = STRATEGY_ID_RE.match(sid)
    if not m:
        errors.append(f"strategy_id {sid!r} does not match the naming rule")
    elif m.group("name") != strategy_name:
        errors.append(f"strategy_id name part {m.group('name')!r} != {strategy_name!r}")
    if sid and sid not in el_text:
        errors.append("strategy_id missing from the rendered EL header")

    return errors
