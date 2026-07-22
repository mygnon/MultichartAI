"""Merge: main strategy (verbatim body, input defaults re-baked) + kept exit
modules (identifier-prefixed, orders named, declarations hoisted) + OMS block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from . import el_lex, templates
from .module_library import ModuleSource
from .reader import BurnSource


class AssemblyError(Exception):
    pass


@dataclass(frozen=True)
class TransformedModule:
    label: str
    signal: str
    decl_lines: str   # hoisted "inputs: ... ;" / "variables: ... ;" lines
    body: str         # decl-stripped, renamed, orders named, section-headed


def transform_module(mod: ModuleSource, params: Dict[str, float]) -> TransformedModule:
    prefix = mod.label.lower() + "_"
    declared = list(mod.declared_inputs) + list(mod.declared_vars)
    unknown = set(params) - set(mod.declared_inputs)
    if unknown:
        raise AssemblyError(f"{mod.label}: params {sorted(unknown)} not declared "
                            f"as inputs of {mod.signal}")
    mapping = {name: prefix + name for name in declared}

    # hoisted declarations, defaults baked from stage3 params
    in_items = []
    for name, raw in mod.declared_inputs.items():
        default = el_lex.format_number(params[name]) if name in params else raw
        in_items.append(f"{prefix}{name}( {default} )")
    decl_lines = f"inputs: {', '.join(in_items)} ;\n"
    if mod.declared_vars:
        var_items = [f"{prefix}{name}( {raw} )"
                     for name, raw in mod.declared_vars.items()]
        decl_lines += f"variables: {', '.join(var_items)} ;\n"

    # body = original minus declaration statements, then rename + name orders
    spans = [d.span for d in el_lex.find_declarations(mod.el_text)]
    body = el_lex.strip_spans(mod.el_text, spans)
    body = el_lex.rename_idents(body, mapping)
    body = el_lex.name_unnamed_orders(body, mod.label)

    # leak check: no original declared name may survive in the body
    declared_lower = {n.lower() for n in declared}
    for t in el_lex.lex(body):
        if t.kind == "ident" and t.text.lower() in declared_lower:
            raise AssemblyError(
                f"{mod.label}: unprefixed identifier {t.text!r} leaked into body")

    params_line = ", ".join(f"{k}={el_lex.format_number(v)}" for k, v in params.items())
    header = templates.MODULE_SECTION_TMPL.substitute(
        label=mod.label, signal=mod.signal, params_line=params_line)
    body = header + body.strip("\n") + "\n"
    return TransformedModule(label=mod.label, signal=mod.signal,
                             decl_lines=decl_lines, body=body)


def assemble(src: BurnSource, main_el: str, mods: List[ModuleSource],
             strategy_id: str, template_version: str) -> str:
    """Body of the burned signal (everything below the AUTOGEN header)."""
    main_defaults = {k: el_lex.format_number(v) for k, v in src.main_params.items()}
    main_text = el_lex.rewrite_input_defaults(main_el, main_defaults)

    by_signal = {m.signal: m for m in mods}
    transformed: List[TransformedModule] = []
    for km in src.kept:  # final_kept keep order
        if km.signal not in by_signal:
            raise AssemblyError(f"module source not loaded: {km.signal}")
        transformed.append(transform_module(by_signal[km.signal], km.params))

    hoist = ""
    if transformed:
        hoist += "{ ==== burned exit-module declarations (params baked from stage3) ==== }\n"
        for tm in transformed:
            hoist += tm.decl_lines
    hoist += templates.OMS_DECLS_TMPL.substitute(template_version=template_version)

    cut = el_lex.first_executable_offset(main_text)
    parts = [main_text[:cut], "\n", hoist, "\n", main_text[cut:]]
    if not main_text.endswith("\n"):
        parts.append("\n")
    for tm in transformed:
        parts.append("\n")
        parts.append(tm.body)
    parts.append("\n")
    parts.append(templates.OMS_BODY_TMPL.substitute(
        strategy_id=strategy_id, symbol=src.ctx.symbol))
    return "".join(parts)
