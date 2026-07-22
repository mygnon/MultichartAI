"""Minimal PowerLanguage (EasyLanguage) tokenizer + lexical transforms.

No parser: every transform is whole-token text surgery.  Comments ``{...}`` /
``//...`` and strings ``"..."`` are opaque to all transforms.  Identifier
renaming is a WHITELIST of declared names, matched case-insensitively as whole
tokens -- builtins are never in the mapping, so no keyword blacklist is needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

TOKEN_RE = re.compile(
    r"""(?P<comment_brace>\{[^}]*\})
      | (?P<comment_line>//[^\n]*)
      | (?P<string>"[^"\n]*")
      | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
      | (?P<number>\d+(?:\.\d+)?)
      | (?P<ws>\s+)
      | (?P<other>.)
    """,
    re.VERBOSE,
)

DECL_KEYWORDS = {"input", "inputs", "var", "vars", "variable", "variables"}
ORDER_VERBS = {"buy": "LE", "sellshort": "SE", "sell": "LX", "buytocover": "SX"}
_SKIP = {"comment_brace", "comment_line", "ws"}


@dataclass(frozen=True)
class Tok:
    kind: str
    text: str
    start: int

    @property
    def end(self) -> int:
        return self.start + len(self.text)


def lex(text: str) -> List[Tok]:
    toks: List[Tok] = []
    pos = 0
    for m in TOKEN_RE.finditer(text):
        if m.start() != pos:  # pragma: no cover - regex covers every char
            raise ValueError(f"lex gap at offset {pos}")
        toks.append(Tok(m.lastgroup, m.group(), m.start()))
        pos = m.end()
    if pos != len(text):  # pragma: no cover
        raise ValueError("lex did not consume input")
    return toks


@dataclass
class DeclItem:
    name: str
    default_raw: str            # text inside the parens, stripped
    default_span: Tuple[int, int]  # offsets of the raw default inside source


@dataclass
class Decl:
    kind: str                   # "input" | "var"
    items: List[DeclItem] = field(default_factory=list)
    span: Tuple[int, int] = (0, 0)  # statement span incl. trailing ';'


def _decl_kind(word: str) -> Optional[str]:
    w = word.lower()
    if w in ("input", "inputs"):
        return "input"
    if w in ("var", "vars", "variable", "variables"):
        return "var"
    return None


def find_declarations(text: str) -> List[Decl]:
    """All input/var declaration statements, in source order."""
    toks = lex(text)
    decls: List[Decl] = []
    i = 0
    n = len(toks)
    while i < n:
        # skip to next significant token (statement start)
        while i < n and toks[i].kind in _SKIP:
            i += 1
        if i >= n:
            break
        t = toks[i]
        kind = _decl_kind(t.text) if t.kind == "ident" else None
        # look ahead for ':'
        j = i + 1
        while j < n and toks[j].kind in _SKIP:
            j += 1
        if kind is not None and j < n and toks[j].text == ":":
            decl = Decl(kind=kind)
            start = t.start
            k = j + 1
            name: Optional[str] = None
            while k < n and toks[k].text != ";":
                tk = toks[k]
                if tk.kind == "ident":
                    name = tk.text  # last ident before '(' wins (type prefixes)
                elif tk.text == "(" and name is not None:
                    depth = 1
                    dstart = tk.end
                    k += 1
                    while k < n and depth > 0:
                        if toks[k].text == "(":
                            depth += 1
                        elif toks[k].text == ")":
                            depth -= 1
                            if depth == 0:
                                break
                        k += 1
                    dend = toks[k].start if k < n else len(text)
                    raw = text[dstart:dend]
                    lead = len(raw) - len(raw.lstrip())
                    trail = len(raw.rstrip())
                    decl.items.append(DeclItem(name=name,
                                               default_raw=raw.strip(),
                                               default_span=(dstart + lead, dstart + trail)))
                    name = None
                k += 1
            end = toks[k].end if k < n else len(text)
            decl.span = (start, end)
            decls.append(decl)
            i = k + 1
        else:
            # executable (or other) statement: skip to its terminating ';'
            k = i
            while k < n and toks[k].text != ";":
                k += 1
            i = k + 1
    return decls


def parse_declarations(text: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(inputs, vars): declared name -> raw default text."""
    inputs: Dict[str, str] = {}
    vars_: Dict[str, str] = {}
    for d in find_declarations(text):
        target = inputs if d.kind == "input" else vars_
        for it in d.items:
            target[it.name] = it.default_raw
    return inputs, vars_


def first_executable_offset(text: str) -> int:
    """Offset where the first non-declaration statement starts (insertion
    point for hoisted module declarations).  Comments before it stay above."""
    toks = lex(text)
    decl_spans = [d.span for d in find_declarations(text)]
    i = 0
    n = len(toks)
    while i < n:
        while i < n and toks[i].kind in _SKIP:
            i += 1
        if i >= n:
            return len(text)
        t = toks[i]
        in_decl = next((sp for sp in decl_spans if sp[0] <= t.start < sp[1]), None)
        if in_decl is None:
            return t.start
        while i < n and toks[i].start < in_decl[1]:
            i += 1
    return len(text)


def rename_idents(text: str, mapping: Dict[str, str]) -> str:
    """Whole-token, case-insensitive rename of declared identifiers.
    Comments and strings are untouched."""
    lower = {k.lower(): v for k, v in mapping.items()}
    out: List[str] = []
    for t in lex(text):
        if t.kind == "ident" and t.text.lower() in lower:
            out.append(lower[t.text.lower()])
        else:
            out.append(t.text)
    return "".join(out)


def rewrite_input_defaults(text: str, new_defaults: Dict[str, str]) -> str:
    """Replace the parenthesized default literal of matching INPUT names
    (case-insensitive).  Everything else is byte-preserved."""
    lower = {k.lower(): v for k, v in new_defaults.items()}
    repl: List[Tuple[int, int, str]] = []
    seen = set()
    for d in find_declarations(text):
        if d.kind != "input":
            continue
        for it in d.items:
            key = it.name.lower()
            if key in lower:
                repl.append((it.default_span[0], it.default_span[1], lower[key]))
                seen.add(key)
    missing = set(lower) - seen
    if missing:
        raise KeyError(f"inputs not found for default rewrite: {sorted(missing)}")
    out = []
    pos = 0
    for s, e, v in sorted(repl):
        out.append(text[pos:s])
        out.append(v)
        pos = e
    out.append(text[pos:])
    return "".join(out)


def name_unnamed_orders(text: str, label: str) -> str:
    """Give every unnamed order verb an explicit unique name:
    buy -> "{label}_LE", sellshort -> "_SE", sell -> "_LX", buytocover -> "_SX".
    Already-named orders (verb followed by ``("name")``) are untouched."""
    toks = lex(text)
    inserts: List[Tuple[int, str]] = []  # (offset, inserted text)
    n = len(toks)
    for i, t in enumerate(toks):
        if t.kind != "ident" or t.text.lower() not in ORDER_VERBS:
            continue
        j = i + 1
        while j < n and toks[j].kind in _SKIP:
            j += 1
        named = False
        if j < n and toks[j].text == "(":
            k = j + 1
            while k < n and toks[k].kind in _SKIP:
                k += 1
            named = k < n and toks[k].kind == "string"
        if not named:
            side = ORDER_VERBS[t.text.lower()]
            inserts.append((t.end, f' ( "{label}_{side}" )'))
    out = []
    pos = 0
    for off, ins in inserts:
        out.append(text[pos:off])
        out.append(ins)
        pos = off
    out.append(text[pos:])
    return "".join(out)


def find_order_names(text: str) -> List[str]:
    """String literals used as order names (verb followed by ``("name")``)."""
    toks = lex(text)
    names: List[str] = []
    n = len(toks)
    for i, t in enumerate(toks):
        if t.kind != "ident" or t.text.lower() not in ORDER_VERBS:
            continue
        j = i + 1
        while j < n and toks[j].kind in _SKIP:
            j += 1
        if j < n and toks[j].text == "(":
            k = j + 1
            while k < n and toks[k].kind in _SKIP:
                k += 1
            if k < n and toks[k].kind == "string":
                names.append(toks[k].text[1:-1])
    return names


def strip_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    """Remove [start,end) spans; collapse the leftover blank lines a bit."""
    out = []
    pos = 0
    for s, e in sorted(spans):
        out.append(text[pos:s])
        pos = e
    out.append(text[pos:])
    result = "".join(out)
    return re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", result)


def format_number(v: float) -> str:
    """Bake a float as an EL literal: integral -> int form, else clean repr."""
    if v == int(v):
        return str(int(v))
    return repr(round(float(v), 10))
