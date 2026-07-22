"""Byte-exact golden: fixture main + fixed BurnSource -> committed golden EL.
Regenerate deliberately with:  py tests/regen_goldens.py  (then review diff)."""
from pathlib import Path

from burner import TEMPLATE_VERSION, assembler, renderer
from burner.instruments import MODULE_REGISTRY
from burner.module_library import load_module
from conftest import FIXTURES, MINIMAIN, make_src

GOLDEN = FIXTURES / "golden"


def build(kept):
    src = make_src(kept)
    sid = "MiniMain_BTC_H1_v1"
    mods = [load_module(lb, MODULE_REGISTRY[lb]) for lb in kept]
    body = assembler.assemble(src, MINIMAIN.read_text(encoding="utf-8"),
                              mods, sid, TEMPLATE_VERSION)
    return renderer.render(sid, body, src, "MiniMain_crypto", TEMPLATE_VERSION)


def test_golden_no_modules():
    expected = (GOLDEN / "MiniMain_nomodules.txt").read_text(encoding="utf-8")
    assert build([]) == expected


def test_golden_m5_m6():
    expected = (GOLDEN / "MiniMain_m5_m6.txt").read_text(encoding="utf-8")
    assert build(["M5", "M6"]) == expected
