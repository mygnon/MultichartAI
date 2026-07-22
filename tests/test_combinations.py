"""All 2^6 kept subsets (plus shuffled keep orders) must assemble into
validator-clean EL with stable module order == keep order."""
import itertools
import random

from burner import TEMPLATE_VERSION, assembler, renderer, static_validator
from burner.instruments import MODULE_REGISTRY
from burner.module_library import load_module
from conftest import MINIMAIN, MODULE_PARAMS, make_src

ALL = list(MODULE_REGISTRY)
MAIN_EL = MINIMAIN.read_text(encoding="utf-8")
MODS = {lb: load_module(lb, MODULE_REGISTRY[lb]) for lb in ALL}


def _burn(kept_order):
    src = make_src(kept_order, main_params={"Length": 8.0, "BandMult": 4.75,
                                            "ATRMult": 5.5, "ReentryBars": 0.0})
    sid = "MiniMain_BTC_H1_v1"
    body = assembler.assemble(src, MAIN_EL, [MODS[lb] for lb in kept_order],
                              sid, TEMPLATE_VERSION)
    text = renderer.render(sid, body, src, "MiniMain_crypto", TEMPLATE_VERSION)
    manifest = {"strategy_id": sid,
                "params": dict(src.main_params),
                "exit_modules": [{"id": m.label, "signal": m.signal,
                                  "params": dict(m.params)} for m in src.kept]}
    return text, manifest


def test_all_subsets_validate_clean():
    rng = random.Random(42)
    for r in range(len(ALL) + 1):
        for combo in itertools.combinations(ALL, r):
            order = list(combo)
            rng.shuffle(order)
            text, manifest = _burn(order)
            errors = static_validator.validate(text, manifest, "MiniMain")
            assert errors == [], f"kept={order}: {errors}"
            # module sections appear in keep order
            positions = [text.index(f"==== {lb}:") for lb in order]
            assert positions == sorted(positions)


def test_output_is_deterministic():
    a, _ = _burn(["M5", "M1", "M2"])
    b, _ = _burn(["M5", "M1", "M2"])
    assert a == b
