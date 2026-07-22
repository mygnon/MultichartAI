from burner import TEMPLATE_VERSION, assembler, renderer, static_validator
from conftest import MINIMAIN, make_src

MAIN_EL = MINIMAIN.read_text(encoding="utf-8")


def _clean():
    src = make_src([])
    sid = "MiniMain_BTC_H1_v1"
    body = assembler.assemble(src, MAIN_EL, [], sid, TEMPLATE_VERSION)
    text = renderer.render(sid, body, src, "MiniMain_crypto", TEMPLATE_VERSION)
    manifest = {"strategy_id": sid, "params": dict(src.main_params),
                "exit_modules": []}
    return text, manifest


def test_clean_passes():
    text, manifest = _clean()
    assert static_validator.validate(text, manifest, "MiniMain") == []


def test_unbalanced_paren_detected():
    text, manifest = _clean()
    assert any("unbalanced" in e for e in
               static_validator.validate(text + "x = ( 1 ;\n", manifest, "MiniMain"))


def test_stray_brace_detected():
    text, manifest = _clean()
    assert any("stray" in e for e in
               static_validator.validate(text + "} \n", manifest, "MiniMain"))


def test_roundtrip_mismatch_detected():
    text, manifest = _clean()
    manifest = dict(manifest, params=dict(manifest["params"], Length=999.0))
    assert any("round-trip" in e for e in
               static_validator.validate(text, manifest, "MiniMain"))


def test_roundtrip_missing_module_input_detected():
    text, manifest = _clean()
    manifest = dict(manifest, exit_modules=[
        {"id": "M5", "signal": "QuantPass_PT_Exit", "params": {"PT_Base": 0.2}}])
    assert any("m5_PT_Base" in e for e in
               static_validator.validate(text, manifest, "MiniMain"))


def test_duplicate_input_detected():
    text, manifest = _clean()
    bad = text + "inputs: LENGTH( 1 ) ;\n"
    assert any("duplicate input" in e for e in
               static_validator.validate(bad, manifest, "MiniMain"))


def test_duplicate_order_name_detected():
    text, manifest = _clean()
    bad = text + 'Sell ( "MmLX" ) next bar at 1 stop ;\n'
    assert any("duplicate order name" in e for e in
               static_validator.validate(bad, manifest, "MiniMain"))


def test_undeclared_external_func_detected():
    text, manifest = _clean()
    bad = text + "x = _Crypto1MUSD ;\n"
    assert any("external PL function" in e for e in
               static_validator.validate(bad, manifest, "MiniMain"))


def test_bad_strategy_id_detected():
    text, manifest = _clean()
    manifest = dict(manifest, strategy_id="MiniMain_XXX_H1_v1")
    assert any("naming rule" in e for e in
               static_validator.validate(text, manifest, "MiniMain"))
