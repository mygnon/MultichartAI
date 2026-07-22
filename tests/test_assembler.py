import pytest

from burner import assembler
from burner.instruments import MODULE_REGISTRY
from burner.module_library import load_module
from conftest import MODULE_PARAMS


@pytest.mark.parametrize("label", list(MODULE_REGISTRY))
def test_transform_each_module(label):
    mod = load_module(label, MODULE_REGISTRY[label])
    tm = assembler.transform_module(mod, MODULE_PARAMS[label])
    prefix = label.lower() + "_"
    # every declared input appears prefixed in the hoisted decls with baked value
    for name, v in MODULE_PARAMS[label].items():
        assert f"{prefix}{name}(" in tm.decl_lines.replace(" ", "")
    # body carries no declaration statements
    from burner import el_lex
    assert el_lex.find_declarations(tm.body) == [] or \
        all(not d.items for d in el_lex.find_declarations(tm.body))


def test_m1_m3_orders_get_named():
    m1 = assembler.transform_module(load_module("M1", MODULE_REGISTRY["M1"]),
                                    MODULE_PARAMS["M1"])
    assert '"M1_LX"' in m1.body and '"M1_SX"' in m1.body
    m3 = assembler.transform_module(load_module("M3", MODULE_REGISTRY["M3"]),
                                    MODULE_PARAMS["M3"])
    assert '"M3_LX"' in m3.body and '"M3_SX"' in m3.body


def test_named_orders_preserved():
    m2 = assembler.transform_module(load_module("M2", MODULE_REGISTRY["M2"]),
                                    MODULE_PARAMS["M2"])
    assert '"AtrLX"' in m2.body and '"AtrSX"' in m2.body
    m6 = assembler.transform_module(load_module("M6", MODULE_REGISTRY["M6"]),
                                    MODULE_PARAMS["M6"])
    assert '"Buytocover "' in m6.body  # trailing space preserved verbatim


def test_m6_length_collision_resolved():
    m6 = assembler.transform_module(load_module("M6", MODULE_REGISTRY["M6"]),
                                    MODULE_PARAMS["M6"])
    assert "m6_Length" in m6.body and "m6_std" in m6.body
    assert "BollingerBand( c, m6_Length, -m6_std )" in m6.body


def test_unknown_param_raises():
    mod = load_module("M1", MODULE_REGISTRY["M1"])
    with pytest.raises(assembler.AssemblyError):
        assembler.transform_module(mod, {"NOPE": 1.0})
