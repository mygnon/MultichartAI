from burner import el_lex


def test_comments_and_strings_are_opaque():
    src = '{ Length inside comment } // Length line\nx = "Length" ;\nLength = 1 ;\n'
    out = el_lex.rename_idents(src, {"Length": "m6_Length"})
    assert "{ Length inside comment }" in out
    assert "// Length line" in out
    assert '"Length"' in out
    assert "m6_Length = 1 ;" in out


def test_rename_is_case_insensitive_whole_token():
    src = "posh = 0 ; POSH = High ; poshx = 1 ;\n"
    out = el_lex.rename_idents(src, {"POSH": "m2_POSH"})
    assert out == "m2_POSH = 0 ; m2_POSH = High ; poshx = 1 ;\n"


def test_series_reference_survives():
    out = el_lex.rename_idents("if MP<>MP[1] then x = 1 ;", {"MP": "m2_MP"})
    assert out == "if m2_MP<>m2_MP[1] then x = 1 ;"


def test_builtin_argument_rename_m6_case():
    src = "if c crosses under BollingerBand( c, Length, -std ) then x = 1 ;"
    out = el_lex.rename_idents(src, {"Length": "m6_Length", "std": "m6_std"})
    assert "BollingerBand( c, m6_Length, -m6_std )" in out
    assert " c," in out  # builtin alias c untouched


def test_builtin_L_untouched():
    src = "IF LOW < POSL THEN POSL = L ;"
    out = el_lex.rename_idents(src, {"POSL": "m2_POSL"})
    assert out == "IF LOW < m2_POSL THEN m2_POSL = L ;"


def test_parse_declarations_multi_and_case():
    src = ("INPUT:STP(1);\nVAR:ATR(0),MP(0),POSL(99999);\n"
           "inputs:  Length( 25 ), std( 3.06 );\n")
    inputs, vars_ = el_lex.parse_declarations(src)
    assert inputs == {"STP": "1", "Length": "25", "std": "3.06"}
    assert vars_ == {"ATR": "0", "MP": "0", "POSL": "99999"}


def test_rewrite_input_defaults_surgical():
    src = "inputs:\n    Length( 20 ),\n    BandMult( 0.0 ) ;\nx = Length ;\n"
    out = el_lex.rewrite_input_defaults(src, {"length": "8", "BANDMULT": "4.75"})
    assert "Length( 8 )" in out or "Length(8)" in out
    assert "BandMult( 4.75 )" in out or "BandMult(4.75)" in out
    assert "x = Length ;" in out  # body untouched


def test_rewrite_missing_input_raises():
    import pytest
    with pytest.raises(KeyError):
        el_lex.rewrite_input_defaults("inputs: A(1) ;", {"B": "2"})


def test_name_unnamed_orders():
    src = ("IF marketposition=1  THEN SELL NEXT BAR MARKET;\n"
           "IF marketposition=-1 THEN BUYTOCOVER NEXT BAR MARKET;\n")
    out = el_lex.name_unnamed_orders(src, "M3")
    assert 'SELL ( "M3_LX" ) NEXT BAR MARKET;' in out
    assert 'BUYTOCOVER ( "M3_SX" ) NEXT BAR MARKET;' in out


def test_named_orders_untouched():
    src = 'SELL ( "AtrLX" ) next bar at 1 stop ;\nBuytocover ( "Buytocover " ) next bar at market;\n'
    assert el_lex.name_unnamed_orders(src, "M2") == src


def test_find_order_names():
    src = ('sell ("LX_PT") next bar market; buytocover ("SX_PT") next bar market;\n'
           "SELL NEXT BAR 1 STOP;\n")
    assert el_lex.find_order_names(src) == ["LX_PT", "SX_PT"]


def test_first_executable_offset_after_decls_and_comments():
    src = "{ header }\ninputs: A( 1 ) ;\nvariables: B( 0 ) ;\nB = A ;\n"
    off = el_lex.first_executable_offset(src)
    assert src[off:].startswith("B = A ;")


def test_format_number():
    assert el_lex.format_number(8.0) == "8"
    assert el_lex.format_number(4.75) == "4.75"
    assert el_lex.format_number(0.1 + 0.2) == "0.3"
    assert el_lex.format_number(140.0) == "140"
