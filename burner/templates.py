"""Fixed text blocks (string.Template).  EL comments use { } so a brace-based
template engine would need escaping everywhere -- $-substitution avoids it.

NOTE: no timestamp inside the EL header (deliberate deviation from spec 4.2):
a burn timestamp in the code would break el_sha256 idempotency and byte-exact
golden tests.  burned_at lives in the manifest only.
"""
from __future__ import annotations

from string import Template

HEADER_TMPL = Template("""{ =====================================================================
  AUTOGEN - do not edit by hand
  strategy_id         : $strategy_id
  source_state_sha256 : $source_state_sha256
  template_version    : $template_version
  main                : $main_source  (params baked from stage2.main_champ)
  exit_modules        : $modules_line  (params baked from stage3)
===================================================================== }
""")

MODULE_SECTION_TMPL = Template(
    "{ ==== $label: $signal (burned: $params_line) ==== }\n")

# ---- OMS signal emit (oms-spec.md section 2.1) -----------------------------
# Declarations are hoisted next to the main declarations (PowerLanguage wants
# declarations before executable statements); the body sits at file end.
# Real-time-only guard: backtests / the equivalence gate never write files.
# Atomic publish: write .tmp, then kernel32.MoveFileExA replace-existing.
# COMPILE-CHECK ITEMS (verify once in MC64 Study Editor):
#   GetAppInfo(aiRealTimeCalc), DoubleQuote, FormatDate/FormatTime,
#   ELDateToDateTime/ELTimeToDateTime, ComputerDateTime, DefineDLLFunc.

OMS_DECLS_TMPL = Template("""{ ==== OMS emit declarations (template v$template_version) ==== }
DefineDLLFunc: "kernel32.dll", int, "MoveFileExA", lpstr, lpstr, int ;
variables:
    oms_q( "" ),
    oms_out( "" ),
    oms_tmp( "" ),
    oms_json( "" ) ;
""")

OMS_BODY_TMPL = Template("""{ ==== OMS signal emit: real-time only, atomic tmp+rename (oms-spec 2.1) ==== }
if GetAppInfo( aiRealTimeCalc ) = 1 then
begin
    oms_q = DoubleQuote ;
    oms_out = "C:\\oms\\signals\\$strategy_id.json" ;
    oms_tmp = oms_out + ".tmp" ;
    oms_json =
        "{" + NewLine +
        "  " + oms_q + "strategy_id" + oms_q + ": " + oms_q + "$strategy_id" + oms_q + "," + NewLine +
        "  " + oms_q + "symbol" + oms_q + ": " + oms_q + "$symbol" + oms_q + "," + NewLine +
        "  " + oms_q + "target_units" + oms_q + ": " + NumToStr( CurrentContracts * MarketPosition, 0 ) + "," + NewLine +
        "  " + oms_q + "theoretical_equity" + oms_q + ": " + NumToStr( NetProfit + OpenPositionProfit, 2 ) + "," + NewLine +
        "  " + oms_q + "bar_time" + oms_q + ": " + oms_q + FormatDate( "yyyy-MM-dd", ELDateToDateTime( Date ) ) + " " + FormatTime( "HH:mm:ss", ELTimeToDateTime( Time ) ) + oms_q + "," + NewLine +
        "  " + oms_q + "emit_time" + oms_q + ": " + oms_q + FormatDate( "yyyy-MM-dd", ComputerDateTime ) + " " + FormatTime( "HH:mm:ss", ComputerDateTime ) + oms_q + NewLine +
        "}" + NewLine ;
    FileDelete( oms_tmp ) ;
    FileAppend( oms_tmp, oms_json ) ;
    MoveFileExA( oms_tmp, oms_out, 1 ) ;
end ;
""")
