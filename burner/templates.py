"""Fixed text blocks (string.Template).  EL comments use { } so a brace-based
template engine would need escaping everywhere -- $-substitution avoids it.

NOTE: no timestamp inside the EL header (deliberate deviation from spec 4.2):
a burn timestamp in the code would break el_sha256 idempotency and byte-exact
golden tests.  burned_at lives in the manifest only.
"""
from __future__ import annotations

from string import Template

# Single definition point for the signal-file directory (oms-spec.md 2.1/9.2).
# Z: is a ramdisk: microsecond writes, no SSD wear, contents vanish on reboot —
# the emit block self-heals the directory, so no boot script is needed.
OMS_ROOT = "Z:\\oms"
OMS_SIGNALS_DIR = "Z:\\oms\\signals"

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

# ---- OMS signal emit v2 (oms-spec.md section 2.1) --------------------------
# Declarations are hoisted next to the main declarations (PowerLanguage wants
# declarations before executable statements); the body sits at file end.
# Real-time-only guard: backtests / the equivalence gate never write files.
#
# v2 design: ZERO-THROW.  EL has no try/catch and a file-builtin runtime error
# DISARMS Auto Order Execution, so every failable step is a WinAPI call with a
# checkable return value:
#   - self-healing directory (CreateDirectoryA idempotent; if the ramdisk is
#     not mounted, GetFileAttributesA says so and the emit is SKIPPED)
#   - unique .tmp name per write (GetTickCount + sequence) -> no name contention
#   - MoveFileExA replace-existing rename retried 5x with Sleep(30ms); on
#     final failure DeleteFileA cleans the tmp and the emit is skipped (next
#     bar re-emits; OMS treats a stale emit_time via its heartbeat rules)
# FileAppend remains the only throwing builtin, guarded by the existence check
# and a fresh unique filename.
# COMPILE-CHECK ITEMS (verified v1: GetAppInfo/DoubleQuote/FormatDate/
# ELDateToDateTime/ComputerDateTime/DefineDLLFunc+MoveFileExA; new in v2:
# CreateDirectoryA, GetFileAttributesA, DeleteFileA, GetTickCount, Sleep).

OMS_DECLS_TMPL = Template("""{ ==== OMS emit declarations (template v$template_version) ==== }
DefineDLLFunc: "kernel32.dll", int, "MoveFileExA", lpstr, lpstr, int ;
DefineDLLFunc: "kernel32.dll", int, "CreateDirectoryA", lpstr, long ;
DefineDLLFunc: "kernel32.dll", long, "GetFileAttributesA", lpstr ;
DefineDLLFunc: "kernel32.dll", int, "DeleteFileA", lpstr ;
DefineDLLFunc: "kernel32.dll", long, "GetTickCount" ;
DefineDLLFunc: "kernel32.dll", void, "Sleep", long ;
variables:
    oms_q( "" ),
    oms_out( "" ),
    oms_tmp( "" ),
    oms_json( "" ),
    oms_seq( 0 ),
    oms_try( 0 ),
    oms_ok( 0 ) ;
""")

OMS_BODY_TMPL = Template("""{ ==== OMS signal emit v2: ramdisk, self-healing dir, zero-throw retry (oms-spec 2.1) ==== }
if GetAppInfo( aiRealTimeCalc ) = 1 then
begin
    { self-healing directory: idempotent, survives ramdisk reboot wipe.
      If the ramdisk is not mounted the emit is SKIPPED -- nothing throws. }
    CreateDirectoryA( "$oms_root", 0 ) ;
    CreateDirectoryA( "$signals_dir", 0 ) ;
    if GetFileAttributesA( "$signals_dir" ) <> -1 then
    begin
        oms_q = DoubleQuote ;
        oms_out = "$signals_dir\\$strategy_id.json" ;
        oms_seq = oms_seq + 1 ;
        oms_tmp = oms_out + "." + NumToStr( GetTickCount, 0 ) + "." + NumToStr( oms_seq, 0 ) + ".tmp" ;
        oms_json =
            "{" + NewLine +
            "  " + oms_q + "strategy_id" + oms_q + ": " + oms_q + "$strategy_id" + oms_q + "," + NewLine +
            "  " + oms_q + "symbol" + oms_q + ": " + oms_q + "$symbol" + oms_q + "," + NewLine +
            "  " + oms_q + "target_units" + oms_q + ": " + NumToStr( CurrentContracts * MarketPosition, 0 ) + "," + NewLine +
            "  " + oms_q + "theoretical_equity" + oms_q + ": " + NumToStr( NetProfit + OpenPositionProfit, 2 ) + "," + NewLine +
            "  " + oms_q + "bar_time" + oms_q + ": " + oms_q + FormatDate( "yyyy-MM-dd", ELDateToDateTime( Date ) ) + " " + FormatTime( "HH:mm:ss", ELTimeToDateTime( Time ) ) + oms_q + "," + NewLine +
            "  " + oms_q + "emit_time" + oms_q + ": " + oms_q + FormatDate( "yyyy-MM-dd", ComputerDateTime ) + " " + FormatTime( "HH:mm:ss", ComputerDateTime ) + oms_q + NewLine +
            "}" + NewLine ;
        FileAppend( oms_tmp, oms_json ) ;
        oms_ok = 0 ;
        oms_try = 0 ;
        while oms_ok = 0 and oms_try < 5
        begin
            oms_ok = MoveFileExA( oms_tmp, oms_out, 1 ) ;
            oms_try = oms_try + 1 ;
            if oms_ok = 0 and oms_try < 5 then
                Sleep( 30 ) ;
        end ;
        if oms_ok = 0 then
            DeleteFileA( oms_tmp ) ;
    end ;
end ;
""")


def render_oms_decls(template_version: str) -> str:
    return OMS_DECLS_TMPL.substitute(template_version=template_version)


def render_oms_body(strategy_id: str, symbol: str) -> str:
    return OMS_BODY_TMPL.substitute(strategy_id=strategy_id, symbol=symbol,
                                    oms_root=OMS_ROOT,
                                    signals_dir=OMS_SIGNALS_DIR)
