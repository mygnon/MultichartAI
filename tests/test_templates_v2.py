"""OMS emit block v4: Z: ramdisk path, self-healing dir, own-handle write,
zero-throw retry.  (v2's FileAppend was field-disproven: MC never releases
FileAppend's handle, so the rename always hit a sharing violation.  v3's
WriteFile was rejected by the MC compiler: lplong byref = "Incorrect argument
type" -> v4 uses the pointer-free legacy _lcreat/_lwrite/_lclose.)"""
from burner import TEMPLATE_VERSION, assembler, renderer, templates
from conftest import MINIMAIN, make_src

MAIN_EL = MINIMAIN.read_text(encoding="utf-8")


def _burned():
    src = make_src([])
    sid = "MiniMain_BTC_H1_v1"
    body = assembler.assemble(src, MAIN_EL, [], sid, TEMPLATE_VERSION)
    return renderer.render(sid, body, src, "MiniMain_crypto", TEMPLATE_VERSION)


def test_template_version_is_4():
    assert TEMPLATE_VERSION == "4"
    assert "template_version    : 4" in _burned()


def test_signals_path_is_ramdisk():
    text = _burned()
    assert templates.OMS_SIGNALS_DIR == "Z:\\oms\\signals"
    assert 'oms_out = "Z:\\oms\\signals\\MiniMain_BTC_H1_v1.json"' in text
    assert "C:\\oms" not in text


def test_all_winapi_declared():
    text = _burned()
    for fn in ("MoveFileExA", "CreateDirectoryA", "GetFileAttributesA",
               "DeleteFileA", "GetTickCount", "Sleep",
               "_lcreat", "_lwrite", "_lclose"):
        assert f'"{fn}"' in text, fn
    # no byref DLL types anywhere (MC rejects them)
    assert "lplong" not in text and "lpdword" not in text


def test_no_el_file_builtins():
    flat = _burned().replace(" ", "")
    assert "FileDelete(" not in flat
    assert "FileAppend(" not in flat  # MC holds FileAppend handles forever


def test_own_handle_write_sequence():
    text = _burned()
    assert "oms_h = _lcreat( oms_tmp, 0 )" in text
    assert "oms_written = _lwrite( oms_h, oms_json, StrLen( oms_json ) )" in text
    assert "_lclose( oms_h )" in text
    assert "if oms_written = StrLen( oms_json ) then" in text
    # handle must be closed BEFORE the rename loop
    assert text.index("_lclose( oms_h )") < text.index("MoveFileExA( oms_tmp")


def test_self_healing_and_skip_guard():
    text = _burned()
    assert 'CreateDirectoryA( "Z:\\oms", 0 )' in text
    assert 'CreateDirectoryA( "Z:\\oms\\signals", 0 )' in text
    assert 'GetFileAttributesA( "Z:\\oms\\signals" ) <> -1' in text


def test_unique_tmp_and_retry_loop():
    text = _burned()
    assert "GetTickCount" in text and '".tmp"' in text
    assert "while oms_ok = 0 and oms_try < 5" in text
    assert "Sleep( 30 )" in text
    assert "DeleteFileA( oms_tmp )" in text


def test_realtime_guard_still_present():
    assert "GetAppInfo( aiRealTimeCalc ) = 1" in _burned()
