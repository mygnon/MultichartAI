"""OMS emit block v3: Z: ramdisk path, self-healing dir, own-handle write,
zero-throw retry.  (v2's FileAppend was field-disproven: MC never releases
FileAppend's handle, so the rename always hit a sharing violation.)"""
from burner import TEMPLATE_VERSION, assembler, renderer, templates
from conftest import MINIMAIN, make_src

MAIN_EL = MINIMAIN.read_text(encoding="utf-8")


def _burned():
    src = make_src([])
    sid = "MiniMain_BTC_H1_v1"
    body = assembler.assemble(src, MAIN_EL, [], sid, TEMPLATE_VERSION)
    return renderer.render(sid, body, src, "MiniMain_crypto", TEMPLATE_VERSION)


def test_template_version_is_3():
    assert TEMPLATE_VERSION == "3"
    assert "template_version    : 3" in _burned()


def test_signals_path_is_ramdisk():
    text = _burned()
    assert templates.OMS_SIGNALS_DIR == "Z:\\oms\\signals"
    assert 'oms_out = "Z:\\oms\\signals\\MiniMain_BTC_H1_v1.json"' in text
    assert "C:\\oms" not in text


def test_all_winapi_declared():
    text = _burned()
    for fn in ("MoveFileExA", "CreateDirectoryA", "GetFileAttributesA",
               "DeleteFileA", "GetTickCount", "Sleep",
               "CreateFileA", "WriteFile", "CloseHandle"):
        assert f'"{fn}"' in text, fn


def test_no_el_file_builtins():
    flat = _burned().replace(" ", "")
    assert "FileDelete(" not in flat
    assert "FileAppend(" not in flat  # MC holds FileAppend handles forever


def test_own_handle_write_sequence():
    text = _burned()
    assert "CreateFileA( oms_tmp, 1073741824, 0, 0, 2, 128, 0 )" in text
    assert "WriteFile( oms_h, oms_json, StrLen( oms_json ), oms_written, 0 )" in text
    assert "CloseHandle( oms_h )" in text
    # CloseHandle must come BEFORE the rename loop
    assert text.index("CloseHandle( oms_h )") < text.index("MoveFileExA( oms_tmp")


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
