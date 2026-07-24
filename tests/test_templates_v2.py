"""OMS emit block v2: Z: ramdisk path, self-healing dir, zero-throw retry."""
from burner import TEMPLATE_VERSION, assembler, renderer, templates
from conftest import MINIMAIN, make_src

MAIN_EL = MINIMAIN.read_text(encoding="utf-8")


def _burned():
    src = make_src([])
    sid = "MiniMain_BTC_H1_v1"
    body = assembler.assemble(src, MAIN_EL, [], sid, TEMPLATE_VERSION)
    return renderer.render(sid, body, src, "MiniMain_crypto", TEMPLATE_VERSION)


def test_template_version_is_2():
    assert TEMPLATE_VERSION == "2"
    assert "template_version    : 2" in _burned()


def test_signals_path_is_ramdisk():
    text = _burned()
    assert templates.OMS_SIGNALS_DIR == "Z:\\oms\\signals"
    assert 'oms_out = "Z:\\oms\\signals\\MiniMain_BTC_H1_v1.json"' in text
    assert "C:\\oms" not in text


def test_all_winapi_declared():
    text = _burned()
    for fn in ("MoveFileExA", "CreateDirectoryA", "GetFileAttributesA",
               "DeleteFileA", "GetTickCount", "Sleep"):
        assert f'"{fn}"' in text, fn


def test_no_throwing_filedelete():
    assert "FileDelete(" not in _burned().replace(" ", "")


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
