"""
probe_signal_status.py — diagnostic probe for the multi-signal Status workflow.

Run BEFORE the first exit-module optimization to verify UIA structure:

  py probe_signal_status.py --dump
      Open Format Signals, dump every list item (name, control types,
      toggle state), close with Cancel/Escape.  Read-only.

  py probe_signal_status.py --toggle "SFJ_15Dworkshop_lesson4_ATRstop"
      Toggle that signal OFF then back ON (read-back verified each step).
      Watch MC64 to confirm visually.

  py probe_signal_status.py --wizard-dump
      With current statuses, launch the optimization wizard, log the
      DataGrid rows and CheckBox list (count, texts, states), then Cancel.
      Confirms grid row count / default check states / Exhaustive identity.

MC64 must be open with workspace 20260101_SFJ_Bollinger_AI.wsp (BNBUSDT HOT
HOURLY tab active).  Auto-elevates via UAC.
"""
from __future__ import annotations
import argparse
import ctypes
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pyautogui

import mc_automation as mc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s -- %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SYMBOL = "BNBUSDT HOT"
MAIN_SIGNAL = "SFJ_15Dworkshop_lesson5_countertrend_LS_crypto"


def _dump_format_signals(conn):
    dlg = mc._open_format_signals(conn)
    mc._click_tab(dlg, ["Signals", "訊號", "Signal"])
    time.sleep(0.5)

    uia = mc._uia_dlg(dlg.handle)
    if uia is None:
        log.error("UIA wrapper unavailable for Format dialog hwnd=%x", dlg.handle)
        return dlg

    for ct in ["List", "DataGrid", "Table", "ListBox"]:
        try:
            lv = uia.child_window(control_type=ct)
            if not lv.exists(timeout=2):
                continue
            log.info("=== %s found ===", ct)
            items = lv.descendants(control_type="ListItem")
            if not items:
                items = lv.descendants(control_type="DataItem")
            log.info("items: %d", len(items))
            for i, it in enumerate(items):
                try:
                    name = it.window_text()
                    txts = [d.window_text() for d in it.descendants(control_type="Text")]
                    cbs = it.descendants(control_type="CheckBox")
                    state = mc._signal_item_state(it)
                    kids = sorted({d.element_info.control_type
                                   for d in it.descendants()})
                    log.info("  [%d] name='%s' state=%s cbs=%d texts=%s kids=%s",
                             i, name, state, len(cbs), txts[:4], kids)
                    try:
                        lp = it.legacy_properties()
                        log.info("      legacy State=0x%x Role=%s",
                                 lp.get("State", 0), lp.get("Role"))
                    except Exception:
                        pass
                except Exception as e:
                    log.info("  [%d] <error: %s>", i, e)
            return dlg
        except Exception as e:
            log.debug("list type %s failed: %s", ct, e)
    log.warning("No signal list found via UIA")
    return dlg


def cmd_dump(conn):
    dlg = _dump_format_signals(conn)
    if not mc._click_button_in_dlg(dlg, ["Cancel", "取消"]):
        mc._pg_press("escape")
    log.info("--dump complete (dialog closed without changes)")


def cmd_toggle(conn, signal_name):
    log.info(">>> Toggle test: '%s' OFF", signal_name)
    mc.set_signal_statuses(conn, {signal_name: False},
                           verify=True, protected=[MAIN_SIGNAL])
    log.info(">>> OFF verified. Waiting 3s — check MC64 visually...")
    time.sleep(3)
    log.info(">>> Toggle test: '%s' back ON", signal_name)
    mc.set_signal_statuses(conn, {signal_name: True},
                           verify=True, protected=[MAIN_SIGNAL])
    log.info(">>> ON verified. Waiting 3s...")
    time.sleep(3)
    log.info(">>> Restoring '%s' to OFF (module default)", signal_name)
    mc.set_signal_statuses(conn, {signal_name: False},
                           verify=True, protected=[MAIN_SIGNAL])
    log.info("--toggle complete: all three transitions verified")


def _probe_launch_wizard(conn):
    """Minimal wizard launch: right-click chart -> Optimize Strategy."""
    mc._focus_window(conn._hwnd)
    time.sleep(0.3)
    gp_hwnd = None
    for h, cls, _ in mc._win32_enum_children(conn._hwnd):
        if cls == "ATL_MCGraphPanel" and mc._win32_is_visible(h):
            gp_hwnd = h
            break
    if gp_hwnd is None:
        raise RuntimeError("ATL_MCGraphPanel not found — is the chart visible?")
    l, t, r, b = mc._win32_get_rect(gp_hwnd)
    x, y = r - 35, (t + b) // 2
    pyautogui.click(x, y)
    time.sleep(0.2)
    pyautogui.rightClick(x, y)
    time.sleep(0.5)
    popup = mc._find_popup_hwnd(timeout=1.5)
    if not popup:
        # PostMessage fallback (panel transparent to WindowFromPoint)
        u32 = ctypes.windll.user32
        cli_x, cli_y = max(10, (r - l) - 35), max(10, (b - t) // 2)
        lp = (cli_y << 16) | (cli_x & 0xFFFF)
        u32.SetForegroundWindow(gp_hwnd)
        time.sleep(0.15)
        u32.PostMessageW(gp_hwnd, 0x0204, 0x0002, lp)
        time.sleep(0.1)
        u32.PostMessageW(gp_hwnd, 0x0205, 0, lp)
        time.sleep(0.5)
        popup = mc._find_popup_hwnd(timeout=1.5)
    if not popup:
        raise RuntimeError("No context menu after right-click")
    mc._pyautogui_click_popup_item(conn.app, [
        "Optimize Strategy", "最佳化策略", "Optimize...", "最佳化...", "優化策略",
    ])
    return mc._wait_for_any_window(
        conn.app, [r".*[Oo]ptimiz.*", r".*最佳化.*", r".*優化.*"], timeout=30)


def cmd_wizard_dump(conn):
    """Open the optimization wizard far enough to log grid/checkbox layout, then cancel."""
    wizard = _probe_launch_wizard(conn)
    time.sleep(1.5)
    uia = mc._uia_dlg(wizard.handle)
    if uia is None:
        log.error("Wizard UIA unavailable")
    else:
        rows = list(uia.descendants(control_type="DataItem"))
        log.info("=== Wizard DataGrid rows: %d ===", len(rows))
        for i, r in enumerate(rows):
            try:
                txts = [d.window_text() for d in r.descendants(control_type="Text")]
                cbs = r.descendants(control_type="CheckBox")
                st = None
                if cbs:
                    try:
                        st = cbs[0].iface_toggle.CurrentToggleState == 1
                    except Exception:
                        pass
                log.info("  row[%d] checked=%s texts=%s", i, st, txts[:5])
            except Exception as e:
                log.info("  row[%d] <error: %s>", i, e)
        cbs_all = list(uia.descendants(control_type="CheckBox"))
        log.info("=== Wizard CheckBoxes: %d ===", len(cbs_all))
        for i, cb in enumerate(cbs_all):
            try:
                st = None
                try:
                    st = cb.iface_toggle.CurrentToggleState == 1
                except Exception:
                    pass
                log.info("  CB[%d] text='%s' checked=%s rect=%s",
                         i, (cb.window_text() or "")[:30], st, cb.rectangle())
            except Exception as e:
                log.info("  CB[%d] <error: %s>", i, e)
    # Cancel out — do NOT start an optimization
    if not mc._click_button_in_dlg(wizard, ["Cancel", "取消"]):
        mc._pg_press("escape")
    time.sleep(0.5)
    log.info("--wizard-dump complete (wizard cancelled)")


def _is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _auto_elevate():
    script = str(Path(__file__).resolve())
    workdir = str(Path(__file__).resolve().parent)
    extra = [a for a in sys.argv[1:] if a != "--_elevated"]
    quoted = [f'"{a}"' if " " in a else a for a in extra]
    all_args = f'"{script}" ' + " ".join(quoted) + " --_elevated"
    print("[auto-elevate] Requesting elevation -- approve the UAC prompt.")
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, all_args, workdir, 1)
    if ret <= 32:
        print(f"[auto-elevate] ShellExecuteW failed (code={ret}).")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description="Probe MC64 signal Status UIA structure")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--toggle", metavar="SIGNAL_NAME")
    ap.add_argument("--wizard-dump", action="store_true")
    ap.add_argument("--_elevated", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not (args.dump or args.toggle or args.wizard_dump):
        ap.print_help()
        return 1

    if not _is_admin():
        _auto_elevate()
        return 0

    conn = mc.MultiChartsConnection()
    conn.connect()

    if args.dump:
        cmd_dump(conn)
    if args.toggle:
        cmd_toggle(conn, args.toggle)
    if args.wizard_dump:
        cmd_wizard_dump(conn)
    print("\nProbe done. Press Enter to close.")
    try:
        input()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
