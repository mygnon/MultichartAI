"""
MultiCharts64 UI Automation
- Window discovery: pywinauto (read-only, no UAC restriction)
- All keyboard/mouse input: pyautogui (works across privilege levels)
"""
from __future__ import annotations
import ctypes
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyautogui
import pyperclip
import pywinauto
from pywinauto import Application, Desktop
from pywinauto.keyboard import send_keys

from config import (
    DateRange, StrategyConfig, MC_PROCESS_NAME,
    OPTIMIZATION_TIMEOUT_SECONDS, POLL_INTERVAL_SECONDS,
    MC_COLUMN_MAP, PLATEAU_MIN_TRADES,
)

logger = logging.getLogger(__name__)

# Skip redundant date-range setting when the range hasn't changed between attempts.
# Saves ~37s per attempt after the first one.
_configure_date_range_cache: Optional[Tuple[str, str]] = None

pyautogui.PAUSE = 0.05
pyautogui.FAILSAFE = False

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class MCNotRunningError(RuntimeError):
    pass

class MCUIError(RuntimeError):
    pass

class WindowNotFoundError(MCUIError):
    pass

class OptimizationFailedError(MCUIError):
    pass


# ---------------------------------------------------------------------------
# Admin rights check
# ---------------------------------------------------------------------------

def check_admin_rights() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Win32 ctypes helpers — bypass pywinauto's process-privilege filtering
#
# UIPI root cause: pywinauto creates WindowSpecifications bound to a specific
# process PID (e.g. process=14420 for admin MC).  When it resolves a spec it
# calls OpenProcess(PROCESS_QUERY_INFORMATION) on that PID — which Windows
# BLOCKS for lower-integrity callers.  Result: ElementNotFoundError even when
# the window is plainly visible on screen.
#
# Fix strategy:
#   1. Never use process-scoped specs for MC dialogs.
#   2. Discover dialog hwnds via ctypes EnumWindows (no process needed).
#   3. Wrap them in _HwndWrapper so helpers can read .handle without
#      pywinauto's OpenProcess path.
#   4. Use ctypes EnumChildWindows + UIAutomation for children.
# ---------------------------------------------------------------------------


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

_EnumChildProcType = ctypes.WINFUNCTYPE(
    ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p
)


def _win32_enum_children(parent_hwnd: int) -> List[Tuple[int, str, str]]:
    """Return [(hwnd, class_name, window_text)] for every child window."""
    results: List[Tuple[int, str, str]] = []

    def _cb(hwnd, _):
        cls_buf = ctypes.create_unicode_buffer(256)
        _user32.GetClassNameW(hwnd, cls_buf, 256)
        n = _user32.GetWindowTextLengthW(hwnd)
        txt_buf = ctypes.create_unicode_buffer(n + 1)
        if n > 0:
            _user32.GetWindowTextW(hwnd, txt_buf, n + 1)
        results.append((hwnd, cls_buf.value, txt_buf.value))
        return True

    cb = _EnumChildProcType(_cb)
    _user32.EnumChildWindows(parent_hwnd, cb, None)
    return results


_user32 = ctypes.windll.user32
_user32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(_RECT)]
_user32.GetWindowRect.restype = ctypes.c_bool
_user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
_user32.IsWindowVisible.restype = ctypes.c_bool
_user32.GetClassNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
_user32.GetClassNameW.restype = ctypes.c_int
_user32.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
_user32.GetWindowTextLengthW.restype = ctypes.c_int
_user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
_user32.GetWindowTextW.restype = ctypes.c_int
_user32.EnumChildWindows.argtypes = [ctypes.c_void_p, _EnumChildProcType, ctypes.c_void_p]
_user32.EnumChildWindows.restype = ctypes.c_bool
_user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
_user32.FindWindowW.restype = ctypes.c_void_p
_user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
_user32.SetForegroundWindow.restype = ctypes.c_bool
# Menu API — return types must be c_void_p (64-bit handle) to avoid truncation
_user32.GetMenu.argtypes = [ctypes.c_void_p]
_user32.GetMenu.restype = ctypes.c_void_p
_user32.GetMenuItemCount.argtypes = [ctypes.c_void_p]
_user32.GetMenuItemCount.restype = ctypes.c_int
_user32.GetMenuStringW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_int, ctypes.c_uint]
_user32.GetMenuStringW.restype = ctypes.c_int
_user32.GetMenuItemRect.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(_RECT)]
_user32.GetMenuItemRect.restype = ctypes.c_bool
_user32.WindowFromPoint.restype = ctypes.c_void_p
_user32.IsChild.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_user32.IsChild.restype = ctypes.c_bool
_user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
_user32.SendMessageW.restype = ctypes.c_long


def _win32_get_rect(hwnd: int) -> Tuple[int, int, int, int]:
    r = _RECT()
    ok = _user32.GetWindowRect(hwnd, ctypes.byref(r))
    if not ok:
        logger.debug("GetWindowRect(hwnd=%x) returned 0 (err=%d)",
                     hwnd, ctypes.get_last_error())
    return r.left, r.top, r.right, r.bottom


class _HwndWrapper:
    """Minimal stand-in for pywinauto WindowSpecification that holds an hwnd.

    All our helpers accept this because they check `dlg.handle`.
    Avoids pywinauto's process-scoped OpenProcess path.
    """
    def __init__(self, hwnd: int):
        self.handle = hwnd

    def rectangle(self):
        l, t, r, b = _win32_get_rect(self.handle)
        class _R:
            pass
        obj = _R()
        obj.left, obj.top, obj.right, obj.bottom = l, t, r, b
        return obj


def _win32_is_visible(hwnd: int) -> bool:
    return bool(_user32.IsWindowVisible(hwnd))


def _win32_click_hwnd(hwnd: int, offset_y: int = 0) -> None:
    l, t, r, b = _win32_get_rect(hwnd)
    pyautogui.click((l + r) // 2, (t + b) // 2 + offset_y)
    time.sleep(0.2)


def _win32_find_child(
    parent_hwnd: int,
    class_name: str = None,
    title_contains: str = None,
    visible_only: bool = True,
) -> Optional[int]:
    for hwnd, cls, txt in _win32_enum_children(parent_hwnd):
        if visible_only and not _win32_is_visible(hwnd):
            continue
        if class_name and cls.lower() != class_name.lower():
            continue
        if title_contains and title_contains.lower() not in txt.lower():
            continue
        return hwnd
    return None


def _uia_dlg(hwnd: int):
    """Return a UIAutomation (uia backend) wrapper for *hwnd*.

    UIAutomation communicates via in-process COM providers, so it works
    across UIPI privilege boundaries — unlike win32 backend OpenProcess.
    Returns None if uia is unavailable.
    """
    try:
        from pywinauto import Desktop as _UIA_Desktop  # local import avoids circular dep
        return _UIA_Desktop(backend="uia").window(handle=hwnd)
    except Exception as e:
        logger.debug("_uia_dlg hwnd=%x failed: %s", hwnd, e)
        return None


# ---------------------------------------------------------------------------
# Window discovery helpers (read-only — no UAC restriction)
# ---------------------------------------------------------------------------

def _window_area(win) -> int:
    try:
        r = win.rectangle()
        return (r.right - r.left) * (r.bottom - r.top)
    except Exception:
        return 0


def _find_all_mc_windows() -> List[Tuple[int, str, int]]:
    """Return list of (hwnd, title, area) for all visible MultiCharts windows."""
    results = []
    try:
        for win in Desktop(backend="win32").windows():
            try:
                title = win.window_text()
                if "MultiCharts" in title and win.is_visible():
                    results.append((win.handle, title, _window_area(win)))
            except Exception:
                pass
    except Exception as e:
        logger.warning("Desktop enumeration error: %s", e)
    return results


def _find_hwnd_for_workspace(workspace_path: str) -> Optional[int]:
    """Find the MC window whose title contains the workspace file stem."""
    stem = Path(workspace_path).stem
    for hwnd, title, area in _find_all_mc_windows():
        if stem in title:
            logger.debug("Workspace match: '%s' -> hwnd=0x%x", title, hwnd)
            return hwnd
    return None


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

class MultiChartsConnection:
    def __init__(self):
        self.app: Optional[Application] = None
        self._hwnd: Optional[int] = None

    def connect(self) -> None:
        if not check_admin_rights():
            logger.warning(
                "Script is NOT running as Administrator. "
                "UI automation may fail. Please relaunch as Administrator."
            )
        windows = _find_all_mc_windows()
        if not windows:
            raise MCNotRunningError("MultiCharts64 not found. Please start it first.")
        windows.sort(key=lambda x: x[2], reverse=True)
        logger.info("Open MultiCharts windows (%d):", len(windows))
        for hwnd, title, _ in windows:
            logger.info("  [0x%x] %s", hwnd, title)
        hwnd, title, _ = windows[0]
        self._connect_to_hwnd(hwnd)
        logger.info("Connected to MC: '%s'", title)

    def _connect_to_hwnd(self, hwnd: int) -> None:
        self.app = Application(backend="win32").connect(handle=hwnd, timeout=10)
        self._hwnd = hwnd

    def switch_to_workspace(self, workspace_path: str) -> bool:
        """Switch active connection to the MC window for this workspace."""
        hwnd = _find_hwnd_for_workspace(workspace_path)
        if hwnd is None:
            logger.warning("Workspace window not found for: %s", Path(workspace_path).stem)
            return False
        if hwnd != self._hwnd:
            self._connect_to_hwnd(hwnd)
            logger.info("Switched to workspace: %s (hwnd=0x%x)", Path(workspace_path).stem, hwnd)
        return True

    def main_window(self):
        wins = [w for w in self.app.windows(visible_only=True) if _window_area(w) > 0]
        if not wins:
            wins = self.app.windows(visible_only=False)
        return max(wins, key=_window_area)

    def is_alive(self) -> bool:
        try:
            return self.app.is_process_running()
        except Exception:
            return False


# ---------------------------------------------------------------------------
# pyautogui helpers — all keyboard/mouse input goes through these
# ---------------------------------------------------------------------------

def _focus_window(hwnd: int) -> None:
    """
    Bring the MC window to foreground using pyautogui (SendInput — crosses UIPI).

    SetForegroundWindow/WM_* messages are UIPI-blocked from non-admin to admin.
    Strategy: use WindowFromPoint to find an unobstructed pixel inside MC (other
    windows such as VS Code may cover parts of MC), then click it with pyautogui
    (SendInput crosses UIPI).  Scan from left edge inward since VS Code typically
    covers the right portion of the MC window.
    """
    import win32gui, win32con

    # Quick bail if already foreground
    try:
        if win32gui.GetForegroundWindow() == hwnd:
            return
    except Exception:
        pass

    try:
        rect = win32gui.GetWindowRect(hwnd)
    except Exception:
        rect = _win32_get_rect(hwnd)
        rect = (rect[0], rect[1], rect[2], rect[3])

    win_w = rect[2] - rect[0]
    win_h = rect[3] - rect[1]

    # Restore if minimised
    try:
        placement = win32gui.GetWindowPlacement(hwnd)
        is_minimised = (placement[1] == win32con.SW_SHOWMINIMIZED)
    except Exception:
        is_minimised = (win_h < 50 or win_w < 50)

    # ── Step 1: Taskbar button click (Shell_TrayWnd is medium-integrity) ────────
    # Clicking the Shell taskbar button activates the window cross-privilege because
    # the Shell (not our process) issues the SetForegroundWindow call.
    _click_taskbar_button_for_hwnd(hwnd)
    time.sleep(0.4)
    try:
        rect = win32gui.GetWindowRect(hwnd)
        win_w = rect[2] - rect[0]
        win_h = rect[3] - rect[1]
    except Exception:
        pass
    try:
        if win32gui.GetForegroundWindow() == hwnd:
            logger.info("_focus_window: MC focused via taskbar button click")
            return
    except Exception:
        return

    if is_minimised:
        logger.debug("_focus_window: taskbar click attempted restore but window still minimised")

    # ── Step 2: UIAutomation SetFocus (COM, routes through MC's own provider)  ──
    # IUIAutomation::SetFocus() runs in the high-integrity MC process via COM,
    # so it isn't blocked by UIPI even though our script is medium-integrity.
    try:
        from pywinauto import Desktop as _UIA_D
        _uia_wnd = _UIA_D(backend="uia").window(handle=hwnd)
        _uia_wnd.set_focus()
        time.sleep(0.4)
        if win32gui.GetForegroundWindow() == hwnd:
            logger.info("_focus_window: MC focused via UIAutomation SetFocus")
            return
        logger.debug("_focus_window: UIA SetFocus did not change foreground")
    except Exception as _e:
        logger.debug("_focus_window: UIA SetFocus failed: %s", _e)

    # ── Fallback: scan for an unobstructed pixel and click via SendInput ──────
    # Prefer left edge (x_frac small) since VS Code overlaps MC's right half.
    for x_frac in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]:
        for y_frac in [0.25, 0.10, 0.40, 0.50]:
            cx = rect[0] + max(5, int(win_w * x_frac))
            cy = rect[1] + max(10, int(win_h * y_frac))
            # Check whether this screen pixel actually belongs to MC (not covered)
            try:
                top = win32gui.WindowFromPoint((cx, cy))
                if top != hwnd and not win32gui.IsChild(hwnd, top):
                    continue  # another window is in front here
            except Exception:
                pass  # can't verify — try clicking anyway
            try:
                logger.info("_focus_window: clicking MC at (%d,%d) x_frac=%.2f y_frac=%.2f",
                            cx, cy, x_frac, y_frac)
                pyautogui.click(cx, cy)
                time.sleep(0.3)
            except Exception as e:
                logger.debug("_focus_window click failed: %s", e)
                continue
            try:
                if win32gui.GetForegroundWindow() == hwnd:
                    logger.info("_focus_window: MC focused successfully at (%d,%d)", cx, cy)
                    return
            except Exception:
                return  # can't verify, assume OK
            logger.debug("_focus_window: click at (%d,%d) did not gain focus, trying next", cx, cy)
            break  # point was unobstructed but click didn't focus — try next x_frac

    logger.warning(
        "_focus_window: MC hwnd=0x%x still not foreground after UIA SetFocus + click scan. "
        "Run this script as Administrator to fix cross-privilege focus.", hwnd)


def _click_taskbar_button_for_hwnd(hwnd: int) -> None:
    """Click the taskbar button for *hwnd* to restore/activate it.

    UIAutomation is used because it works cross-privilege (unlike SendMessage).
    Taskbar buttons are virtual list items — not real child hwnds — so ctypes
    EnumChildWindows won't find them; UIAutomation exposes them properly.
    """
    import win32gui
    try:
        target_title = win32gui.GetWindowText(hwnd)
    except Exception:
        target_title = ""

    if not target_title:
        logger.debug("_click_taskbar_button: no title for hwnd=%x", hwnd)
        return

    logger.debug("Restoring hwnd=%x via taskbar: looking for '%s'", hwnd, target_title[:30])

    # Primary: UIAutomation (cross-privilege)
    try:
        from pywinauto import Desktop as _UIA_D
        taskbar_uia = _UIA_D(backend="uia").window(class_name="Shell_TrayWnd")
        for btn in taskbar_uia.descendants(control_type="Button"):
            try:
                txt = btn.window_text()
                if target_title[:20].lower() in txt.lower():
                    btn.click_input()
                    logger.debug("UIA taskbar click: '%s'", txt)
                    return
            except Exception:
                pass
        # Fallback: any button with "MultiCharts" in the name
        for btn in taskbar_uia.descendants(control_type="Button"):
            try:
                txt = btn.window_text()
                if "multicharts" in txt.lower():
                    btn.click_input()
                    logger.debug("UIA taskbar click (fallback): '%s'", txt)
                    return
            except Exception:
                pass
    except Exception as e:
        logger.debug("UIA taskbar search failed: %s", e)

    # Fallback: Win32 EnumChildWindows on Shell_TrayWnd
    try:
        taskbar_hwnd = _user32.FindWindowW("Shell_TrayWnd", None)
        if taskbar_hwnd:
            for child_h, cls, txt in _win32_enum_children(taskbar_hwnd):
                if (target_title[:15].lower() in txt.lower() or
                        "multicharts" in txt.lower()) and _win32_is_visible(child_h):
                    _win32_click_hwnd(child_h)
                    logger.debug("Win32 taskbar click: '%s'", txt)
                    return
    except Exception as e:
        logger.debug("Win32 taskbar search failed: %s", e)

    # Last resort: try ShowWindow via ctypes (may or may not work across UIPI)
    try:
        SW_RESTORE = 9
        ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)
        time.sleep(0.5)
        _user32.SetForegroundWindow(hwnd)
        logger.debug("Attempted ShowWindow SW_RESTORE for hwnd=%x", hwnd)
    except Exception as e:
        logger.debug("ShowWindow fallback failed: %s", e)


def _pg_hotkey(*keys) -> None:
    pyautogui.hotkey(*keys)
    time.sleep(0.3)


def _ensure_english_ime() -> None:
    """Force the active window's IME into English/alphanumeric mode.

    Chinese IMEs (e.g. Microsoft New Phonetic / Zhuyin) intercept numeric key
    presses when in Chinese-conversion mode, breaking parameter value entry.
    This function attaches to the foreground thread's input, reads the current
    IME conversion status, and if non-zero (CJK mode) resets it to 0
    (IME_CMODE_ALPHANUMERIC = English half-width). As a belt-and-suspenders
    fallback it also fires a Shift keypress, which toggles most CJK IMEs from
    Chinese→English mode.
    """
    try:
        user32   = ctypes.windll.user32
        imm32    = ctypes.windll.imm32
        kernel32 = ctypes.windll.kernel32
        hwnd     = user32.GetForegroundWindow()
        if not hwnd:
            return

        our_tid    = kernel32.GetCurrentThreadId()
        target_tid = user32.GetWindowThreadProcessId(hwnd, None)
        attached   = False
        if target_tid and target_tid != our_tid:
            attached = bool(user32.AttachThreadInput(our_tid, target_tid, True))
        try:
            focused = user32.GetFocus() or hwnd
            himc = imm32.ImmGetContext(focused)
            if himc:
                conv = ctypes.c_ulong(0)
                sent = ctypes.c_ulong(0)
                if imm32.ImmGetConversionStatus(himc, ctypes.byref(conv),
                                                ctypes.byref(sent)):
                    if conv.value != 0:
                        # CJK conversion mode active — force alphanumeric
                        imm32.ImmSetConversionStatus(himc, 0, sent.value)
                        logger.debug("_ensure_english_ime: conv %d→0 (alphanumeric)", conv.value)
                        time.sleep(0.05)
                imm32.ImmReleaseContext(focused, himc)
        finally:
            if attached:
                user32.AttachThreadInput(our_tid, target_tid, False)
    except Exception as _e:
        logger.debug("_ensure_english_ime failed: %s", _e)


def _cb_paste(text: str) -> None:
    """Set text via clipboard paste — bypasses IME entirely.

    Chinese IMEs (Microsoft New Phonetic / Zhuyin) intercept typewrite() even
    after ImmSetConversionStatus because WPF manages IME focus independently.
    Pasting from the clipboard skips all IME processing.
    """
    pyperclip.copy(text)
    time.sleep(0.05)
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.08)


def _pg_type(text: str, interval: float = 0.05) -> None:
    _ensure_english_ime()
    pyautogui.typewrite(text, interval=interval)
    time.sleep(0.1)


def _pg_press(key: str) -> None:
    pyautogui.press(key)
    time.sleep(0.2)


# ---------------------------------------------------------------------------
# Window wait helper
# ---------------------------------------------------------------------------

def _wait_for_window(
    app: Application,
    title_re: str,
    timeout: float = 30.0,
    interval: float = 0.8,
) -> _HwndWrapper:
    return _wait_for_any_window(app, [title_re], timeout=timeout)


def _wait_for_any_window(
    app: Application,
    title_res: List[str],
    timeout: float = 30.0,
) -> _HwndWrapper:
    """
    Poll until a top-level window matching any regex in *title_res* appears.
    Returns an _HwndWrapper (not a process-scoped pywinauto spec) so callers
    can safely access .handle without triggering pywinauto's OpenProcess path.

    Uses Desktop(win32).windows() — enumerates ALL top-level windows with no
    process constraint, so it works even when MC runs as Administrator.
    """
    import re as _re
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            for win in Desktop(backend="win32").windows():
                try:
                    if not win.is_visible():
                        continue
                    title = win.window_text()
                    for tr in title_res:
                        if _re.match(tr, title):
                            hwnd = win.handle
                            logger.debug("_wait_for_any_window: found hwnd=%x title='%s'",
                                         hwnd, title)
                            return _HwndWrapper(hwnd)
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(0.2)
    raise WindowNotFoundError(f"None of {title_res} found after {timeout:.0f}s")


def _dismiss_error_dialogs(app: Application) -> None:
    for title_re in [r".*[Ee]rror.*", r".*[Ww]arning.*"]:
        try:
            dlg = app.window(title_re=title_re, class_name="#32770")
            if dlg.exists(timeout=0):
                msg = ""
                try:
                    msg = dlg.child_window(class_name="Static").window_text()
                except Exception:
                    pass
                logger.warning("Dismissing dialog: %s", msg)
                rect = dlg.rectangle()
                pyautogui.click((rect.left + rect.right) // 2,
                                (rect.top + rect.bottom) // 2)
                _pg_press("enter")
        except Exception:
            pass


def _save_screenshot(name: str, output_dir: str) -> None:
    try:
        path = os.path.join(output_dir, f"FAILURE_{name}_{int(time.time())}.png")
        pyautogui.screenshot(path)
        logger.info("Screenshot: %s", path)
    except Exception as e:
        logger.warning("Screenshot failed: %s", e)


# ---------------------------------------------------------------------------
# Workspace / chart management
# ---------------------------------------------------------------------------

def _close_optimization_report() -> None:
    """Close any open Optimization Report / ORVisualizer windows before the next run.

    Uses ctypes EnumWindows (bypasses UIPI) instead of pywinauto Desktop so it
    works even when MC64 is running at a different privilege level.
    """
    _KEYWORDS = ["Optimization Report", "ORVisualizer", "優化報告", "最佳化報告"]

    _closed_hwnds: list = []

    def _enum_cb(hwnd, _):
        try:
            length = _user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            _user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            if any(kw.lower() in title.lower() for kw in _KEYWORDS):
                if _user32.IsWindowVisible(hwnd):
                    _closed_hwnds.append((hwnd, title))
        except Exception:
            pass
        return True

    _WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
    try:
        _user32.EnumWindows(_WNDENUMPROC(_enum_cb), 0)
    except Exception as _ee:
        logger.debug("EnumWindows error in _close_optimization_report: %s", _ee)

    for hwnd, title in _closed_hwnds:
        try:
            _user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
            time.sleep(0.4)
            logger.info("Closed optimization report window: '%s'", title)
        except Exception as _ce:
            logger.debug("Could not close '%s': %s", title, _ce)

    if _closed_hwnds:
        time.sleep(0.5)
    else:
        logger.debug("_close_optimization_report: no report window found")


def ensure_chart_ready(conn: MultiChartsConnection, cfg: StrategyConfig) -> None:
    """Switch to the correct MC window for this strategy's workspace."""
    found = conn.switch_to_workspace(cfg.chart_workspace)
    if not found:
        if not os.path.exists(cfg.chart_workspace):
            raise MCUIError(
                f"Workspace file not found: {cfg.chart_workspace}\n"
                f"Please open it in MultiCharts manually."
            )
        logger.info("Workspace not open, attempting to open: %s", cfg.chart_workspace)
        _open_workspace_file(conn, cfg.chart_workspace)
        # Re-check after open attempt
        found = conn.switch_to_workspace(cfg.chart_workspace)
        if not found:
            raise MCUIError(
                f"Workspace '{Path(cfg.chart_workspace).stem}' could not be opened.\n"
                f"Please open it manually in MultiCharts64 before running the script:\n"
                f"  {cfg.chart_workspace}"
            )
    # Bring to foreground. If Optimization Report is blocking focus, close it first.
    if conn._hwnd:
        import win32gui as _w32g
        for _focus_try in range(3):
            _focus_window(conn._hwnd)
            time.sleep(0.5)
            try:
                _fg = _w32g.GetForegroundWindow()
                if _fg == conn._hwnd:
                    break
                # Check if the foreground window is an Optimization Report
                _fg_len = _user32.GetWindowTextLengthW(_fg)
                _fg_buf = ctypes.create_unicode_buffer(_fg_len + 1)
                _user32.GetWindowTextW(_fg, _fg_buf, _fg_len + 1)
                _fg_title = _fg_buf.value
                _REPORT_KW = ["Optimization Report", "ORVisualizer", "優化報告", "最佳化報告"]
                if any(kw.lower() in _fg_title.lower() for kw in _REPORT_KW):
                    logger.info("ensure_chart_ready: closing blocking report '%s'", _fg_title)
                    _user32.PostMessageW(_fg, 0x0010, 0, 0)  # WM_CLOSE
                    time.sleep(0.8)
            except Exception:
                break
    logger.info("Chart ready for: %s", cfg.name)


def _open_workspace_file(conn: MultiChartsConnection, workspace_path: str) -> None:
    """Open a workspace file via File > Open Workspace using pyautogui."""
    main = conn.main_window()
    rect = main.rectangle()
    cx = (rect.left + rect.right) // 2
    cy = (rect.top + rect.bottom) // 2

    # Focus MC
    pyautogui.click(cx, cy)
    time.sleep(0.4)

    # Alt+F to open File menu
    _pg_hotkey("alt", "f")
    time.sleep(0.5)

    # Press O for "Open Workspace..." or navigate menu
    _pg_press("o")
    time.sleep(0.5)

    # Look for open dialog (title varies by locale)
    try:
        open_dlg = _wait_for_any_window(
            conn.app,
            [r".*[Oo]pen.*", r".*開啟.*", r".*選擇.*"],
            timeout=8,
        )
        # Type the path and press Enter
        pyautogui.hotkey("ctrl", "a")
        pyautogui.typewrite(workspace_path, interval=0.03)
        _pg_press("enter")
        time.sleep(4)

        # Re-scan for the new window
        hwnd = _find_hwnd_for_workspace(workspace_path)
        if hwnd:
            conn._connect_to_hwnd(hwnd)
            logger.info("Workspace opened: %s", Path(workspace_path).stem)
    except WindowNotFoundError:
        logger.warning(
            "Open dialog didn't appear. Please open the workspace manually "
            "and re-run the script."
        )


# ---------------------------------------------------------------------------
# Set strategy date range (IS / OOS)
# ---------------------------------------------------------------------------

def set_strategy_date_range(
    conn: MultiChartsConnection,
    cfg: StrategyConfig,
    date_range: DateRange,
) -> None:
    """
    Set the strategy's begin/end backtest dates via Format Signals → General tab.
    Uses pyautogui for all input (bypasses admin restriction).
    """
    logger.info("Setting date range: %s to %s", date_range.from_date, date_range.to_date)

    format_dlg = _open_format_signals(conn)
    _select_signal_in_list(format_dlg, cfg.mc_signal_name)
    time.sleep(0.2)

    # Click "Format" button (not Optimize)
    _click_button_in_dlg(format_dlg, ["Format", "格式", "編輯"])
    time.sleep(0.8)

    signal_dlg = _wait_for_any_window(
        conn.app,
        [r".*Format.*", r".*Signal.*", r".*格式.*"],
        timeout=10,
    )

    # Click "General" tab
    _click_tab(signal_dlg, ["General", "一般"])
    time.sleep(0.3)

    # Enable "Begin date" checkbox and set date
    _set_date_field(signal_dlg, conn.app,
                    checkbox_titles=["Begin date:", "Begin Date", "開始日期", "起始日期"],
                    date_str=date_range.from_date)

    # Enable "End date" checkbox and set date
    _set_date_field(signal_dlg, conn.app,
                    checkbox_titles=["End date:", "End Date", "結束日期", "終止日期"],
                    date_str=date_range.to_date)

    # Click OK/Apply
    _click_button_in_dlg(signal_dlg, ["OK", "確定", "Apply", "套用"])
    time.sleep(0.5)
    _close_format_dialog(conn)
    logger.info("Date range set: %s ~ %s", date_range.from_date, date_range.to_date)


def _click_tab(dlg, tab_titles: List[str]) -> None:
    hwnd = dlg.handle if hasattr(dlg, "handle") else None

    # Primary: UIAutomation — works across UIPI privilege boundary
    if hwnd:
        uia = _uia_dlg(hwnd)
        if uia is not None:
            # Exact title match
            for title in tab_titles:
                try:
                    tab = uia.child_window(title=title, control_type="TabItem")
                    if tab.exists(timeout=2):
                        tab.click_input()
                        time.sleep(0.3)
                        logger.debug("UIA: clicked tab '%s'", title)
                        return
                except Exception:
                    pass
            # Partial/contains match across all TabItem descendants
            try:
                for tab in uia.descendants(control_type="TabItem"):
                    txt = tab.window_text()
                    if any(t.lower() in txt.lower() for t in tab_titles):
                        tab.click_input()
                        time.sleep(0.3)
                        logger.debug("UIA: clicked tab '%s'", txt)
                        return
            except Exception:
                pass

    # Fallback: ctypes — find SysTabControl32 child, click at estimated tab position.
    # Since SendMessage(TCM_GETITEM) is UIPI-blocked we can't read tab text,
    # so we map known tab names to approximate indices.
    if hwnd:
        tab_ctrl_hwnd = _win32_find_child(hwnd, class_name="SysTabControl32")
        if tab_ctrl_hwnd:
            l, t, r, b = _win32_get_rect(tab_ctrl_hwnd)
            tab_w = max(60, (r - l) // 6)
            _TAB_INDEX_HINTS = {
                "general": 0, "一般": 0,
                "inputs": 0, "輸入": 0,
                "signals": 1, "訊號": 1, "signal": 1,
                "properties": 1,
            }
            idx = 0
            for name in tab_titles:
                hint = _TAB_INDEX_HINTS.get(name.lower())
                if hint is not None:
                    idx = hint
                    break
            cx = l + tab_w * idx + tab_w // 2
            cy = t + 10
            pyautogui.click(cx, cy)
            time.sleep(0.25)
            logger.debug("Ctypes: clicked tab index %d (%s) at (%d,%d)", idx, tab_titles, cx, cy)
            return
        logger.debug("_click_tab: no SysTabControl32 found for hwnd=%x", hwnd)

    # Last resort: original pywinauto approach (may fail for admin windows)
    for title in tab_titles:
        try:
            tab = dlg.child_window(title=title, class_name="SysTabControl32")
            tab.click_input()
            return
        except Exception:
            pass


def _set_date_field(
    dlg,
    app: Application,
    checkbox_titles: List[str],
    date_str: str,
) -> None:
    """Enable a date checkbox and type the date string."""
    hwnd = dlg.handle if hasattr(dlg, "handle") else None
    children = _win32_enum_children(hwnd) if hwnd else []

    def _click_checkbox_and_edit(cb_hwnd: int) -> bool:
        l, t, r, b = _win32_get_rect(cb_hwnd)
        pyautogui.click((l + r) // 2, (t + b) // 2)
        time.sleep(0.2)
        # Find SysDateTimePick32 sibling/descendant near the checkbox
        dt_hwnd = _win32_find_child(hwnd, class_name="SysDateTimePick32") if hwnd else None
        if dt_hwnd:
            dl, dt, dr, db = _win32_get_rect(dt_hwnd)
            pyautogui.click((dl + dr) // 2, (dt + db) // 2)
            time.sleep(0.2)
            _pg_hotkey("ctrl", "a")
            _ensure_english_ime()
            pyautogui.typewrite(date_str.replace("/", ""), interval=0.05)
            return True
        # Fallback: type into whatever is focused after clicking checkbox
        pyautogui.press("tab")
        time.sleep(0.1)
        _pg_hotkey("ctrl", "a")
        _ensure_english_ime()
        pyautogui.typewrite(date_str.replace("/", ""), interval=0.05)
        return True

    # Primary: ctypes child enumeration
    for title in checkbox_titles:
        for ch_hwnd, cls, txt in children:
            if (title.lower() in txt.lower() and
                    cls.lower() in ("button", "static") and
                    _win32_is_visible(ch_hwnd)):
                if _click_checkbox_and_edit(ch_hwnd):
                    return

    # Secondary: UIA
    if hwnd:
        uia = _uia_dlg(hwnd)
        if uia is not None:
            for title in checkbox_titles:
                try:
                    cb = uia.child_window(title=title, control_type="CheckBox")
                    if cb.exists(timeout=1):
                        cb.click_input()
                        time.sleep(0.2)
                        try:
                            dp = uia.child_window(control_type="Pane",
                                                   class_name="SysDateTimePick32")
                            if dp.exists(timeout=1):
                                dp.click_input()
                                time.sleep(0.2)
                                _pg_hotkey("ctrl", "a")
                                _ensure_english_ime()
                                pyautogui.typewrite(date_str.replace("/", ""),
                                                    interval=0.05)
                        except Exception:
                            pass
                        return
                except Exception:
                    pass

    # Last resort: original pywinauto
    for title in checkbox_titles:
        try:
            cb = dlg.child_window(title=title, class_name="Button")
            if cb.exists(timeout=1):
                rect = cb.rectangle()
                if not cb.get_check_state():
                    pyautogui.click((rect.left + rect.right) // 2,
                                    (rect.top + rect.bottom) // 2)
                    time.sleep(0.2)
                try:
                    edit = dlg.child_window(class_name="SysDateTimePick32", found_index=0)
                    edit.set_focus()
                    edit_rect = edit.rectangle()
                    pyautogui.click((edit_rect.left + edit_rect.right) // 2,
                                    (edit_rect.top + edit_rect.bottom) // 2)
                    time.sleep(0.2)
                    _pg_hotkey("ctrl", "a")
                    _ensure_english_ime()
                    pyautogui.typewrite(date_str.replace("/", ""), interval=0.05)
                except Exception:
                    pass
                return
        except Exception:
            pass


def _close_format_dialog(conn: MultiChartsConnection) -> None:
    """Close any remaining Format-related dialog."""
    for title_re in [r".*Format.*", r".*格式.*"]:
        try:
            dlg = conn.app.window(title_re=title_re, class_name="#32770")
            if dlg.exists(timeout=0):
                _click_button_in_dlg(dlg, ["OK", "確定", "Close", "關閉", "Cancel", "取消"])
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Format Signals dialog helpers
# ---------------------------------------------------------------------------

def _find_popup_hwnd(timeout: float = 3.0) -> Optional[int]:
    """Find the most-recently-shown popup menu (#32768) via Desktop enumeration.

    Using Desktop() avoids the process-scoped lookup that fails for admin MC.
    Returns hwnd of the popup, or None if not found within *timeout* seconds.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            for win in Desktop(backend="win32").windows():
                try:
                    if win.class_name() == "#32768" and win.is_visible():
                        return win.handle
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(0.15)
    return None


def _pyautogui_click_popup_item(app: Application, item_titles: List[str]) -> bool:
    """
    Locate a popup menu item and click it with pyautogui.
    Pure-ctypes implementation — avoids pywinauto Desktop/menu() which can
    crash the process at the C level after repeated calls.
    Returns True if an item was found and clicked.
    """
    popup_hwnd = _find_popup_hwnd(timeout=1.5)
    if popup_hwnd is None:
        logger.info("_pyautogui_click_popup_item: no #32768 popup found within timeout")
        return False

    logger.info("_pyautogui_click_popup_item: popup=0x%x searching for %s", popup_hwnd, item_titles)

    # Primary: ctypes MN_GETHMENU + GetMenuItemInfo (avoids pywinauto crash).
    # MN_GETHMENU is the undocumented but well-known message (0x01E1) that
    # returns the HMENU handle for a #32768 popup-menu window.
    # IMPORTANT: dwTypeData must be c_void_p (raw address) so that
    # ctypes.addressof(buf) is stored correctly — c_wchar_p would let ctypes
    # copy the string internally, breaking the pointer passed to GetMenuItemInfoW.
    try:
        _u32 = ctypes.windll.user32
        _MN_GETHMENU      = 0x01E1
        _MF_BYPOSITION    = 0x0400
        _MIIM_STRING      = 0x0040
        _MIIM_FTYPE       = 0x0100

        class _MENUITEMINFOW(ctypes.Structure):
            _fields_ = [
                ("cbSize",        ctypes.c_uint),
                ("fMask",         ctypes.c_uint),
                ("fType",         ctypes.c_uint),
                ("fState",        ctypes.c_uint),
                ("wID",           ctypes.c_uint),
                ("hSubMenu",      ctypes.c_void_p),
                ("hbmpChecked",   ctypes.c_void_p),
                ("hbmpUnchecked", ctypes.c_void_p),
                ("dwItemData",    ctypes.c_size_t),   # ULONG_PTR (pointer-sized int)
                ("dwTypeData",    ctypes.c_void_p),   # LPWSTR stored as raw address
                ("cch",           ctypes.c_uint),
                ("hbmpItem",      ctypes.c_void_p),
            ]

        hmenu = _u32.SendMessageW(popup_hwnd, _MN_GETHMENU, 0, 0)
        logger.info("popup hmenu=0x%x", hmenu)
        if hmenu:
            count = _u32.GetMenuItemCount(hmenu)
            logger.info("popup item count=%d", count)
            _all_texts = []
            for i in range(count):
                buf = ctypes.create_unicode_buffer(512)
                mii = _MENUITEMINFOW()
                mii.cbSize = ctypes.sizeof(_MENUITEMINFOW)
                mii.fMask = _MIIM_STRING | _MIIM_FTYPE
                mii.dwTypeData = ctypes.addressof(buf)   # explicit address — not c_wchar_p copy
                mii.cch = 512
                ok = _u32.GetMenuItemInfoW(hmenu, i, _MF_BYPOSITION, ctypes.byref(mii))
                text = buf.value if ok else ""
                _all_texts.append(text)
                if text and any(t.lower() in text.lower() for t in item_titles):
                    rect = _RECT()   # must use module-level _RECT (matches argtypes prototype)
                    _u32.GetMenuItemRect(popup_hwnd, hmenu, i, ctypes.byref(rect))
                    x = (rect.left + rect.right) // 2
                    y = (rect.top + rect.bottom) // 2
                    logger.info("ctypes menu click [%d] '%s' at (%d,%d)", i, text, x, y)
                    pyautogui.click(x, y)
                    time.sleep(0.3)
                    return True
            logger.info("No match in popup items: %s", _all_texts)
    except Exception as _mge:
        logger.info("ctypes MN_GETHMENU enum failed: %s", _mge)

    # Fallback: enumerate child windows of the popup via ctypes
    _child_texts = []
    for h, cls, txt in _win32_enum_children(popup_hwnd):
        _child_texts.append(txt)
        if any(t.lower() in txt.lower() for t in item_titles) and _win32_is_visible(h):
            l, t_, r_, b = _win32_get_rect(h)
            pyautogui.click((l + r_) // 2, (t_ + b) // 2)
            time.sleep(0.3)
            logger.info("Ctypes popup click child: '%s'", txt)
            return True
    logger.info("No match in child windows: %s", _child_texts[:20])
    return False


def _open_format_signals(conn: MultiChartsConnection) -> _HwndWrapper:
    """Open Format Signals/Objects dialog.

    Returns _HwndWrapper (not a process-scoped pywinauto spec) so callers
    can use .handle without triggering pywinauto's OpenProcess privilege check.
    """
    import re as _re
    # Reuse if already open — scan ALL top-level windows (no process constraint)
    _fmt_patterns = [r".*Format Objects.*", r".*Format Signals.*", r".*Format Strategies.*",
                     r".*格式物件.*", r".*格式訊號.*", r".*格式策略.*"]
    try:
        for win in Desktop(backend="win32").windows():
            try:
                if not win.is_visible():
                    continue
                title = win.window_text()
                for pat in _fmt_patterns:
                    if _re.match(pat, title):
                        hwnd = win.handle
                        logger.info("Format dialog already open (hwnd=%x title='%s') — reusing", hwnd, title)
                        l, t, r, b = _win32_get_rect(hwnd)
                        pyautogui.click((l + r) // 2, (t + b) // 2)
                        time.sleep(0.3)
                        return _HwndWrapper(hwnd)
            except Exception:
                pass
    except Exception:
        pass

    _focus_window(conn._hwnd)
    time.sleep(0.2)

    # Use ctypes GetWindowRect (no process constraint) to get MC window bounds.
    import win32gui as _w32
    try:
        _r = _w32.GetWindowRect(conn._hwnd)
        l, t, r, b = _r[0], _r[1], _r[2], _r[3]
    except Exception as _re:
        logger.warning("win32gui.GetWindowRect(hwnd=%x) failed: %s; using ctypes",
                       conn._hwnd, _re)
        l, t, r, b = _win32_get_rect(conn._hwnd)
    logger.info("MC window rect: hwnd=0x%x  left=%d top=%d right=%d bottom=%d",
                conn._hwnd, l, t, r, b)

    # If rect is degenerate (window minimised or hwnd stale), try to restore and re-read
    if r - l < 100 or b - t < 100:
        logger.warning("MC is minimized (rect=%d,%d,%d,%d) — restoring via taskbar",
                       l, t, r, b)
        _click_taskbar_button_for_hwnd(conn._hwnd)
        time.sleep(1.5)
        try:
            _r2 = _w32.GetWindowRect(conn._hwnd)
            l, t, r, b = _r2[0], _r2[1], _r2[2], _r2[3]
            logger.info("After restore: rect=%d,%d,%d,%d", l, t, r, b)
        except Exception:
            l, t, r, b = _win32_get_rect(conn._hwnd)

        # Still degenerate: scan all windows for any visible MC window
        if r - l < 100 or b - t < 100:
            logger.warning("Still degenerate — scanning all top-level windows for MC")
            for _win in Desktop(backend="win32").windows():
                try:
                    _title = _win.window_text()
                    if "MultiCharts" in _title and "QuoteManager" not in _title:
                        _wl, _wt, _wr, _wb = _win32_get_rect(_win.handle)
                        if _wr - _wl > 200 and _wb - _wt > 50:
                            l, t, r, b = _wl, _wt, _wr, _wb
                            conn._hwnd = _win.handle
                            logger.info("Re-found MC hwnd=%x rect=(%d,%d,%d,%d)",
                                        conn._hwnd, l, t, r, b)
                            break
                except Exception:
                    pass

    rect_left, rect_top, rect_right, rect_bottom = l, t, r, b
    win_h = rect_bottom - rect_top
    cx = (rect_left + rect_right) // 2

    if win_h < 50 or cx == 0:
        raise MCUIError(
            f"MC window rect is invalid (hwnd=0x{conn._hwnd:x} "
            f"rect={rect_left},{rect_top},{rect_right},{rect_bottom}). "
            "Ensure MultiCharts is open and visible (not minimized)."
        )

    # Ensure MC has foreground focus before any click — call _focus_window which
    # scans for an unobstructed pixel using WindowFromPoint, then verify.
    import win32gui as _w32fg
    _focus_window(conn._hwnd)
    time.sleep(0.15)
    _fg_after_focus = 0
    try:
        _fg_after_focus = _w32fg.GetForegroundWindow()
        logger.info("Foreground after _focus_window: hwnd=0x%x '%s' (MC=0x%x, match=%s)",
                    _fg_after_focus, _w32fg.GetWindowText(_fg_after_focus),
                    conn._hwnd, _fg_after_focus == conn._hwnd)
    except Exception:
        pass

    # --- Attempt 1a: UIAutomation menu bar click (cross-privilege, no message-sending) ---
    _fmt_dropdown_items = [
        "Strategies...", "Signals...", "Format Signals...",
        "格式訊號...", "格式策略...", "訊號...", "策略...",
    ]
    try:
        from pywinauto import Desktop as _UIA_D2
        uia_mc = _UIA_D2(backend="uia").window(handle=conn._hwnd)
        _found_format = False
        for mb in uia_mc.descendants(control_type="MenuBar"):
            for mi in mb.descendants(control_type="MenuItem"):
                txt = mi.window_text().strip()
                logger.info("UIA menu item: '%s'", txt)
                if "format" in txt.lower() or "格式" in txt.lower():
                    mi.click_input()
                    time.sleep(0.3)
                    logger.info("UIA: clicked Format menu item '%s'", txt)
                    _found_format = True
                    break
            if _found_format:
                break
        if _found_format:
            _pyautogui_click_popup_item(conn.app, _fmt_dropdown_items)
            try:
                return _wait_for_any_window(conn.app, [r".*Format.*", r".*格式.*"], timeout=8)
            except WindowNotFoundError:
                pass
            _pg_press("escape")
            time.sleep(0.2)
    except Exception as e:
        logger.info("UIA menu bar approach failed: %s", e)

    # --- Attempt 1b: ctypes GetMenuItemRect + pyautogui click (no message-sending) ---
    try:
        _MF_BYPOSITION = 0x0400
        _hMenu = _user32.GetMenu(conn._hwnd)
        _count = _user32.GetMenuItemCount(_hMenu)
        logger.info("ctypes menu: GetMenu=%s count=%s", hex(_hMenu) if _hMenu else "NULL", _count)
        for _i in range(_count):
            _buf = ctypes.create_unicode_buffer(256)
            _user32.GetMenuStringW(_hMenu, _i, _buf, 256, _MF_BYPOSITION)
            _text = _buf.value.replace("&", "").strip()
            logger.info("ctypes menu item[%d]: '%s'", _i, _text)
            if "format" in _text.lower() or "格式" in _text.lower():
                _mr = _RECT()
                _user32.GetMenuItemRect(conn._hwnd, _hMenu, _i, ctypes.byref(_mr))
                _mcx = (_mr.left + _mr.right) // 2
                _mcy = (_mr.top + _mr.bottom) // 2
                logger.info("ctypes Format menu rect: (%d,%d,%d,%d) → clicking (%d,%d)",
                            _mr.left, _mr.top, _mr.right, _mr.bottom, _mcx, _mcy)
                pyautogui.click(_mcx, _mcy)
                time.sleep(0.3)
                _pyautogui_click_popup_item(conn.app, _fmt_dropdown_items)
                try:
                    return _wait_for_any_window(conn.app, [r".*Format.*", r".*格式.*"], timeout=8)
                except WindowNotFoundError:
                    pass
                _pg_press("escape")
                time.sleep(0.2)
                break
    except Exception as e:
        logger.info("ctypes menu approach failed: %s", e)

    # --- Attempt 1c: keyboard navigation (Alt → accelerator) ────────────────────
    # After taskbar-click focus, Alt key activates MC's WTL command bar navigation.
    # Press Escape first to clear any chart cursor mode, then Alt+letter to open Format.
    try:
        pyautogui.press('escape')
        time.sleep(0.2)
        # Snapshot windows before Alt press
        _pre_hwnds = set()
        for _w in Desktop(backend="win32").windows():
            try:
                _pre_hwnds.add(_w.handle)
            except Exception:
                pass
        pyautogui.press('alt')
        time.sleep(0.2)
        # Log any new window that appeared (menu popup candidate)
        for _w in Desktop(backend="win32").windows():
            try:
                if _w.handle not in _pre_hwnds and _w.is_visible():
                    logger.info("New window after Alt: hwnd=%x cls=%s txt=%s",
                                _w.handle, _w.class_name(), _w.window_text()[:40])
            except Exception:
                pass
        # Press 'f' to open "Format" (or try arrow-key navigation)
        pyautogui.press('f')
        time.sleep(0.2)
        popup_after_alt = _find_popup_hwnd(timeout=1.2)
        if not popup_after_alt:
            # Try finding any new popup-like window
            for _w in Desktop(backend="win32").windows():
                try:
                    if _w.handle not in _pre_hwnds and _w.is_visible():
                        _cls = _w.class_name()
                        if _cls in ["#32768", "SysShadow", "ToolbarWindow32"]:
                            popup_after_alt = _w.handle
                            break
                except Exception:
                    pass
        if popup_after_alt:
            logger.info("Popup via Alt+F: hwnd=%x", popup_after_alt)
            _pyautogui_click_popup_item(conn.app, _fmt_dropdown_items)
            try:
                return _wait_for_any_window(conn.app, [r".*Format.*", r".*格式.*"], timeout=8)
            except WindowNotFoundError:
                pass
            _pg_press("escape")
            time.sleep(0.2)
        else:
            logger.info("Alt+F: no popup found — pressing Escape to reset")
            pyautogui.press('escape')
            time.sleep(0.2)
    except Exception as _e:
        logger.info("Keyboard menu approach failed: %s", _e)

    # --- Attempt 2: right-click on the chart MDI child directly ---
    # Re-verify focus; enumerate MC's descendants to find the chart MDI child
    # (ATL_MCMDIChildFrame) and right-click inside it.
    try:
        _fg_now = _w32fg.GetForegroundWindow()
        if _fg_now != conn._hwnd:
            logger.info("MC lost focus before right-click (fg=0x%x) — re-focusing", _fg_now)
            _focus_window(conn._hwnd)
            time.sleep(0.4)
    except Exception:
        pass

    # Find the chart MDI child window and its centre
    _chart_hwnd = None
    _rclick_x, _rclick_y = cx, rect_top + int(win_h * 0.40)  # fallback
    try:
        _all_children = _win32_enum_children(conn._hwnd)
        logger.info("MC descendants (%d total): %s",
                    len(_all_children),
                    [(hex(h), c, t[:25]) for h, c, t in _all_children[:15]])
        # Priority order: ATL_MCGraphPanel > ATL_MCMDIChildFrame > ATL_MCChartManager
        # ATL_MCGraphPanel is the actual price-plot canvas where right-click shows
        # "Format Signals/Strategies".  Price/time scales give a different menu.
        for _chart_cls in ["ATL_MCGraphPanel", "ATL_MCMDIChildFrame", "ATL_MCChartManager"]:
            for _h, _cls, _ttl in _all_children:
                if _cls == _chart_cls and _win32_is_visible(_h):
                    _cr = _win32_get_rect(_h)
                    _cw = _cr[2] - _cr[0]
                    _ch = _cr[3] - _cr[1]
                    if _cw > 50 and _ch > 50:
                        _chart_hwnd = _h
                        # Find a click position that is visually blank (no price bars).
                        # Strategy: try right-edge x first (the "future" area after the
                        # last bar is always empty), then fall back to center x.
                        # For y: scan from near the top (blank space above price range)
                        # rather than 20% down (which may land on a price bar).
                        _y_start = _cr[1] + 8  # start 8px from panel top
                        # Default fallback: vertical center of panel (more reliable than top edge)
                        _best_x, _best_y = _cr[2] - 35, _cr[1] + _ch // 2
                        _pos_found = False
                        for _try_x in [_cr[2] - 35, _cr[0] + _cw // 2]:
                            if _pos_found:
                                break
                            for _dy in range(0, _ch - 10, 5):
                                _ty = _y_start + _dy
                                try:
                                    _wfp_h = _w32fg.WindowFromPoint((_try_x, _ty))
                                    if _wfp_h == _h:
                                        _best_x, _best_y = _try_x, _ty
                                        _pos_found = True
                                        break
                                except Exception:
                                    break
                        _rclick_x, _rclick_y = _best_x, _best_y
                        logger.info("Chart child: cls=%s hwnd=%s rect=%s click=(%d,%d)",
                                    _cls, hex(_h), _cr, _rclick_x, _rclick_y)
                        break
            if _chart_hwnd:
                break
    except Exception as _e:
        logger.info("MDI child scan failed: %s", _e)

    # Log foreground window before right-click for diagnostics
    try:
        import win32gui as _wg2
        _fg = _wg2.GetForegroundWindow()
        _fg_title = _wg2.GetWindowText(_fg)
        logger.info("Foreground before right-click: hwnd=0x%x '%s' (MC=0x%x)",
                    _fg, _fg_title, conn._hwnd)
    except Exception:
        pass

    # Confirm which window is actually at the click position
    try:
        _wfp = _w32fg.WindowFromPoint((_rclick_x, _rclick_y))
        _wfp_cls = ctypes.create_unicode_buffer(256)
        _user32.GetClassNameW(_wfp, _wfp_cls, 256)
        _wfp_txt = _w32fg.GetWindowText(_wfp)
        logger.info("WindowFromPoint(%d,%d) = hwnd=0x%x cls='%s' txt='%s'",
                    _rclick_x, _rclick_y, _wfp, _wfp_cls.value, _wfp_txt[:30])
    except Exception as _e:
        logger.debug("WindowFromPoint failed: %s", _e)

    # Left-click to activate the chart MDI child, then right-click.
    # Attempt 1: pyautogui — may fail if ATL_MCGraphPanel is transparent to hit-testing.
    logger.info("Left-click at (%d,%d) to activate chart child", _rclick_x, _rclick_y)
    pyautogui.click(_rclick_x, _rclick_y)
    time.sleep(0.25)
    logger.info("Right-clicking at (%d, %d)", _rclick_x, _rclick_y)
    pyautogui.rightClick(_rclick_x, _rclick_y)
    time.sleep(0.5)

    # Snapshot all windows before/after to catch ANY new popup (MC may not use #32768)
    _snap_before = set()
    for _w in Desktop(backend="win32").windows():
        try:
            _snap_before.add(_w.handle)
        except Exception:
            pass

    popup_hwnd_debug = _find_popup_hwnd(timeout=1.5)
    _new_wins = []
    for _w in Desktop(backend="win32").windows():
        try:
            if _w.handle not in _snap_before and _w.is_visible():
                _new_wins.append((_w.handle, _w.class_name(), _w.window_text()[:30]))
        except Exception:
            pass

    if popup_hwnd_debug:
        try:
            popup_spec = Desktop(backend="win32").window(handle=popup_hwnd_debug)
            items = [popup_spec.menu().item(i).text()
                     for i in range(popup_spec.menu().item_count())]
            logger.info("Context menu items found: %s", items)
        except Exception:
            logger.info("Popup hwnd=%x found but couldn't enumerate items", popup_hwnd_debug)
    else:
        logger.info("No popup (#32768) after right-click. New windows: %s", _new_wins)

    # Attempt 2: PostMessage(WM_RBUTTONDOWN/UP) directly to the chart panel hwnd.
    # ATL_MCGraphPanel may have WS_EX_TRANSPARENT so pyautogui clicks pass through
    # to the parent frame. PostMessage bypasses WindowFromPoint entirely and
    # delivers WM_RBUTTONDOWN straight to the child window's message queue.
    # NOTE: use _p_u32 (not _user32) to avoid Python UnboundLocalError — assigning
    # any name in a function makes Python treat ALL references to that name as local.
    if not popup_hwnd_debug and _chart_hwnd:
        try:
            _WM_RBUTTONDOWN = 0x0204
            _WM_RBUTTONUP   = 0x0205
            _WM_CONTEXTMENU = 0x007B
            _MK_RBUTTON     = 0x0002
            _p_u32 = ctypes.windll.user32
            _pcr = _win32_get_rect(_chart_hwnd)
            _pw  = _pcr[2] - _pcr[0]   # panel width
            _ph  = _pcr[3] - _pcr[1]   # panel height
            # Use the same position as the pyautogui click: right-edge x (avoids
            # price bars / indicator lines that trigger a different context menu),
            # vertical center y.  Convert screen→client by subtracting panel origin.
            _cli_x = _rclick_x - _pcr[0]   # matches pyautogui click x
            _cli_y = _rclick_y - _pcr[1]   # matches pyautogui click y
            # Clamp to [10, panel-10] just in case
            _cli_x = max(10, min(_pw - 10, _cli_x))
            _cli_y = max(10, min(_ph - 10, _cli_y))
            _cli_lp = (_cli_y << 16) | (_cli_x & 0xFFFF)
            # Screen coords for WM_CONTEXTMENU fallback
            _scr_x = _pcr[0] + _cli_x
            _scr_y = _pcr[1] + _cli_y
            _scr_lp = (_scr_y << 16) | (_scr_x & 0xFFFF)
            logger.info("PostMessage right-click: hwnd=0x%x client=(%d,%d) screen=(%d,%d)",
                        _chart_hwnd, _cli_x, _cli_y, _scr_x, _scr_y)
            # Activate the chart window first
            _p_u32.SetForegroundWindow(_chart_hwnd)
            time.sleep(0.3)
            # Send WM_RBUTTONDOWN + WM_RBUTTONUP (triggers WM_CONTEXTMENU internally)
            _p_u32.PostMessageW(_chart_hwnd, _WM_RBUTTONDOWN, _MK_RBUTTON, _cli_lp)
            time.sleep(0.15)
            _p_u32.PostMessageW(_chart_hwnd, _WM_RBUTTONUP,   0,           _cli_lp)
            time.sleep(0.7)
            popup_hwnd_debug = _find_popup_hwnd(timeout=1.5)
            if popup_hwnd_debug:
                logger.info("PostMessage WM_RBUTTON* → popup 0x%x", popup_hwnd_debug)
            else:
                # Fallback B: WM_CONTEXTMENU with screen coords
                _p_u32.PostMessageW(_chart_hwnd, _WM_CONTEXTMENU, _chart_hwnd, _scr_lp)
                time.sleep(0.7)
                popup_hwnd_debug = _find_popup_hwnd(timeout=1.5)
                logger.info("PostMessage WM_CONTEXTMENU → popup %s",
                            hex(popup_hwnd_debug) if popup_hwnd_debug else None)
        except Exception as _pme:
            logger.warning("PostMessage right-click failed: %s", _pme)

    target_names = [
        "Format Signals...", "Format Objects...", "Format Strategies...",
        "格式物件...", "格式訊號...", "格式策略...",
    ]
    if _pyautogui_click_popup_item(conn.app, target_names):
        try:
            return _wait_for_any_window(
                conn.app, [r".*Format.*", r".*格式.*"], timeout=8
            )
        except WindowNotFoundError:
            pass

    _pg_press("escape")
    time.sleep(0.2)

    raise WindowNotFoundError(
        "Could not open Format Signals/Objects dialog.\n"
        "Ensure the chart is visible, has a strategy applied, and MC is in the foreground."
    )


def _select_signal_in_list(
    format_dlg: pywinauto.WindowSpecification,
    signal_name: str,
) -> None:
    # Ensure the Signals tab is active
    _click_tab(format_dlg, ["Signals", "訊號", "Signal"])
    time.sleep(0.5)

    hwnd = format_dlg.handle

    # ── Approach 1: UIAutomation (works across UIPI privilege boundary) ──────
    uia = _uia_dlg(hwnd)
    if uia is not None:
        for ct in ["List", "DataGrid", "Table", "ListBox"]:
            try:
                lv = uia.child_window(control_type=ct)
                if not lv.exists(timeout=2):
                    continue
                items = lv.descendants(control_type="ListItem")
                if not items:
                    items = lv.descendants(control_type="DataItem")
                matched = [it for it in items
                           if signal_name.lower() in it.window_text().lower()]
                if matched:
                    matched[0].click_input()
                    logger.info("UIA: selected signal '%s'", matched[0].window_text())
                    return
                if items:
                    # Single-signal chart — select whatever is there
                    items[0].click_input()
                    logger.warning("UIA: signal '%s' not found; selected '%s'",
                                   signal_name, items[0].window_text())
                    return
            except Exception as e:
                logger.debug("UIA list type %s failed: %s", ct, e)
        logger.debug("UIA: no list items found for signal selection")

    # ── Approach 2: ctypes child enumeration ─────────────────────────────────
    lv_hwnd = None
    lv_cls = None
    children_info = _win32_enum_children(hwnd)
    logger.debug("Format Objects ctypes children: %s",
                 [(hex(h), c, t) for h, c, t in children_info])

    for list_cls in ["SysListView32", "ListBox", "ListView20WndClass", "TListView"]:
        for h, cls, txt in children_info:
            if cls.lower() == list_cls.lower() and _win32_is_visible(h):
                lv_hwnd = h
                lv_cls = cls
                break
        if lv_hwnd:
            break

    if lv_hwnd:
        l, t, r, b = _win32_get_rect(lv_hwnd)
        # Click near the top of the list to focus it, then Home to select first item
        pyautogui.click((l + r) // 2, t + 12)
        time.sleep(0.2)
        pyautogui.hotkey("ctrl", "home")
        time.sleep(0.1)
        logger.info("Ctypes: focused signal list (class=%s); assuming '%s' selected",
                    lv_cls, signal_name)
        return

    # ── Approach 3: last resort — assume single signal already selected ───────
    # Log child classes so we can diagnose further if still failing
    logger.warning(
        "Could not find signal list via UIA or ctypes for hwnd=%x. "
        "Assuming single signal '%s' is already active. "
        "Children: %s",
        hwnd, signal_name,
        [(hex(h), c) for h, c, _ in children_info],
    )
    # Proceed — single-signal charts have the signal pre-selected


# ── Signal Status (On/Off checkbox) API ──────────────────────────────────────
# Used by multi-signal module workflows: keep the main signal ON and toggle
# exit-module signals one at a time.  All discovery goes through UIA
# (Desktop backend) — never process-scoped pywinauto (UIPI).

def _find_signal_item(format_dlg, signal_name: str):
    """Return the UIA ListItem/DataItem for *signal_name* in the Format
    Signals list, or None.  Mirrors _select_signal_in_list's discovery."""
    hwnd = format_dlg.handle
    uia = _uia_dlg(hwnd)
    if uia is None:
        return None
    for ct in ["List", "DataGrid", "Table", "ListBox"]:
        try:
            lv = uia.child_window(control_type=ct)
            if not lv.exists(timeout=2):
                continue
            items = lv.descendants(control_type="ListItem")
            if not items:
                items = lv.descendants(control_type="DataItem")
            for it in items:
                try:
                    if signal_name.lower() in it.window_text().lower():
                        return it
                    # Some MC builds put the name in a Text descendant, not
                    # the item's own window_text
                    txts = [d.window_text() for d in it.descendants(control_type="Text")]
                    if any(signal_name.lower() in (t or "").lower() for t in txts):
                        return it
                except Exception:
                    pass
        except Exception as e:
            logger.debug("_find_signal_item: list type %s failed: %s", ct, e)
    return None


def _signal_item_state(item) -> Optional[bool]:
    """Read the enabled/checked state of a signal list item.
    True=on, False=off, None=unknown."""
    # 1. CheckBox descendant with TogglePattern
    try:
        cbs = list(item.descendants(control_type="CheckBox"))
        if cbs:
            try:
                return cbs[0].iface_toggle.CurrentToggleState == 1
            except Exception:
                try:
                    return cbs[0].is_checked()
                except Exception:
                    pass
    except Exception:
        pass
    # 2. The item itself may support TogglePattern (SysListView32 checkboxes)
    try:
        return item.iface_toggle.CurrentToggleState == 1
    except Exception:
        pass
    # 3. Legacy accessibility state bit (STATE_SYSTEM_CHECKED = 0x10)
    try:
        st = item.legacy_properties().get("State", 0)
        return bool(st & 0x10)
    except Exception:
        pass
    return None


def get_signal_status(format_dlg, signal_name: str) -> Optional[bool]:
    """Read a signal's Status checkbox. True=on, False=off, None=not found/unknown."""
    item = _find_signal_item(format_dlg, signal_name)
    if item is None:
        logger.warning("get_signal_status: signal '%s' not found in list", signal_name)
        return None
    return _signal_item_state(item)


def _read_signal_states(format_dlg, signal_names) -> Dict[str, Optional[bool]]:
    """ONE-PASS read of the current Status of every signal in *signal_names*.
    Much faster than calling get_signal_status per signal (single UIA scan)."""
    states: Dict[str, Optional[bool]] = {n: None for n in signal_names}
    uia = _uia_dlg(format_dlg.handle)
    if uia is None:
        return states
    for ct in ["List", "DataGrid", "Table", "ListBox"]:
        try:
            lv = uia.child_window(control_type=ct)
            if not lv.exists(timeout=2):
                continue
            items = lv.descendants(control_type="ListItem")
            if not items:
                items = lv.descendants(control_type="DataItem")
            for it in items:
                try:
                    label = (it.window_text() or "").lower()
                    if not any(n.lower() in label for n in signal_names):
                        txts = [d.window_text() or "" for d in
                                it.descendants(control_type="Text")]
                        label = " ".join(txts).lower()
                    for n in signal_names:
                        if states[n] is None and n.lower() in label:
                            states[n] = _signal_item_state(it)
                            break
                except Exception:
                    pass
            if any(v is not None for v in states.values()):
                return states
        except Exception as e:
            logger.debug("_read_signal_states: list type %s failed: %s", ct, e)
    return states


def set_signal_status(format_dlg, signal_name: str, enabled: bool) -> bool:
    """Set a signal's Status checkbox to *enabled*.  Returns True only when the
    new state was VERIFIED by read-back (never assumes a click worked)."""
    item = _find_signal_item(format_dlg, signal_name)
    if item is None:
        logger.error("set_signal_status: signal '%s' not found", signal_name)
        return False

    cur = _signal_item_state(item)
    logger.info("set_signal_status('%s', %s): current=%s", signal_name, enabled, cur)
    if cur is enabled:
        return True

    for attempt in range(3):
        try:
            if attempt == 0:
                # Path 1: click the CheckBox descendant directly
                cbs = list(item.descendants(control_type="CheckBox"))
                if cbs:
                    cbs[0].click_input()
                else:
                    # ListItem with its own TogglePattern
                    try:
                        item.iface_toggle.Toggle()
                    except Exception:
                        raise RuntimeError("no checkbox descendant / toggle")
            elif attempt == 1:
                # Path 2: select the row (click its text), then Space toggles
                # the checkbox in SysListView32 LVS_EX_CHECKBOXES lists
                item.click_input()
                time.sleep(0.2)
                _pg_press("space")
            else:
                # Path 3: coordinate click on the checkbox column (left edge)
                r = item.rectangle()
                pyautogui.click(r.left + 8, (r.top + r.bottom) // 2)
            time.sleep(0.35)
        except Exception as e:
            logger.debug("set_signal_status path %d failed: %s", attempt + 1, e)

        # Re-find the item (UIA refs can go stale after clicks) and verify
        item = _find_signal_item(format_dlg, signal_name) or item
        new = _signal_item_state(item)
        logger.info("set_signal_status('%s'): after path %d state=%s",
                    signal_name, attempt + 1, new)
        if new is enabled:
            return True
        if new is None and attempt == 2:
            logger.warning("set_signal_status('%s'): state unreadable — NOT verified",
                           signal_name)
            return False
    logger.error("set_signal_status('%s', %s): all paths failed (state=%s)",
                 signal_name, enabled, _signal_item_state(item))
    return False


def set_signal_statuses(
    conn: "MultiChartsConnection",
    status_map: Dict[str, bool],
    verify: bool = True,
    protected: Optional[List[str]] = None,
) -> Dict[str, Optional[bool]]:
    """Open Format Signals, apply Status for every signal in *status_map*,
    click OK, then (optionally) reopen and verify.  Raises MCUIError on any
    unverified state.  Signals in *protected* must never be turned off."""
    protected = protected or []
    for sig in protected:
        if status_map.get(sig) is False:
            raise ValueError(f"Refusing to disable protected signal '{sig}'")

    dlg = _open_format_signals(conn)
    _click_tab(dlg, ["Signals", "訊號", "Signal"])
    time.sleep(0.5)

    # Fast path: ONE scan for all current states; only toggle the diffs
    # (between consecutive module attempts only 2 signals actually change).
    _t0 = time.time()
    current = _read_signal_states(dlg, list(status_map.keys()))
    logger.info("set_signal_statuses: current=%s (read in %.1fs)",
                current, time.time() - _t0)

    results: Dict[str, Optional[bool]] = {}
    for sig, want in status_map.items():
        if current.get(sig) is want:
            results[sig] = want   # already correct — no per-signal rescan needed
            continue
        ok = set_signal_status(dlg, sig, want)
        results[sig] = want if ok else None
        if not ok:
            logger.error("set_signal_statuses: '%s' -> %s FAILED", sig, want)

    # Commit with OK (Escape would cancel all changes)
    if not _click_button_in_dlg(dlg, ["OK", "確定"]):
        _pg_press("enter")
    time.sleep(1.0)

    if any(v is None for v in results.values()):
        raise MCUIError(f"Signal status not verified during apply: "
                        f"{[s for s, v in results.items() if v is None]}")

    if verify:
        dlg2 = _open_format_signals(conn)
        _click_tab(dlg2, ["Signals", "訊號", "Signal"])
        time.sleep(0.5)
        mismatches = []
        for sig, want in status_map.items():
            got = get_signal_status(dlg2, sig)
            logger.info("verify status '%s': want=%s got=%s", sig, want, got)
            if got is not want:
                mismatches.append((sig, want, got))
        # Close read-only verify pass without applying anything
        if not _click_button_in_dlg(dlg2, ["Cancel", "取消"]):
            _pg_press("escape")
        time.sleep(0.7)
        if mismatches:
            raise MCUIError(f"Signal status verify failed: {mismatches}")

    return results


def _click_button_in_dlg(dlg, titles: List[str]) -> bool:
    hwnd = dlg.handle if hasattr(dlg, "handle") else None

    # Primary: ctypes child enumeration — bypasses pywinauto process-privilege checks
    if hwnd:
        for title in titles:
            btn_hwnd = _win32_find_child(hwnd, class_name="Button",
                                         title_contains=title)
            if btn_hwnd:
                _win32_click_hwnd(btn_hwnd)
                logger.debug("Ctypes: clicked button '%s'", title)
                return True
        # Also try without class_name restriction (some buttons use other classes)
        for title in titles:
            for child_hwnd, cls, txt in _win32_enum_children(hwnd):
                if title.lower() in txt.lower() and _win32_is_visible(child_hwnd):
                    _win32_click_hwnd(child_hwnd)
                    logger.debug("Ctypes: clicked control '%s' (class=%s)", txt, cls)
                    return True

    # Secondary: UIAutomation
    if hwnd:
        uia = _uia_dlg(hwnd)
        if uia is not None:
            for title in titles:
                try:
                    btn = uia.child_window(title=title, control_type="Button")
                    if btn.exists(timeout=1):
                        btn.click_input()
                        time.sleep(0.3)
                        logger.debug("UIA: clicked button '%s'", title)
                        return True
                except Exception:
                    pass
            # Partial match
            try:
                for btn in uia.descendants(control_type="Button"):
                    txt = btn.window_text()
                    if any(t.lower() in txt.lower() for t in titles):
                        btn.click_input()
                        time.sleep(0.3)
                        logger.debug("UIA: clicked button '%s'", txt)
                        return True
            except Exception:
                pass

    # Last resort: original pywinauto
    for title in titles:
        try:
            btn = dlg.child_window(title=title, class_name="Button")
            if btn.exists(timeout=0):
                rect = btn.rectangle()
                pyautogui.click((rect.left + rect.right) // 2,
                                (rect.top + rect.bottom) // 2)
                time.sleep(0.3)
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Optimization wizard
# ---------------------------------------------------------------------------

def _get_dpi_scale(hwnd: int) -> float:
    try:
        import ctypes as _ct
        v = _ct.windll.user32.GetDpiForWindow(hwnd)
        return v / 96.0 if v > 0 else 1.0
    except Exception:
        return 1.0


def _click_next_in_wizard(dlg, dpi_scale: float = 1.0) -> bool:
    """Click the Next button in the MC optimization wizard (WPF-safe).

    WPF wizards don't expose a Win32 Button child for Next — we must use
    UIAutomation invoke().  Falls back to keyboard (Alt+N) and coordinate
    click at the bottom-right of the dialog.
    """
    hwnd = dlg.handle if hasattr(dlg, "handle") else None
    NEXT_TITLES = ["Next >", "Next", "下一步", "下一步(N)", "下一步 >", ">"]

    # 1. UIA invoke — most reliable for WPF
    if hwnd:
        uia = _uia_dlg(hwnd)
        if uia:
            # Dump all UIA buttons so we can see what's there
            try:
                all_btns = list(uia.descendants(control_type="Button"))
                logger.info("_click_next: UIA buttons (%d): %s",
                            len(all_btns),
                            [(b.window_text() or "")[:20] for b in all_btns])
            except Exception:
                all_btns = []

            # Try exact title first
            for title in NEXT_TITLES:
                try:
                    btn = uia.child_window(title=title, control_type="Button")
                    if btn.exists(timeout=1):
                        try:
                            btn.invoke()
                        except Exception:
                            btn.click_input()
                        logger.info("_click_next: invoked '%s'", title)
                        return True
                except Exception:
                    pass

            # Partial match over all buttons
            for btn in all_btns:
                txt = (btn.window_text() or "").strip()
                if any(t.lower() in txt.lower() for t in NEXT_TITLES):
                    try:
                        try:
                            btn.invoke()
                        except Exception:
                            btn.click_input()
                        logger.info("_click_next: invoked partial '%s'", txt)
                        return True
                    except Exception as _be:
                        logger.warning("_click_next: button '%s' invoke failed: %s", txt, _be)

    # 2. Win32 Button children
    if hwnd:
        for title in NEXT_TITLES:
            btn_hwnd = _win32_find_child(hwnd, class_name="Button", title_contains=title)
            if btn_hwnd:
                _win32_click_hwnd(btn_hwnd)
                logger.info("_click_next: ctypes clicked '%s'", title)
                return True

    # 3. Keyboard — Alt+N is the standard Windows accelerator for "Next"
    logger.warning("_click_next: button not found by UIA/ctypes — trying Alt+N")
    pyautogui.hotkey('alt', 'n')
    time.sleep(0.4)

    # 4. Coordinate click: Next is always near bottom-right of wizard
    if hwnd:
        l, t, r, b = _win32_get_rect(hwnd)
        w, h = r - l, b - t
        # Next sits at roughly 82% width, 92% height of the dialog
        nx = int(l + w * 0.82)
        ny = int(t + h * 0.92)
        logger.warning("_click_next: coordinate fallback at (%d,%d)", nx, ny)
        pyautogui.click(nx, ny)
        return True

    return False


def _wizard_on_param_page(dlg) -> bool:
    """Return True if the wizard is showing the parameter grid (Page 2)."""
    hwnd = dlg.handle if hasattr(dlg, "handle") else None
    if not hwnd:
        return False
    uia = _uia_dlg(hwnd)
    if uia is None:
        return False
    try:
        items = list(uia.descendants(control_type="DataItem"))
        # Require at least 3 DataItems — Page 2 has one per parameter row.
        # Page 1 (optimization type selection) may have 0–2 DataItems from its
        # radio-button list, so a low count means we're still on Page 1.
        if len(items) >= 3:
            return True
        grids = list(uia.descendants(control_type="DataGrid"))
        return bool(grids)
    except Exception:
        return False


def _set_cell_value_at_coords(x: int, y: int, value: str) -> None:
    """Double-click a list-view cell at screen coords and type a value."""
    pyautogui.doubleClick(x, y)
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.05)
    _cb_paste(str(value))   # clipboard paste bypasses IME
    pyautogui.press("tab")
    time.sleep(0.2)


def _set_cell_value(wizard, lv, row: int, col: int, value: str) -> None:
    """Set a ListView cell value — try pywinauto rect first, fall back to coords."""
    try:
        item = lv.get_item(row, col)
        rect = item.rectangle()
        _set_cell_value_at_coords((rect.left + rect.right) // 2,
                                   (rect.top + rect.bottom) // 2, value)
    except Exception as e:
        logger.debug("Cell (%d,%d) set via pywinauto failed: %s", row, col, e)


def _read_listview_items_uia(wizard_hwnd: int) -> List[Tuple[int, str]]:
    """
    Return [(row_index, cell_text), ...] for visible text in the wizard's
    parameter grid using UIAutomation.  Works across UIPI privilege boundary.
    """
    rows: List[Tuple[int, str]] = []
    uia = _uia_dlg(wizard_hwnd)
    if uia is None:
        return rows
    for ct in ["DataGrid", "List", "Table"]:
        try:
            grid = uia.child_window(control_type=ct)
            if not grid.exists(timeout=2):
                continue
            for i, item in enumerate(grid.descendants(control_type="DataItem")):
                rows.append((i, item.window_text()))
            if rows:
                logger.debug("UIA wizard grid rows: %s", rows)
                return rows
        except Exception as e:
            logger.debug("UIA grid type %s: %s", ct, e)
    # Fallback: ListItem
    try:
        grid = uia.child_window(control_type="List")
        if grid.exists(timeout=2):
            for i, item in enumerate(grid.descendants(control_type="ListItem")):
                rows.append((i, item.window_text()))
    except Exception:
        pass
    logger.debug("UIA wizard grid rows (ListItem): %s", rows)
    return rows


def _get_listview_hwnd(parent_hwnd: int) -> Optional[int]:
    """Find a SysListView32 child via ctypes (bypasses UIPI process check)."""
    for cls in ["SysListView32", "ListView20WndClass"]:
        h = _win32_find_child(parent_hwnd, class_name=cls)
        if h:
            return h
    return None


def _estimate_cell_coords(
    lv_hwnd: int,
    row: int,
    col: int,
    num_cols: int = 7,
    row_height: int = 22,
    header_height: int = 26,
) -> Tuple[int, int]:
    """
    Estimate screen coordinates for a cell using the list-view rect.
    Used when SendMessage (LVM_GETITEMRECT) is UIPI-blocked.
    """
    l, t, r, b = _win32_get_rect(lv_hwnd)
    col_w = (r - l) // num_cols
    x = l + col_w * col + col_w // 2
    y = t + header_height + row_height * row + row_height // 2
    return x, y


def configure_optimization(
    conn: MultiChartsConnection,
    cfg: StrategyConfig,
    date_range: Optional[DateRange] = None,
) -> None:
    """
    Open Format Signals, select strategy, set date range, then launch
    the optimization wizard via right-click → Optimize Strategy.
    """
    import win32gui as _w32fg
    _cfg_t0 = time.time()
    def _cfg_lap(label: str) -> None:
        logger.info("[TIMING] cfg.%s=%.2fs  (total_so_far=%.2fs)",
                    label, time.time() - _cfg_lap._prev, time.time() - _cfg_t0)
        _cfg_lap._prev = time.time()
    _cfg_lap._prev = _cfg_t0

    # ── Step 0: Dismiss any stale Format Signals/Strategies dialog ────────────
    # A dialog left open from a previous run causes mis-clicks and wrong state.
    _fs_patterns_kw = ["Format Objects", "Format Signals", "Format Strategies",
                       "格式物件", "格式訊號", "格式策略"]
    for _sw in Desktop(backend="win32").windows():
        try:
            if not _sw.is_visible():
                continue
            _st = _sw.window_text()
            if any(kw in _st for kw in _fs_patterns_kw):
                logger.info("Dismissing stale Format dialog: '%s' hwnd=%x", _st, _sw.handle)
                # Strategy: press Escape (= Cancel, no save prompt) rather than
                # WM_CLOSE.  SendMessage(WM_CLOSE) deadlocks if MC64 shows a
                # "Save changes?" modal — it blocks MC64's message pump and
                # all subsequent GetWindowText calls hang indefinitely.
                # Escape closes the dialog cleanly without any modal prompt.
                try:
                    _user32.SetForegroundWindow(_sw.handle)
                    time.sleep(0.2)
                    pyautogui.press('escape')
                    time.sleep(0.4)
                    logger.info("  Dismissed via Escape key")
                except Exception as _esc_err:
                    logger.debug("  Escape failed: %s — falling back to PostMessage WM_CLOSE", _esc_err)
                    _user32.PostMessageW(_sw.handle, 0x0010, 0, 0)
                    time.sleep(0.6)
                    logger.info("  Dismissed via PostMessage WM_CLOSE")
        except Exception:
            pass

    _cfg_lap("step0_dismiss_stale")

    # ── Step 1+2: Open Format Signals only when date range needs updating ─────
    # When date range is already cached (every attempt after A01) skip opening
    # Format Signals entirely and jump straight to right-click → Optimize
    # Strategy.  Saves ~20 s per attempt.
    global _configure_date_range_cache
    _needs_date_set = (
        date_range is not None and
        _configure_date_range_cache != (date_range.from_date, date_range.to_date)
    )

    format_dlg = None
    if _needs_date_set or date_range is None:
        format_dlg = _open_format_signals(conn)
        _select_signal_in_list(format_dlg, cfg.mc_signal_name)
        time.sleep(0.2)

        try:
            _btn_names = []
            for _bh, _bc, _bt in _win32_enum_children(format_dlg.handle):
                if _bc == "Button" and _bt and _win32_is_visible(_bh):
                    _btn_names.append(_bt)
            logger.info("Format Signals dialog buttons: %s", _btn_names)
        except Exception as _be:
            logger.debug("Button enumeration failed: %s", _be)

        _cfg_lap("step1_open_format_signals")

        if _needs_date_set:
            # Snapshot existing windows so we can detect the NEW signal-properties dialog
            _pre_snap: set = set()
            for _sw in Desktop(backend="win32").windows():
                try:
                    if _sw.is_visible():
                        _pre_snap.add(_sw.handle)
                except Exception:
                    pass

            # Dump Format dialog UIA descendants for diagnostics
            _fmt_uia = _uia_dlg(format_dlg.handle)
            if _fmt_uia:
                try:
                    _fmt_desc = [(d.element_info.control_type,
                                  (d.element_info.name or "")[:25])
                                 for d in list(_fmt_uia.descendants())[:60]]
                    logger.info("Format dialog UIA descendants: %s", _fmt_desc)
                except Exception as _fde:
                    logger.debug("Format dialog UIA dump: %s", _fde)

            _fmt_btn_clicked = _click_button_in_dlg(
                format_dlg, ["Format", "格式", "Properties", "屬性", "編輯",
                             "Format Signal", "Edit", "Edit..."]
            )
            logger.info("Format/Properties button click: %s", _fmt_btn_clicked)

            # Fallback: double-click selected signal list item to open its properties
            if not _fmt_btn_clicked and _fmt_uia:
                try:
                    for _ct in ["List", "DataGrid", "Table", "ListBox"]:
                        _lv = _fmt_uia.child_window(control_type=_ct)
                        if _lv.exists(timeout=1):
                            _items = _lv.descendants(control_type="ListItem")
                            if not _items:
                                _items = _lv.descendants(control_type="DataItem")
                            if _items:
                                _items[0].double_click_input()
                                logger.info("Double-clicked signal item to open properties")
                                _fmt_btn_clicked = True
                                break
                except Exception as _dce:
                    logger.debug("Double-click signal failed: %s", _dce)

            time.sleep(0.3)

            # Wait for a NEW top-level window (not in pre-snapshot)
            signal_dlg = None
            _deadline = time.time() + 12.0
            while time.time() < _deadline and signal_dlg is None:
                for _sw in Desktop(backend="win32").windows():
                    try:
                        if _sw.is_visible() and _sw.handle not in _pre_snap:
                            _stitle = _sw.window_text()
                            logger.info("Signal properties dialog: hwnd=%x title='%s'",
                                        _sw.handle, _stitle)
                            signal_dlg = _HwndWrapper(_sw.handle)
                            break
                    except Exception:
                        pass
                if signal_dlg is None:
                    time.sleep(0.5)

            if signal_dlg is not None:
                _click_tab(signal_dlg, ["General", "一般"])
                time.sleep(0.3)
                _set_date_field(signal_dlg, conn.app,
                                ["Begin date:", "Begin Date", "開始日期", "起始日期"],
                                date_range.from_date)
                _set_date_field(signal_dlg, conn.app,
                                ["End date:", "End Date", "結束日期", "終止日期"],
                                date_range.to_date)
                _click_button_in_dlg(signal_dlg, ["OK", "確定"])
                time.sleep(0.2)
            else:
                logger.warning("Signal properties dialog not found — date range NOT set")

            # Re-open Format Signals and re-select after date change
            format_dlg = _open_format_signals(conn)
            _select_signal_in_list(format_dlg, cfg.mc_signal_name)
            time.sleep(0.2)
            _configure_date_range_cache = (date_range.from_date, date_range.to_date)
            logger.info("Date range cached: %s ~ %s", date_range.from_date, date_range.to_date)

        else:
            # date_range is None — no date filtering requested
            logger.info("[TIMING] step2: no date_range — skipped")

        _cfg_lap("step2_set_date_range")

    else:
        # date_range is already cached: skip step1+step2 entirely
        logger.info("[TIMING] step1+step2: SKIPPED — date cached (%s ~ %s)",
                    date_range.from_date, date_range.to_date)
        _cfg_lap("step1_open_format_signals")
        _cfg_lap("step2_set_date_range")

    # ── Step 3: Launch optimization wizard ───────────────────────────────────
    # Primary path A: click the "最佳化" / "Optimize" button INSIDE the
    # Format Signals dialog — this is the direct path in MC64 TW.
    _OPT_BTN_LABELS = [
        "最佳化", "Optimize", "優化",
        "最佳化...", "Optimize...", "優化...",
        "最佳化設定", "Optimization Settings",
    ]
    _opt_from_dlg = False
    if format_dlg is not None:
        _opt_from_dlg = _click_button_in_dlg(format_dlg, _OPT_BTN_LABELS)
        logger.info("Optimize button in Format Signals dialog: clicked=%s", _opt_from_dlg)

    if not _opt_from_dlg:
        if format_dlg is not None:
            # Close the Format Signals dialog before right-clicking
            logger.info("No Optimize button in dialog — closing and using right-click approach")
            _closed = _click_button_in_dlg(
                format_dlg, ["Cancel", "Close", "關閉", "取消", "結束", "離開"]
            )
            if not _closed:
                logger.warning("Close button not found — sending WM_CLOSE to Format dialog")
                _user32.PostMessageW(format_dlg.handle, 0x0010, 0, 0)
            time.sleep(0.25)
        else:
            logger.info("Direct right-click path (Format Signals skipped)")


        _focus_window(conn._hwnd)
        time.sleep(0.2)

        # Find ATL_MCGraphPanel (chart area) for right-click
        _all_ch = _win32_enum_children(conn._hwnd)
        _gp_hwnd = None
        for _h, _cls, _ in _all_ch:
            if _cls == "ATL_MCGraphPanel" and _win32_is_visible(_h):
                _gp_hwnd = _h
                break
        if _gp_hwnd is None:
            for _h, _cls, _ in _all_ch:
                if _cls == "ATL_MCMDIChildFrame" and _win32_is_visible(_h):
                    _gp_hwnd = _h
                    break

        if _gp_hwnd:
            _gpr = _win32_get_rect(_gp_hwnd)
            _gh = _gpr[3] - _gpr[1]
            # Try right-edge x (future blank area) before center x, and scan y
            # from near the top (blank space above price range) rather than 20% down.
            _gy_start = _gpr[1] + 8
            _gpx, _gpy = (_gpr[0] + _gpr[2]) // 2, _gy_start
            _gp_found = False
            for _try_gx in [_gpr[2] - 35, (_gpr[0] + _gpr[2]) // 2]:
                if _gp_found:
                    break
                for _dy in range(0, _gh - 10, 5):
                    _ty = _gy_start + _dy
                    try:
                        if _w32fg.WindowFromPoint((_try_gx, _ty)) == _gp_hwnd:
                            _gpx, _gpy = _try_gx, _ty
                            _gp_found = True
                            break
                    except Exception:
                        break
            logger.info("Right-click for Optimize Strategy at (%d,%d)", _gpx, _gpy)
            pyautogui.click(_gpx, _gpy)
            time.sleep(0.2)
            pyautogui.rightClick(_gpx, _gpy)
            time.sleep(0.35)
            _opt_popup = _find_popup_hwnd(timeout=1.0)
            # PostMessage fallback — ATL_MCGraphPanel is transparent to WindowFromPoint
            if not _opt_popup and _gp_hwnd:
                try:
                    _WM_RBUTTONDOWN = 0x0204
                    _WM_RBUTTONUP   = 0x0205
                    _WM_CONTEXTMENU = 0x007B
                    _MK_RBUTTON     = 0x0002
                    _opt_u32 = ctypes.windll.user32
                    _op_pcr  = _win32_get_rect(_gp_hwnd)
                    _op_pw = _op_pcr[2] - _op_pcr[0]
                    _op_ph = _op_pcr[3] - _op_pcr[1]
                    # Use right-edge / vertical-center (same as _open_format_signals)
                    # to avoid price-bar area that may show a different context menu.
                    _op_cli_x = max(10, min(_op_pw - 10, _op_pw - 35))
                    _op_cli_y = max(10, min(_op_ph - 10, _op_ph // 2))
                    _op_cli_lp = (_op_cli_y << 16) | (_op_cli_x & 0xFFFF)
                    _op_scr_x  = _op_pcr[0] + _op_cli_x
                    _op_scr_y  = _op_pcr[1] + _op_cli_y
                    _op_scr_lp = (_op_scr_y << 16) | (_op_scr_x & 0xFFFF)
                    logger.info("PostMessage Optimize right-click: hwnd=0x%x client=(%d,%d) screen=(%d,%d)",
                                _gp_hwnd, _op_cli_x, _op_cli_y, _op_scr_x, _op_scr_y)
                    _opt_u32.SetForegroundWindow(_gp_hwnd)
                    time.sleep(0.15)
                    _opt_u32.PostMessageW(_gp_hwnd, _WM_RBUTTONDOWN, _MK_RBUTTON, _op_cli_lp)
                    time.sleep(0.1)
                    _opt_u32.PostMessageW(_gp_hwnd, _WM_RBUTTONUP, 0, _op_cli_lp)
                    time.sleep(0.35)
                    _opt_popup = _find_popup_hwnd(timeout=1.0)
                    if _opt_popup:
                        logger.info("PostMessage WM_RBUTTON* → Optimize popup 0x%x", _opt_popup)
                    else:
                        _opt_u32.PostMessageW(_gp_hwnd, _WM_CONTEXTMENU, _gp_hwnd, _op_scr_lp)
                        time.sleep(0.35)
                        _opt_popup = _find_popup_hwnd(timeout=1.0)
                        logger.info("PostMessage WM_CONTEXTMENU → Optimize popup %s",
                                    hex(_opt_popup) if _opt_popup else None)
                except Exception as _op_pme:
                    logger.warning("PostMessage Optimize right-click failed: %s", _op_pme)
            if _opt_popup:
                _clicked = _pyautogui_click_popup_item(conn.app, [
                    "Optimize Strategy", "最佳化策略", "Optimize...",
                    "最佳化...", "優化策略",
                ])
                logger.info("Optimize Strategy context-menu click: %s", _clicked)
            else:
                logger.warning("No context menu after right-click")
        else:
            logger.warning("No chart child found for right-click optimization launch")

    time.sleep(0.3)
    _cfg_lap("step3_launch_wizard")

    # ── Step 4: Wait for optimization wizard ─────────────────────────────────
    # MC Traditional Chinese wizard title is "最佳化設定"; English is "Optimization".
    wizard = _wait_for_any_window(
        conn.app,
        [r".*[Oo]ptimiz.*", r".*最佳化.*", r".*優化.*", r".*Optimis.*"],
        timeout=30
    )
    try:
        _wiz_buf = ctypes.create_unicode_buffer(256)
        _user32.GetWindowTextW(wizard.handle, _wiz_buf, 256)
        logger.info("Optimization wizard opened: hwnd=%x title='%s'", wizard.handle, _wiz_buf.value)
    except Exception:
        logger.info("Optimization wizard opened")

    # Log wizard structure so we know what we're working with
    try:
        _wiz_ch = _win32_enum_children(wizard.handle)
        _wiz_btns = [(hex(h), t) for h, c, t in _wiz_ch if c == "Button" and t and _win32_is_visible(h)]
        logger.info("Wizard buttons: %s", _wiz_btns)
        logger.info("Wizard children (first 20): %s",
                    [(hex(h), c, t[:25]) for h, c, t in _wiz_ch[:20]])
    except Exception as _we:
        logger.debug("Wizard enumeration failed: %s", _we)

    # Page 1 (if present): choose Regular Optimization then Next.
    # MC may open the wizard directly on the parameter page — in that case
    # _wizard_on_param_page() returns True immediately and we skip this block.
    _early_dpi = _get_dpi_scale(wizard.handle)
    if not _wizard_on_param_page(wizard):
        logger.info("Wizard on Page 1 (type selection) — clicking Regular Optimization + Next")
        _click_button_in_dlg(wizard, ["Regular Optimization", "標準優化", "一般優化", "最佳化", "Optimization"])
        time.sleep(0.4)
        _click_next_in_wizard(wizard, _early_dpi)
        time.sleep(1.0)
        # Verify we advanced to Page 2; retry once if not
        if not _wizard_on_param_page(wizard):
            logger.warning("Still on Page 1 after first Next click — retrying")
            _click_next_in_wizard(wizard, _early_dpi)
            time.sleep(1.0)
            if not _wizard_on_param_page(wizard):
                logger.warning("Still on Page 1 after second Next click — proceeding anyway")
    else:
        logger.info("Wizard opened directly on parameter page (Page 1 skipped)")

    _cfg_lap("step4_wizard_appeared")

    # Page 2 — configure parameters via UIA
    # MC64 optimization wizard uses custom-drawn controls (no Win32 children).
    # UIA is the only programmatic path.
    time.sleep(0.5)

    # Diagnostic screenshot so we can see the wizard state
    try:
        _wss_path = str(
            Path(__file__).parent.parent / "results" /
            f"WIZARD_{cfg.name}_{int(time.time())}.png"
        )
        pyautogui.screenshot(_wss_path)
        logger.info("Wizard page2 screenshot: %s", _wss_path)
    except Exception as _sse:
        logger.debug("Screenshot failed: %s", _sse)

    _wiz_uia = _uia_dlg(wizard.handle)

    if _wiz_uia:
        # Dump ALL UIA descendants for diagnostics
        try:
            _all_uia = list(_wiz_uia.descendants())
            _uia_log = [(d.element_info.control_type,
                         (d.element_info.name or "")[:25],
                         (d.element_info.class_name or "")[:12])
                        for d in _all_uia[:100]]
            logger.info("Wizard UIA page2 descendants (%d): %s", len(_all_uia), _uia_log)
        except Exception as _de:
            logger.debug("UIA desc dump: %s", _de)

        # Detect DPI scale so pyautogui (logical pixels) matches UIA (physical pixels).
        import ctypes as _ctypes
        _dpi_scale = 1.0
        try:
            _win_dpi = _ctypes.windll.user32.GetDpiForWindow(wizard.handle)
            if _win_dpi > 0:
                _dpi_scale = _win_dpi / 96.0
        except Exception:
            pass
        logger.info("Wizard DPI scale: %.3f", _dpi_scale)

        # Focus wizard before any interaction
        try:
            _wiz_uia.set_focus()
            time.sleep(0.2)
        except Exception:
            pass

        # Log all CheckBox bounds so we can verify coordinates in the log
        _wiz_cbs_all = list(_wiz_uia.descendants(control_type="CheckBox"))
        logger.info("Wizard CheckBox count: %d", len(_wiz_cbs_all))
        for _dbi, _dbc in enumerate(_wiz_cbs_all[:8]):
            try:
                _r = _dbc.rectangle()
                logger.info("  CB[%d] rect=(L%d T%d R%d B%d) logical_center=(%d,%d)",
                            _dbi, _r.left, _r.top, _r.right, _r.bottom,
                            int((_r.left + _r.right) / 2 / _dpi_scale),
                            int((_r.top + _r.bottom) / 2 / _dpi_scale))
            except Exception:
                pass

        # Step A: check the Optimize checkbox for each param via UIA invoke().
        # invoke() is more reliable than pyautogui.click for WPF CheckBox in DataGrid.
        # Edit field layout (confirmed from logs): each row has 7 Edit descendants:
        #   [0]=Current Value (read-only)
        #   [1]=[2]=Start Value (two UIA refs for same WPF TextBox — use [2])
        #   [3]=[4]=End Value   (two UIA refs for same WPF TextBox — use [3])
        #   [5]=[6]=Step Value  (two UIA refs for same WPF TextBox — use [5])
        # Tab navigation: click at Start position → Tab → End → Tab → Step → Tab commit
        _all_items = list(_wiz_uia.descendants(control_type="DataItem"))
        logger.info("Wizard DataGrid: %d rows (total)", len(_all_items))

        # Step 0: uncheck ALL rows first — clears stale settings from previous
        # sessions that would leave extra params checked and inflate combo count.
        _step5_t0 = time.time()
        for _item0 in _all_items:
            try:
                _txts0 = [d.window_text() for d in _item0.descendants(control_type="Text")]
                _row_nm0 = next((t for t in _txts0 if t), "?")
                _cbs0 = list(_item0.descendants(control_type="CheckBox"))
                if not _cbs0:
                    continue
                _cb0 = _cbs0[0]
                _chk0 = False
                try:
                    _chk0 = (_cb0.iface_toggle.CurrentToggleState == 1)
                except Exception:
                    try:
                        _chk0 = _cb0.is_checked()
                    except Exception:
                        pass
                if _chk0:
                    try:
                        # invoke() always fails for this WPF wizard — use click_input directly
                        _cb0.click_input()
                        time.sleep(0.2)
                        logger.info("  Step0 unchecked row: '%s'", _row_nm0)
                    except Exception as _ue0:
                        try:
                            _cbr0 = _cb0.rectangle()
                            _cbx0 = int((_cbr0.left + _cbr0.right) / 2 / _dpi_scale)
                            _cby0 = int((_cbr0.top + _cbr0.bottom) / 2 / _dpi_scale)
                            pyautogui.click(_cbx0, _cby0)
                            time.sleep(0.25)
                            logger.info("  Step0 pyautogui unchecked row: '%s'", _row_nm0)
                        except Exception as _fe0:
                            logger.warning("  Step0 fallback failed for '%s': %s", _row_nm0, _fe0)
            except Exception:
                pass

        logger.info("[TIMING] cfg.step5a_uncheck_all=%.2fs", time.time() - _step5_t0)

        for _pi, _param in enumerate(cfg.params):
            _param_t0 = time.time()
            # Find the row containing this param's input name
            _target_row = None
            _all_items_cur = list(_wiz_uia.descendants(control_type="DataItem"))
            for _item in _all_items_cur:
                try:
                    _txts = [d.window_text() for d in _item.descendants(control_type="Text")]
                    if _param.name in _txts and any(t == cfg.mc_signal_name for t in _txts):
                        _target_row = _item
                        break
                except Exception:
                    pass

            if _target_row is None:
                # Positional fallback is only safe on single-signal charts where the
                # grid has exactly len(cfg.params) rows.  With multiple signals
                # enabled, row[_pi] would be some OTHER signal's input — skip instead.
                if len(_all_items_cur) == len(cfg.params) and _pi < len(_all_items_cur):
                    _target_row = _all_items_cur[_pi]
                    logger.warning("  Using positional row[%d] for param '%s'", _pi, _param.name)
                else:
                    logger.warning("  No DataItem row found for param '%s' "
                                   "(grid rows=%d, cfg params=%d) — skipping; "
                                   "check ParamAxis.name and mc_signal_name match MC exactly",
                                   _param.name, len(_all_items_cur), len(cfg.params))
                    continue

            # Get the checkbox within this row
            _row_cbs = list(_target_row.descendants(control_type="CheckBox"))
            if not _row_cbs:
                logger.warning("  No CheckBox in row for param '%s'", _param.name)
                continue

            _row_cb = _row_cbs[0]
            try:
                _cbr = _row_cb.rectangle()
                _cbx = int((_cbr.left + _cbr.right) / 2 / _dpi_scale)
                _cby = int((_cbr.top + _cbr.bottom) / 2 / _dpi_scale)

                # Helper: read toggle state → True=checked, False=unchecked, None=unknown
                def _cb_state(cb):
                    try:
                        return cb.iface_toggle.CurrentToggleState == 1
                    except Exception:
                        pass
                    try:
                        return cb.is_checked()
                    except Exception:
                        return None

                _cur = _cb_state(_row_cb)
                logger.info("  Param '%s': checkbox physical(%d,%d) logical(%d,%d) state=%s",
                            _param.name,
                            (_cbr.left + _cbr.right) // 2, (_cbr.top + _cbr.bottom) // 2,
                            _cbx, _cby, _cur)

                # MUST be checked for Start/End/Step fields to become editable.
                # Only skip if we are CERTAIN it is already checked (True).
                if _cur is not True:
                    _ck_ok = False
                    for _ck_try in range(3):
                        if _ck_try == 0:
                            # Attempt 1: UIA invoke() — fires WPF Click event directly
                            try:
                                _row_cb.invoke()
                                time.sleep(0.2)
                                logger.info("  Param '%s': invoke() to check (try 1)", _param.name)
                            except Exception as _ie:
                                logger.debug("  invoke() failed: %s", _ie)
                                # Sub-fallback: click_input() sends mouse event to the
                                # CheckBox control directly (bypasses DataGrid row selection)
                                try:
                                    _row_cb.click_input()
                                    time.sleep(0.25)
                                    logger.info("  Param '%s': click_input() to check", _param.name)
                                except Exception:
                                    pass
                        else:
                            # Attempt 2/3: click row body first (selects the row WITHOUT
                            # toggling checkbox), then single-click on checkbox area.
                            # Using row_body_x = _cbx + 40 = parameter-name column area.
                            _body_x = min(_cbx + 40, int(_cbr.right / _dpi_scale) + 60)
                            pyautogui.click(_body_x, _cby)
                            time.sleep(0.2)
                            pyautogui.click(_cbx, _cby)
                            time.sleep(0.3)
                            logger.info("  Param '%s': row+checkbox click (try %d)",
                                        _param.name, _ck_try + 1)

                        # Verify state after this attempt
                        _new = _cb_state(_row_cb)
                        if _new is True:
                            logger.info("  Param '%s': checkbox confirmed CHECKED", _param.name)
                            _ck_ok = True
                            break
                        elif _new is False:
                            logger.warning("  Param '%s': still unchecked after try %d",
                                           _param.name, _ck_try + 1)
                        else:
                            # State unknown — assume success and proceed
                            logger.info("  Param '%s': checkbox state unknown → assuming checked",
                                        _param.name)
                            _ck_ok = True
                            break

                    if not _ck_ok:
                        logger.warning("  Param '%s': checkbox may NOT be checked — "
                                       "Start/End/Step fields may be read-only!", _param.name)

                # Wait for edit fields to become enabled after checkbox is checked
                time.sleep(0.3)

            except Exception as _ce:
                logger.warning("  Param '%s' checkbox step failed: %s", _param.name, _ce)

            # Step B: set Start/End/Step via Tab navigation.
            # Edit field layout (7 UIA Edit per row, confirmed from logs):
            #   [0]=Current Value  [1]=[2]=Start  [3]=[4]=End  [5]=[6]=Step
            # Duplicates share the same on-screen rect; Tab correctly navigates
            # Start→End→Step regardless of UIA index duplication.
            # Strategy: click the Start field by coordinate, then Tab+typewrite to End and Step.
            try:
                # Poll until at least 3 Edit fields are visible (row expanded)
                _row_edits = []
                for _exp_try in range(7):   # 7 × 0.25 s = 1.75 s max
                    _all_items_b = list(_wiz_uia.descendants(control_type="DataItem"))
                    for _ib in _all_items_b:
                        try:
                            _txts_b = [d.window_text() for d in _ib.descendants(control_type="Text")]
                            if _param.name in _txts_b and any(t == cfg.mc_signal_name for t in _txts_b):
                                _target_row = _ib
                                break
                        except Exception:
                            pass
                    _row_edits = list(_target_row.descendants(control_type="Edit"))
                    if len(_row_edits) >= 3:
                        break
                    logger.debug("  Param '%s': only %d edit(s) — waiting for expansion (try %d)",
                                 _param.name, len(_row_edits), _exp_try + 1)
                    time.sleep(0.25)

                # Log edit field positions for diagnosis
                _edit_rects = []
                for _re_el in _row_edits[:8]:
                    try:
                        _edit_rects.append(_re_el.rectangle())
                    except Exception:
                        _edit_rects.append(None)

                logger.info("  Param '%s': %d Edit(s) after expansion",
                            _param.name, len(_row_edits))
                for _ri, _rr in enumerate(_edit_rects):
                    if _rr:
                        logger.info("    edit[%d] L%d T%d R%d B%d logical=(%d,%d)",
                                    _ri, _rr.left, _rr.top, _rr.right, _rr.bottom,
                                    int((_rr.left + _rr.right) / 2 / _dpi_scale),
                                    int((_rr.top + _rr.bottom) / 2 / _dpi_scale))

                # Determine Start field index:
                # Confirmed layout (4 distinct fields): [0]=Current, [1]=Start, [2]=End, [3]=Step
                if len(_row_edits) >= 4 and _edit_rects[1] is not None:
                    _ei_start = 1
                elif len(_row_edits) >= 2 and _edit_rects[1] is not None:
                    _ei_start = 1
                    logger.warning("  Param '%s': only %d edits — using _ei_start=1",
                                   _param.name, len(_row_edits))
                else:
                    logger.warning("  Param '%s': too few edits (%d) — skipping param",
                                   _param.name, len(_row_edits))
                    _ei_start = -1

                if _ei_start >= 0:
                    _start_rr = _edit_rects[_ei_start]
                    _sx = int((_start_rr.left + _start_rr.right) / 2 / _dpi_scale)
                    _sy = int((_start_rr.top + _start_rr.bottom) / 2 / _dpi_scale)

                    # Click Start field, then Tab to End, Tab to Step.
                    # pyautogui.typewrite is reliable for WPF TextBox (confirmed from logs).
                    def _set_field(val_str: str, label: str) -> None:
                        pyautogui.hotkey('ctrl', 'a')
                        time.sleep(0.08)
                        _cb_paste(val_str)   # clipboard paste bypasses IME
                        time.sleep(0.08)
                        logger.info("  Set '%s'.%s = %s", _param.name, label, val_str)

                    try:
                        # Focus Start field
                        pyautogui.click(_sx, _sy)
                        time.sleep(0.15)
                        _set_field(str(_param.start), "Start")
                        pyautogui.press('tab');  time.sleep(0.12)  # → End
                        _set_field(str(_param.stop),  "End")
                        pyautogui.press('tab');  time.sleep(0.12)  # → Step
                        _set_field(str(_param.step),  "Step")
                        pyautogui.press('tab');  time.sleep(0.10)  # commit
                    except Exception as _ee:
                        logger.warning("  '%s' Start/End/Step entry failed: %s",
                                       _param.name, _ee)
            except Exception as _re:
                logger.warning("  Row edit pass for '%s' failed: %s", _param.name, _re)
            logger.info("[TIMING] cfg.step5b_param[%d]_%s=%.2fs",
                        _pi, _param.name, time.time() - _param_t0)
        _cfg_lap("step5_configure_params")
    else:
        logger.warning("Wizard UIA wrapper not available — param config skipped")

    # Select Exhaustive method and click Optimize via UIA.
    # The wizard is a WPF single-page dialog; Win32 button enumeration always returns [].
    # All controls must be reached via the UIA wrapper acquired earlier.
    _wiz_uia2 = _uia_dlg(wizard.handle)
    if _wiz_uia2:
        # Log all UIA Button controls for diagnosis
        _all_uia_btns = list(_wiz_uia2.descendants(control_type="Button"))
        logger.info("Wizard UIA Button list (%d): %s",
                    len(_all_uia_btns),
                    [((b.window_text() or "")[:20], b.rectangle()) for b in _all_uia_btns])

        # Select Exhaustive.
        # Layout: CB[0]=header select-all, CB[1..N]=one per DataGrid row, CB[N+1]=Exhaustive,
        # where N = the TOTAL DataGrid row count (all inputs of all ENABLED signals),
        # which equals len(cfg.params) only on single-signal charts.
        # Priority: (1) name match, (2) actual row count + 1, (3) len(cfg.params)+1.
        try:
            _ecbs = list(_wiz_uia2.descendants(control_type="CheckBox"))
            _exh_cb = None
            _exh_how = None
            # Path 1: locate by checkbox text
            for _ec in _ecbs:
                _ect = (_ec.window_text() or "").strip().lower()
                if _ect and ("exhaustive" in _ect or "窮舉" in _ect or "暴力" in _ect):
                    _exh_cb = _ec
                    _exh_how = f"name-match '{_ect[:20]}'"
                    break
            # Path 2: actual DataGrid row count + 1
            if _exh_cb is None:
                _n_rows = len(list(_wiz_uia2.descendants(control_type="DataItem")))
                _exhaustive_idx = _n_rows + 1
                if 0 < _exhaustive_idx < len(_ecbs):
                    _exh_cb = _ecbs[_exhaustive_idx]
                    _exh_how = f"row-count idx={_exhaustive_idx} (rows={_n_rows})"
            # Path 3: legacy len(cfg.params)+1 (single-signal charts only)
            if _exh_cb is None:
                _exhaustive_idx = len(cfg.params) + 1
                if len(_ecbs) > _exhaustive_idx:
                    _exh_cb = _ecbs[_exhaustive_idx]
                    _exh_how = f"legacy idx={_exhaustive_idx}"
            if _exh_cb is not None:
                _ecbr = _exh_cb.rectangle()
                _ecbx = int((_ecbr.left + _ecbr.right) / 2 / _dpi_scale)
                _ecby = int((_ecbr.top + _ecbr.bottom) / 2 / _dpi_scale)
                logger.info("Clicking Exhaustive option via %s at (%d,%d)", _exh_how, _ecbx, _ecby)
                pyautogui.click(_ecbx, _ecby)
                time.sleep(0.4)
            else:
                logger.warning("Exhaustive checkbox not located (CBs=%d)", len(_ecbs))
        except Exception as _exhe:
            logger.warning("Exhaustive option click failed: %s", _exhe)

        # Click the Optimize (start) button.
        # MC64 Traditional Chinese label is "最佳化"; Simplified Chinese is "優化".
        # WPF buttons respond more reliably to invoke() than to coordinate clicks.
        _opt_clicked = False
        _opt_kws = [
            "最佳化",          # Traditional Chinese MC64 ← most common for TW users
            "optimize",        # English
            "優化",            # Simplified Chinese
            "開始最佳化",
            "開始優化",
            "start optim",
            "apply",           # MC64 EN wizard: "Apply" button starts optimization
            "執行",            # Execute
            "開始",            # Start
            "run",
            "ok",
            "確定",
        ]
        _skip_kws = ["close", "關閉", "cancel", "取消", "next", "下一步",
                     "back", "上一步", "finish", "完成", "previous", "< previous"]

        for _ub in _all_uia_btns:
            _ubt = (_ub.window_text() or "").strip()
            _ubt_low = _ubt.lower()
            if any(kw.lower() in _ubt_low for kw in _skip_kws):
                continue
            if any(kw.lower() in _ubt_low for kw in _opt_kws):
                try:
                    _ubr = _ub.rectangle()
                    _ubx = int((_ubr.left + _ubr.right) / 2 / _dpi_scale)
                    _uby = int((_ubr.top + _ubr.bottom) / 2 / _dpi_scale)
                    logger.info("Clicking Optimize button '%s' at logical(%d,%d)",
                                _ubt[:25], _ubx, _uby)
                    try:
                        _ub.invoke()            # WPF-safe: fires Click event directly
                    except Exception:
                        pyautogui.click(_ubx, _uby)
                    _opt_clicked = True
                    break
                except Exception as _oe:
                    logger.warning("Optimize button '%s' click failed: %s", _ubt[:20], _oe)

        if not _opt_clicked:
            # Fallback A: bottom-most button that isn't a dismiss/nav button
            _candidates = []
            for _b in _all_uia_btns:
                _bt = (_b.window_text() or "").strip().lower()
                if not any(kw in _bt for kw in _skip_kws):
                    _candidates.append(_b)
            if _candidates:
                _ub = max(_candidates, key=lambda b: b.rectangle().top)
                _ubt = (_ub.window_text() or "?")[:25]
                _ubr = _ub.rectangle()
                _ubx = int((_ubr.left + _ubr.right) / 2 / _dpi_scale)
                _uby = int((_ubr.top + _ubr.bottom) / 2 / _dpi_scale)
                logger.warning("Fallback A: bottom-most button '%s' at logical(%d,%d)",
                               _ubt, _ubx, _uby)
                try:
                    _ub.invoke()
                except Exception:
                    pyautogui.click(_ubx, _uby)
                _opt_clicked = True

        if not _opt_clicked:
            # Fallback B: coordinate click at bottom-right of wizard (Optimize is typically there)
            _wiz_l, _wiz_t, _wiz_r, _wiz_b = _win32_get_rect(wizard.handle)
            _btn_x = int(_wiz_l + (_wiz_r - _wiz_l) * 0.72)
            _btn_y = int(_wiz_t + (_wiz_b - _wiz_t) * 0.93)
            logger.warning("Fallback B: coordinate click for Optimize at (%d,%d)", _btn_x, _btn_y)
            pyautogui.click(_btn_x, _btn_y)
            _opt_clicked = True

        if not _opt_clicked:
            # Fallback C: Enter key (default button in most dialogs)
            logger.warning("Fallback C: pressing Enter to trigger default Optimize action")
            pyautogui.press('enter')
    else:
        # Win32 fallback (unlikely to work but try anyway)
        _click_button_in_dlg(wizard, ["Optimize", "Finish", "完成", "優化"])

    _cfg_lap("step6_exhaustive_and_start")
    logger.info("[TIMING] cfg.TOTAL=%.2fs", time.time() - _cfg_t0)
    logger.info("Optimization started: %s", cfg.name)


# ---------------------------------------------------------------------------
# Set fixed params for single backtest (OOS validation)
# ---------------------------------------------------------------------------

def set_params_and_date_for_single_run(
    conn: MultiChartsConnection,
    cfg: StrategyConfig,
    params: Dict[str, float],
    date_range: DateRange,
) -> None:
    """
    Set specific (non-optimized) parameter values + date range for OOS backtest.
    """
    format_dlg = _open_format_signals(conn)
    _select_signal_in_list(format_dlg, cfg.mc_signal_name)
    time.sleep(0.2)

    _click_button_in_dlg(format_dlg, ["Format", "格式", "編輯"])
    time.sleep(0.8)

    signal_dlg = _wait_for_any_window(
        conn.app, [r".*Format.*", r".*Signal.*", r".*格式.*"], timeout=10
    )

    # --- Inputs tab: set fixed values ---
    _click_tab(signal_dlg, ["Inputs", "輸入"])
    time.sleep(0.3)

    sig_hwnd = signal_dlg.handle
    lv_hwnd = _get_listview_hwnd(sig_hwnd)

    # Try UIA to read param names and set values
    inputs_set = False
    uia = _uia_dlg(sig_hwnd)
    if uia is not None:
        try:
            for ct in ["DataGrid", "List", "Table"]:
                grid = uia.child_window(control_type=ct)
                if not grid.exists(timeout=2):
                    continue
                for item in grid.descendants(control_type="DataItem"):
                    cell_text = item.window_text().strip()
                    if cell_text in params:
                        item.click_input()
                        time.sleep(0.15)
                        _pg_hotkey("ctrl", "a")
                        time.sleep(0.05)
                        _cb_paste(str(params[cell_text]))   # clipboard paste bypasses IME
                        pyautogui.press("tab")
                        time.sleep(0.1)
                        inputs_set = True
                break
        except Exception as e:
            logger.debug("UIA inputs set failed: %s", e)

    if not inputs_set and lv_hwnd:
        # Ctypes + estimated coords — use cfg.params ordering
        for row_idx, (pname, pval) in enumerate(params.items()):
            # Click at value column (col 1)
            cx, cy = _estimate_cell_coords(lv_hwnd, row_idx, col=1)
            _set_cell_value_at_coords(cx, cy, str(pval))
        inputs_set = True

    if not inputs_set:
        # Last resort: pywinauto (may fail for admin windows)
        try:
            lv = signal_dlg.child_window(class_name="SysListView32")
            count = lv.item_count()
            for i in range(count):
                for col in range(0, 3):
                    try:
                        cell_text = lv.get_item(i, col).text().strip()
                        if cell_text in params:
                            val_col = col + 1 if col < 2 else col
                            _set_cell_value(signal_dlg, lv, i, val_col,
                                            str(params[cell_text]))
                            break
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("Could not set inputs: %s", e)

    # --- General tab: set date range ---
    _click_tab(signal_dlg, ["General", "一般"])
    time.sleep(0.3)

    _set_date_field(signal_dlg, conn.app,
                    ["Begin date:", "Begin Date", "開始日期", "起始日期"],
                    date_range.from_date)
    _set_date_field(signal_dlg, conn.app,
                    ["End date:", "End Date", "結束日期", "終止日期"],
                    date_range.to_date)

    _click_button_in_dlg(signal_dlg, ["OK", "確定"])
    time.sleep(1)
    logger.info("Params+dates set for OOS: %s / %s~%s",
                params, date_range.from_date, date_range.to_date)


def read_performance_report(conn: MultiChartsConnection) -> Optional[Dict[str, float]]:
    """
    Read NetProfit and MaxDrawdown from the Strategy Performance Report window.
    Returns dict with 'NetProfit', 'MaxDrawdown', or None if not found.
    """
    time.sleep(2)  # wait for recalculation

    rpt_wnd = None
    for title_re in [r".*Strategy Performance Report.*", r".*策略績效報告.*",
                     r".*Performance Report.*", r".*績效報告.*"]:
        try:
            w = conn.app.window(title_re=title_re)
            if w.exists(timeout=2):
                rpt_wnd = w
                break
        except Exception:
            pass

    if rpt_wnd is None:
        try:
            rpt_wnd = conn.main_window().child_window(
                title_re=r".*[Pp]erformance.*[Rr]eport.*"
            )
            if not rpt_wnd.exists(timeout=2):
                rpt_wnd = None
        except Exception:
            rpt_wnd = None

    if rpt_wnd is None:
        logger.warning("Strategy Performance Report window not found")
        return None

    result = {}
    try:
        text = rpt_wnd.window_text() + " " + " ".join(
            c.window_text() for c in rpt_wnd.children()
        )

        import re
        # Match "Net Profit" followed by a number (possibly with commas/sign)
        for pattern, key in [
            (r"Net Profit[^\d\-]*?([-+]?[\d,]+\.?\d*)", "NetProfit"),
            (r"淨利[^\d\-]*?([-+]?[\d,]+\.?\d*)", "NetProfit"),
            (r"Max(?:imum)?\s*(?:Strategy\s*)?Drawdown[^\d\-]*?([-+]?[\d,]+\.?\d*)", "MaxDrawdown"),
            (r"最大策略虧損[^\d\-]*?([-+]?[\d,]+\.?\d*)", "MaxDrawdown"),
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val_str = m.group(1).replace(",", "")
                try:
                    result[key] = float(val_str)
                except ValueError:
                    pass
    except Exception as e:
        logger.warning("Could not parse Performance Report: %s", e)

    return result if result else None


def run_oos_backtest_for_candidate(
    conn: MultiChartsConnection,
    cfg: StrategyConfig,
    params: Dict[str, float],
) -> Optional[Dict[str, float]]:
    """
    Set params + OOS date range, wait for recalc, return performance metrics.
    """
    try:
        ensure_chart_ready(conn, cfg)
        set_params_and_date_for_single_run(conn, cfg, params, cfg.outsample)
        return read_performance_report(conn)
    except Exception as e:
        logger.error("OOS backtest failed for %s params=%s: %s", cfg.name, params, e)
        return None


# ---------------------------------------------------------------------------
# Wait for optimization complete
# ---------------------------------------------------------------------------

def wait_for_optimization_complete(
    conn: MultiChartsConnection,
    timeout: int = OPTIMIZATION_TIMEOUT_SECONDS,
    poll_interval: int = POLL_INTERVAL_SECONDS,
    opt_start_time: float = 0.0,
) -> bool:
    """Wait until optimization finishes.

    Detection methods (first match wins):
    1. A new .MCReport file appears in Documents (most reliable).
    2. An "Optimization Report" top-level or MDI-child window appears.
    """
    import glob as _glob

    _docs = os.path.join(os.path.expanduser("~"), "Documents")
    _t0 = opt_start_time or time.time()

    logger.info("Waiting for optimization (timeout=%ds)...", timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Method 0: fast MCReport check BEFORE anything else so a quick
        # optimization (< poll_interval) is caught on the very first iteration.
        for _f in _glob.glob(os.path.join(_docs, "*.MCReport")):
            try:
                if os.path.getmtime(_f) >= _t0 - 2:
                    logger.info("Optimization complete — new MCReport: %s", os.path.basename(_f))
                    return True
            except OSError:
                pass

        try:
            _dismiss_error_dialogs(conn.app)
        except BaseException as _ded_err:
            logger.warning("_dismiss_error_dialogs raised %s: %s — continuing poll",
                           type(_ded_err).__name__, _ded_err)

        # Method 1: new .MCReport in Documents (second pass after dismiss)
        for _f in _glob.glob(os.path.join(_docs, "*.MCReport")):
            try:
                if os.path.getmtime(_f) >= _t0 - 2:
                    logger.info("Optimization complete — new MCReport: %s", os.path.basename(_f))
                    return True
            except OSError:
                pass

        # Method 2: window title search
        for title_re in [r".*Optimization Report.*", r".*優化報告.*", r".*優化結果.*",
                         r".*ORVisualizer.*"]:
            try:
                rpt = conn.app.window(title_re=title_re)
                if rpt.exists(timeout=0):
                    logger.info("Optimization complete (window: %s).", title_re)
                    return True
            except Exception:
                pass
        try:
            rpt = conn.main_window().child_window(
                title_re=r".*[Oo]ptimization.*[Rr]eport.*"
            )
            if rpt.exists(timeout=0):
                logger.info("Optimization complete (MDI child).")
                return True
        except Exception:
            pass

        elapsed = timeout - (deadline - time.time())
        if int(elapsed) % 60 < poll_interval:
            logger.info("Still optimizing... (%.0f min elapsed)", elapsed / 60)
        # Sleep in 1-second increments so MCReport detection responds within 1 s.
        _sleep_end = time.time() + poll_interval
        while time.time() < _sleep_end:
            time.sleep(1)
            # Quick MCReport check during sleep so we don't miss fast completions.
            for _f in _glob.glob(os.path.join(_docs, "*.MCReport")):
                try:
                    if os.path.getmtime(_f) >= _t0 - 2:
                        logger.info("Optimization complete (during sleep) — new MCReport: %s",
                                    os.path.basename(_f))
                        return True
                except OSError:
                    pass

    logger.warning("Optimization timed out after %d seconds", timeout)
    return False


# ---------------------------------------------------------------------------
# MCReport binary decoder — parse Vals/Inp fields without ORVisualizer UI
# ---------------------------------------------------------------------------

def _decode_mcreport_to_csv(mcreport_path: str, cfg: StrategyConfig, out_csv: str) -> bool:
    """
    Decode the MCReport binary (Vals field = gzip+base64 of performance metrics)
    and write a standard CSV.  Returns True on success.
    """
    import base64 as _b64
    import gzip as _gz
    import struct as _struct
    import re as _re
    import csv as _csv_mod

    # MCReport caption order (0-indexed, matches Caption_N keys)
    _CAPTION_ORDER = [
        "Net Profit", "Gross Profit", "Gross Loss", "Total Trades",
        "Pct Profitable", "Winning Trades", "Losing Trades",
        "Avg Trade", "Avg Winning Trade", "Avg Losing Trade",
        "Win/Loss Ratio", "Max Consecutive Winners", "Max Consecutive Losers",
        "Avg Bars in Winner", "Avg Bars in Loser",
        "Max Intraday Drawdown", "Profit Factor",
        "Return on Account", "Custom Fitness Value",
    ]

    try:
        with open(mcreport_path, "r", encoding="utf-8", errors="replace") as _fh:
            _lines = [_l.rstrip("\r\n") for _l in _fh]
    except Exception as _e:
        logger.warning("MCReport decode: cannot read file: %s", _e)
        return False

    def _extract_b64(field, start=0):
        for _i in range(start, len(_lines)):
            _m = _re.match(rf"^\s*{_re.escape(field)}\s*=\s*'(.*)", _lines[_i])
            if _m:
                _acc = _m.group(1)
                if _acc.endswith("'"):
                    return _acc[:-1], _i
                for _j in range(_i + 1, len(_lines)):
                    _ln = _lines[_j]
                    if _ln.endswith("'"):
                        _acc += _ln[:-1].strip()
                        break
                    elif _ln.startswith("[") or _re.match(r"^\s*\w+\s*=", _ln):
                        break
                    else:
                        _acc += _ln.strip()
                return _acc, _i
        return None, -1

    def _decode(b64str):
        _raw = _b64.b64decode(b64str)
        with _gz.GzipFile(fileobj=__import__("io").BytesIO(_raw)) as _gz_fh:
            return _gz_fh.read()

    inp_b64, _ = _extract_b64("Inp")
    vals_b64, _ = _extract_b64("Vals")
    if not inp_b64 or not vals_b64:
        logger.warning("MCReport decode: Inp or Vals field not found")
        return False

    try:
        _inp_raw = _decode(inp_b64)
        _vals_raw = _decode(vals_b64)
    except Exception as _e:
        logger.warning("MCReport decode: gzip error: %s", _e)
        return False

    _n_params_total = len(_inp_raw) // 8
    _n_vals_total   = len(_vals_raw) // 8

    # Auto-detect actual param count: MC64 may omit fixed params (e.g. LenLE) from
    # the Inp block even when they are checked in the wizard.  Try len(cfg.params)
    # first; if that yields garbled values (any param column out of its declared
    # range), fall back to len(cfg.params)-1, then -2, etc.
    _param_cols = len(cfg.params)
    _n_runs = 0
    for _try_p in range(len(cfg.params), 0, -1):
        if _n_params_total % _try_p != 0:
            continue
        _try_runs = _n_params_total // _try_p
        if _try_runs < 1:
            continue
        if _n_vals_total % _try_runs != 0:
            continue
        # Validate first 200 rows' param values against expected ranges
        _try_data = _struct.unpack(f"<{_n_params_total}d", _inp_raw)
        _ok = True
        for _pi, _p in enumerate(cfg.params[:_try_p]):
            _lo = min(_p.start, _p.stop) - abs(_p.step) * 4
            _hi = max(_p.start, _p.stop) + abs(_p.step) * 4
            for _ri in range(min(200, _try_runs)):
                _v = _try_data[_ri * _try_p + _pi]
                if not (_lo <= _v <= _hi):
                    _ok = False
                    break
            if not _ok:
                break
        if _ok:
            _param_cols = _try_p
            _n_runs = _try_runs
            break

    # Upward fallback for multi-signal charts: the Inp block may contain MORE
    # columns per run than len(cfg.params) (fixed inputs of other signals).
    # Try larger strides and locate each cfg.param's column by range-matching.
    _col_map = None   # list of column indices (one per cfg.param) when stride > len(cfg.params)
    if _n_runs < 1:
        for _try_p in range(len(cfg.params) + 1, 17):
            if _n_params_total % _try_p != 0:
                continue
            _try_runs = _n_params_total // _try_p
            if _try_runs < 1 or _n_vals_total % _try_runs != 0:
                continue
            _try_data = _struct.unpack(f"<{_n_params_total}d", _inp_raw)
            _sample = min(200, _try_runs)
            _assign = []
            _used = set()
            for _p in cfg.params:
                _lo = min(_p.start, _p.stop) - abs(_p.step) * 4
                _hi = max(_p.start, _p.stop) + abs(_p.step) * 4
                _found = -1
                for _ci in range(_try_p):
                    if _ci in _used:
                        continue
                    _vals_c = [_try_data[_ri * _try_p + _ci] for _ri in range(_sample)]
                    if all(_lo <= _v <= _hi for _v in _vals_c) and (
                            _try_runs == 1 or len(set(_vals_c)) > 1):
                        _found = _ci
                        break
                if _found < 0:
                    break
                _assign.append(_found)
                _used.add(_found)
            if len(_assign) == len(cfg.params):
                _param_cols = _try_p
                _n_runs = _try_runs
                _col_map = _assign
                logger.info("MCReport decode: upward stride=%d col_map=%s", _try_p, _col_map)
                break

    if _n_runs < 1:
        logger.warning("MCReport decode: unexpected sizes inp=%d vals=%d", len(_inp_raw), len(_vals_raw))
        return False

    _n_cols = _n_vals_total // _n_runs if _n_runs else 0
    if _n_cols < 1:
        logger.warning("MCReport decode: no output cols inp=%d vals=%d runs=%d", len(_inp_raw), len(_vals_raw), _n_runs)
        return False

    logger.info("MCReport decode: %d runs x %d param inputs, %d output cols",
                _n_runs, _param_cols, _n_cols)

    # Param values per run
    _param_data = _struct.unpack(f"<{_n_params_total}d", _inp_raw)

    # Metrics column names from Caption order (up to n_cols)
    _stat_names = _CAPTION_ORDER[:_n_cols]
    if _col_map is not None:
        _headers = [p.name for p in cfg.params] + _stat_names
    else:
        _headers = [p.name for p in cfg.params[:_param_cols]] + _stat_names

    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as _fcsv:
        _w = _csv_mod.writer(_fcsv)
        _w.writerow(_headers)
        for _i in range(_n_runs):
            if _col_map is not None:
                _params = [_param_data[_i * _param_cols + _ci] for _ci in _col_map]
            else:
                _params = list(_param_data[_i * _param_cols:(_i + 1) * _param_cols])
            _metrics = list(_struct.unpack(
                f"<{_n_cols}d",
                _vals_raw[_i * _n_cols * 8:(_i + 1) * _n_cols * 8]
            ))
            _w.writerow([f"{v:.6g}" for v in _params + _metrics])

    logger.info("MCReport decode: wrote %d rows -> %s", _n_runs, out_csv)
    return True


# ---------------------------------------------------------------------------
# Export optimization results to CSV
# ---------------------------------------------------------------------------

def export_optimization_results(
    conn: MultiChartsConnection,
    cfg: StrategyConfig,
    output_dir: str,
    opt_start_time: float = 0.0,
) -> str:
    """Find newest .MCReport in ~/Documents and decode it to CSV.

    Primary path: parse the MCReport binary (Vals field) directly — no UI needed.
    Fallback: open in ORVisualizer and use Alt+F → Save As → CSV.
    """
    import glob as _glob

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{cfg.name}_raw.csv")

    _docs = os.path.join(os.path.expanduser("~"), "Documents")
    _t0 = opt_start_time or (time.time() - 300)

    # ── Step 1: find newest .MCReport ────────────────────────────────────────
    _candidates = []
    for _f in _glob.glob(os.path.join(_docs, "*.MCReport")):
        try:
            if os.path.getmtime(_f) >= _t0 - 2:
                _candidates.append((_f, os.path.getmtime(_f)))
        except OSError:
            pass
    _candidates.sort(key=lambda x: x[1], reverse=True)

    if not _candidates:
        raise MCUIError(
            f"No .MCReport found in {_docs} newer than opt_start_time={_t0:.0f}"
        )
    _mcreport_path = _candidates[0][0]
    logger.info("Using MCReport: %s", os.path.basename(_mcreport_path))

    # ── Step 2: decode MCReport binary directly (preferred) ──────────────────
    if _decode_mcreport_to_csv(_mcreport_path, cfg, out_path):
        logger.info("Results exported (binary decode): %s", out_path)
        return out_path

    # ── Step 3: fallback — open in ORVisualizer, File→Save As→CSV ────────────
    logger.warning("Binary decode failed; trying ORVisualizer File→Save As")

    def _find_orv_window():
        for _tr in [r".*Optimization Report.*", r".*優化報告.*", r".*ORVisualizer.*"]:
            try:
                w = conn.app.window(title_re=_tr)
                if w.exists(timeout=0):
                    return w
            except Exception:
                pass
        return None

    _orv_wnd = _find_orv_window()
    if _orv_wnd is None:
        os.startfile(_mcreport_path)
        for _ in range(30):
            time.sleep(1)
            _orv_wnd = _find_orv_window()
            if _orv_wnd is not None:
                break

    if _orv_wnd is not None:
        try:
            _orv_wnd.set_focus()
        except Exception:
            pass
        time.sleep(0.5)

        # Alt+F → open File menu, then look for Save As / Export
        _pg_hotkey("alt", "f")
        time.sleep(0.4)
        # Try pressing 'a' for Save As, then 'e' for Export
        for _key in ["a", "e"]:
            _pg_press(_key)
            time.sleep(0.5)
            if _wait_for_save_dialog_and_save(conn, out_path):
                break
            _pg_press("escape")
            time.sleep(0.2)

    if not os.path.exists(out_path):
        raise MCUIError(f"Failed to export CSV to {out_path}")

    logger.info("Results exported (ORVisualizer): %s", out_path)
    return out_path


def _wait_for_save_dialog_and_save(conn: MultiChartsConnection, out_path: str) -> bool:
    """Handle a Save/SaveAs dialog: type path, select CSV, press Enter."""
    try:
        save_dlg = _wait_for_any_window(
            conn.app, [r".*[Ss]ave.*", r".*儲存.*", r".*另存.*"], timeout=6
        )
        pyautogui.hotkey("ctrl", "a")
        pyautogui.typewrite(out_path, interval=0.03)
        try:
            combo = save_dlg.child_window(class_name="ComboBox")
            rect_c = combo.rectangle()
            pyautogui.click((rect_c.left + rect_c.right) // 2,
                            (rect_c.top + rect_c.bottom) // 2)
            time.sleep(0.3)
            for label in ["CSV (*.csv)", "CSV", "*.csv"]:
                try:
                    combo.select(label)
                    break
                except Exception:
                    pass
        except Exception:
            pass
        _pg_press("enter")
        time.sleep(1.5)
        return os.path.exists(out_path)
    except (WindowNotFoundError, Exception):
        _pg_press("escape")
        return False


# ---------------------------------------------------------------------------
# Load and clean CSV
# ---------------------------------------------------------------------------

def load_results_csv(csv_path: str, cfg: StrategyConfig) -> pd.DataFrame:
    # index_col=False prevents pandas from treating extra values per row as a
    # MultiIndex — MC64 exports 2 metric sets per row (all-trades + long-only)
    # but the header only covers the first set, causing a column-count mismatch.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        df = pd.read_csv(csv_path, encoding="utf-8-sig", thousands=",", index_col=False)

    rename = {}
    for col in df.columns:
        stripped = col.strip()
        rename[col] = MC_COLUMN_MAP.get(stripped, stripped)
    df.rename(columns=rename, inplace=True)

    if "NetProfit" not in df.columns:
        raise ValueError(
            f"Column 'NetProfit' not found in {csv_path}. "
            f"Columns: {list(df.columns)}. Update MC_COLUMN_MAP in config.py."
        )
    if "MaxDrawdown" not in df.columns:
        # MCReport for some timeframes only exports 12 metric columns (no Max Intraday Drawdown).
        # Fall back to Gross Loss (always ≤0) as a signed proxy so optimization can continue.
        if "Gross Loss" in df.columns:
            logger.warning(
                "MaxDrawdown column missing in %s — using 'Gross Loss' as proxy", csv_path
            )
            df["MaxDrawdown"] = pd.to_numeric(df["Gross Loss"], errors="coerce")
        else:
            raise ValueError(
                f"Column 'MaxDrawdown' not found in {csv_path} and no 'Gross Loss' fallback available. "
                f"Columns: {list(df.columns)}. Update MC_COLUMN_MAP in config.py."
            )

    if "TotalTrades" not in df.columns:
        df["TotalTrades"] = PLATEAU_MIN_TRADES

    # Param name matching (case-insensitive)
    for param in cfg.params:
        if param.name not in df.columns:
            for col in df.columns:
                if col.lower() == param.name.lower():
                    df.rename(columns={col: param.name}, inplace=True)
                    break

    for col in ["NetProfit", "MaxDrawdown", "TotalTrades"] + [p.name for p in cfg.params]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["NetProfit", "MaxDrawdown"], inplace=True)
    df = df[df["TotalTrades"] >= PLATEAU_MIN_TRADES].copy()

    logger.info("Loaded %d valid rows from %s", len(df), csv_path)
    return df


# ---------------------------------------------------------------------------
# Top-level per-strategy orchestration
# ---------------------------------------------------------------------------

def run_optimization_for_strategy(
    conn: MultiChartsConnection,
    cfg: StrategyConfig,
    output_dir: str,
) -> str:
    logger.info("=== Starting %s (%d runs) ===", cfg.name, cfg.total_runs())
    _run_t0 = time.time()
    try:
        _t = time.time()
        _close_optimization_report()
        logger.info("[TIMING] close_report=%.2fs", time.time() - _t)

        _t = time.time()
        ensure_chart_ready(conn, cfg)
        logger.info("[TIMING] ensure_chart_ready=%.2fs", time.time() - _t)

        _t = time.time()
        configure_optimization(conn, cfg, date_range=cfg.insample)
        logger.info("[TIMING] configure_optimization=%.2fs", time.time() - _t)

        _opt_t0 = time.time()
        completed = wait_for_optimization_complete(conn, opt_start_time=_opt_t0)
        if not completed:
            raise OptimizationFailedError(f"{cfg.name}: timed out")
        logger.info("[TIMING] mc_optimization=%.2fs", time.time() - _opt_t0)

        _t = time.time()
        result = export_optimization_results(conn, cfg, output_dir, opt_start_time=_opt_t0)
        logger.info("[TIMING] export_results=%.2fs", time.time() - _t)

        logger.info("[TIMING] TOTAL=%.2fs  (setup=%.2fs  opt=%.2fs)",
                    time.time() - _run_t0,
                    _opt_t0 - _run_t0,
                    time.time() - _opt_t0)
        return result
    except Exception:
        _save_screenshot(cfg.name, output_dir)
        raise
