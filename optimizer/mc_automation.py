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
import pywinauto
from pywinauto import Application, Desktop
from pywinauto.keyboard import send_keys

from config import (
    DateRange, StrategyConfig, MC_PROCESS_NAME,
    OPTIMIZATION_TIMEOUT_SECONDS, POLL_INTERVAL_SECONDS,
    MC_COLUMN_MAP, PLATEAU_MIN_TRADES,
)

logger = logging.getLogger(__name__)

pyautogui.PAUSE = 0.05
pyautogui.FAILSAFE = True

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
    Bring MC window to foreground.
    Win+D clears all overlapping windows first; after the desktop shell
    is the foreground there is no high-integrity window competing, so
    ShowWindow + SetForegroundWindow succeed from a non-admin process.
    """
    import win32gui
    import win32con

    # Skip if MC already has focus
    try:
        if win32gui.GetForegroundWindow() == hwnd:
            return
    except Exception:
        pass

    # Minimize all other windows via Win+D
    pyautogui.hotkey("win", "d")
    time.sleep(0.6)

    # Restore the MC window (now nothing is covering it)
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(0.4)
    except Exception:
        pass

    # SetForegroundWindow now works because the desktop was foreground
    try:
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        time.sleep(0.3)
    except Exception:
        pass

    # Final click to confirm focus
    try:
        rect = win32gui.GetWindowRect(hwnd)
        cx = (rect[0] + rect[2]) // 2
        cy = (rect[1] + rect[3]) // 2
        pyautogui.click(cx, cy)
        time.sleep(0.3)
    except Exception as e:
        logger.debug("_focus_window click failed: %s", e)


def _pg_hotkey(*keys) -> None:
    pyautogui.hotkey(*keys)
    time.sleep(0.3)


def _pg_type(text: str, interval: float = 0.05) -> None:
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
) -> pywinauto.WindowSpecification:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            wnd = app.window(title_re=title_re)
            if wnd.exists(timeout=0):
                return wnd
        except Exception:
            pass
        time.sleep(interval)
    raise WindowNotFoundError(f"Window '{title_re}' not found after {timeout:.0f}s")


def _wait_for_any_window(
    app: Application,
    title_res: List[str],
    timeout: float = 30.0,
) -> pywinauto.WindowSpecification:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for title_re in title_res:
            try:
                wnd = app.window(title_re=title_re)
                if wnd.exists(timeout=0):
                    return wnd
            except Exception:
                pass
        time.sleep(0.8)
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
    # Bring to foreground
    if conn._hwnd:
        _focus_window(conn._hwnd)
        time.sleep(0.5)
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
    for title in tab_titles:
        try:
            tab = dlg.child_window(title=title, class_name="SysTabControl32")
            tab.click_input()
            return
        except Exception:
            pass
    # Fallback: try TabControl and select by text
    try:
        tab_ctrl = dlg.child_window(class_name="SysTabControl32")
        for i in range(tab_ctrl.tab_count()):
            text = tab_ctrl.tab_text(i)
            if any(t.lower() in text.lower() for t in tab_titles):
                tab_ctrl.select(i)
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
    for title in checkbox_titles:
        try:
            cb = dlg.child_window(title=title, class_name="Button")
            if cb.exists(timeout=1):
                rect = cb.rectangle()
                # Enable checkbox via pyautogui click
                if not cb.get_check_state():
                    pyautogui.click((rect.left + rect.right) // 2,
                                    (rect.top + rect.bottom) // 2)
                    time.sleep(0.2)
                # Find the nearby Edit or DateTimePicker control
                try:
                    edit = dlg.child_window(class_name="SysDateTimePick32",
                                            found_index=0)
                    edit.set_focus()
                    edit_rect = edit.rectangle()
                    pyautogui.click((edit_rect.left + edit_rect.right) // 2,
                                    (edit_rect.top + edit_rect.bottom) // 2)
                    time.sleep(0.2)
                    _pg_hotkey("ctrl", "a")
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

def _pyautogui_click_popup_item(app: Application, item_titles: List[str]) -> bool:
    """
    Locate a popup menu item via pywinauto (read-only, no UIPI) and click with pyautogui.
    Returns True if an item was found and clicked.
    """
    try:
        popup = app.window(class_name="#32768")
        if not popup.exists(timeout=2):
            return False
        # Primary: enumerate via menu() API
        try:
            menu = popup.menu()
            for i in range(menu.item_count()):
                try:
                    item = menu.item(i)
                    text = item.text()
                    if any(t.lower() in text.lower() for t in item_titles):
                        r = item.rectangle()
                        pyautogui.click((r.left + r.right) // 2, (r.top + r.bottom) // 2)
                        time.sleep(0.3)
                        logger.debug("Clicked popup item: '%s'", text)
                        return True
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback: child_window by title
        for title in item_titles:
            try:
                item = popup.child_window(title=title)
                if item.exists(timeout=0):
                    r = item.rectangle()
                    pyautogui.click((r.left + r.right) // 2, (r.top + r.bottom) // 2)
                    time.sleep(0.3)
                    return True
            except Exception:
                pass
    except Exception as e:
        logger.debug("_pyautogui_click_popup_item failed: %s", e)
    return False


def _open_format_signals(conn: MultiChartsConnection) -> pywinauto.WindowSpecification:
    """Open Format Signals/Objects dialog via right-click → pyautogui click."""
    _focus_window(conn._hwnd)
    time.sleep(0.4)

    main = conn.main_window()
    rect = main.rectangle()
    cx = (rect.left + rect.right) // 2
    cy = (rect.top + rect.bottom) // 2

    # Right-click in chart area
    pyautogui.rightClick(cx, cy)
    time.sleep(0.7)

    target_names = [
        "Format Signals...", "Format Objects...", "Format Strategies...",
        "格式物件...", "格式訊號...", "格式策略...",
    ]
    if _pyautogui_click_popup_item(conn.app, target_names):
        try:
            return _wait_for_any_window(
                conn.app,
                [r".*Format.*", r".*格式.*"],
                timeout=8,
            )
        except WindowNotFoundError:
            pass

    # Dismiss any leftover menu
    _pg_press("escape")
    time.sleep(0.2)

    raise WindowNotFoundError(
        "Could not open Format Signals/Objects dialog.\n"
        "Make sure the chart is visible, has a strategy applied, and MC is in the foreground."
    )


def _select_signal_in_list(
    format_dlg: pywinauto.WindowSpecification,
    signal_name: str,
) -> None:
    try:
        lv = format_dlg.child_window(class_name="SysListView32")
        count = lv.item_count()
        for i in range(count):
            text = lv.get_item(i).text()
            if signal_name.lower() in text.lower():
                item = lv.get_item(i)
                rect = item.rectangle()
                pyautogui.click((rect.left + rect.right) // 2,
                                (rect.top + rect.bottom) // 2)
                time.sleep(0.2)
                logger.info("Selected signal '%s' (row %d)", text, i)
                return
        available = [lv.get_item(i).text() for i in range(count)]
        raise MCUIError(
            f"Signal '{signal_name}' not found. Available: {available}"
        )
    except MCUIError:
        raise
    except Exception as e:
        raise MCUIError(f"Could not find signal list: {e}") from e


def _click_button_in_dlg(dlg, titles: List[str]) -> bool:
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

def _set_cell_value(wizard, lv, row: int, col: int, value: str) -> None:
    """Set a ListView cell value via pyautogui double-click."""
    try:
        item = lv.get_item(row, col)
        rect = item.rectangle()
        x = (rect.left + rect.right) // 2
        y = (rect.top + rect.bottom) // 2
        pyautogui.doubleClick(x, y)
        time.sleep(0.25)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.typewrite(str(value), interval=0.04)
        pyautogui.press("tab")
        time.sleep(0.15)
    except Exception as e:
        logger.debug("Cell (%d,%d) set failed: %s", row, col, e)


def configure_optimization(
    conn: MultiChartsConnection,
    cfg: StrategyConfig,
    date_range: Optional[DateRange] = None,
) -> None:
    """
    Open Format Signals, select strategy, open optimization wizard,
    configure parameter grid, optionally set date range, and start.
    """
    format_dlg = _open_format_signals(conn)
    _select_signal_in_list(format_dlg, cfg.mc_signal_name)
    time.sleep(0.2)

    # First: set date range via Format button if date_range provided
    if date_range is not None:
        _click_button_in_dlg(format_dlg, ["Format", "格式", "編輯"])
        time.sleep(0.8)
        signal_dlg = _wait_for_any_window(
            conn.app, [r".*Format.*", r".*Signal.*", r".*格式.*"], timeout=10
        )
        _click_tab(signal_dlg, ["General", "一般"])
        time.sleep(0.3)
        _set_date_field(signal_dlg, conn.app,
                        ["Begin date:", "Begin Date", "開始日期", "起始日期"],
                        date_range.from_date)
        _set_date_field(signal_dlg, conn.app,
                        ["End date:", "End Date", "結束日期", "終止日期"],
                        date_range.to_date)
        _click_button_in_dlg(signal_dlg, ["OK", "確定"])
        time.sleep(0.5)
        # Re-open Format Signals
        format_dlg = _open_format_signals(conn)
        _select_signal_in_list(format_dlg, cfg.mc_signal_name)
        time.sleep(0.2)

    # Click Optimize... button
    if not _click_button_in_dlg(format_dlg, ["Optimize...", "Optimize", "優化..."]):
        # Fallback: Format → signal dialog → Optimize
        _click_button_in_dlg(format_dlg, ["Format", "格式"])
        time.sleep(0.8)
        signal_dlg = _wait_for_any_window(
            conn.app, [r".*Format.*", r".*Signal.*"], timeout=10
        )
        _click_button_in_dlg(signal_dlg, ["Optimize...", "Optimize", "優化..."])
    time.sleep(1)

    # Optimization wizard — Page 1
    wizard = _wait_for_any_window(
        conn.app, [r".*[Oo]ptimiz.*", r".*優化.*設定.*"], timeout=15
    )
    logger.info("Optimization wizard opened")

    _click_button_in_dlg(wizard, ["Regular Optimization", "標準優化", "一般優化"])
    _click_button_in_dlg(wizard, ["Next >", "下一步"])
    time.sleep(0.5)

    # Page 2 — configure parameter ListView
    lv = wizard.child_window(class_name="SysListView32")
    time.sleep(0.3)

    for param in cfg.params:
        row_idx = None
        count = lv.item_count()
        for i in range(count):
            for col in range(1, 4):
                try:
                    if param.name.lower() in lv.get_item(i, col).text().lower():
                        row_idx = i
                        break
                except Exception:
                    pass
            if row_idx is not None:
                break

        if row_idx is None:
            logger.warning("Param '%s' not found in wizard", param.name)
            continue

        logger.info("Config param '%s' row=%d [%s, %s, step=%s]",
                    param.name, row_idx, param.start, param.stop, param.step)

        # Enable Optimize checkbox (col 0) via pyautogui
        try:
            item = lv.get_item(row_idx, 0)
            rect = item.rectangle()
            pyautogui.click((rect.left + rect.right) // 2,
                            (rect.top + rect.bottom) // 2)
            time.sleep(0.15)
        except Exception:
            pass

        # Set Start/End/Step values — try col offsets 3,4,5 then 4,5,6
        for start_col in [3, 4]:
            try:
                _set_cell_value(wizard, lv, row_idx, start_col,     str(param.start))
                _set_cell_value(wizard, lv, row_idx, start_col + 1, str(param.stop))
                _set_cell_value(wizard, lv, row_idx, start_col + 2, str(param.step))
                break
            except Exception as e:
                logger.debug("Column offset %d failed: %s", start_col, e)

    # Exhaustive method
    _click_button_in_dlg(wizard, ["Exhaustive", "窮舉法", "完整搜尋"])

    _click_button_in_dlg(wizard, ["Next >", "下一步"])
    time.sleep(0.5)

    # Page 3 — just Finish
    _click_button_in_dlg(wizard, ["Finish", "完成"])
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

    try:
        lv = signal_dlg.child_window(class_name="SysListView32")
        count = lv.item_count()
        for i in range(count):
            for col in range(0, 3):
                try:
                    cell_text = lv.get_item(i, col).text().strip()
                    if cell_text in params:
                        # Set value column (usually col 1 or 2)
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
) -> bool:
    logger.info("Waiting for optimization (timeout=%ds)...", timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        _dismiss_error_dialogs(conn.app)

        for title_re in [r".*Optimization Report.*", r".*優化報告.*",
                         r".*優化結果.*"]:
            try:
                rpt = conn.app.window(title_re=title_re)
                if rpt.exists(timeout=0):
                    logger.info("Optimization complete.")
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
        time.sleep(poll_interval)

    logger.warning("Optimization timed out after %d seconds", timeout)
    return False


# ---------------------------------------------------------------------------
# Export optimization results to CSV
# ---------------------------------------------------------------------------

def export_optimization_results(
    conn: MultiChartsConnection,
    cfg: StrategyConfig,
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{cfg.name}_raw.csv")

    rpt_wnd = None
    for title_re in [r".*Optimization Report.*", r".*優化報告.*"]:
        try:
            w = conn.app.window(title_re=title_re)
            if w.exists(timeout=3):
                rpt_wnd = w
                break
        except Exception:
            pass
    if rpt_wnd is None:
        try:
            rpt_wnd = conn.main_window().child_window(
                title_re=r".*[Oo]ptimization.*[Rr]eport.*"
            )
        except Exception:
            pass
    if rpt_wnd is None:
        raise MCUIError("Optimization Report not found")

    rect = rpt_wnd.rectangle()
    pyautogui.click((rect.left + rect.right) // 2,
                    (rect.top + rect.bottom) // 2)
    time.sleep(0.3)

    saved = False
    # Try File > Save As via pyautogui
    _pg_hotkey("alt", "f")
    time.sleep(0.4)
    _pg_press("a")  # 'A' for Save As
    time.sleep(0.5)

    try:
        save_dlg = _wait_for_any_window(
            conn.app, [r".*[Ss]ave.*", r".*儲存.*", r".*另存.*"], timeout=8
        )
        # Type output path
        pyautogui.hotkey("ctrl", "a")
        pyautogui.typewrite(out_path, interval=0.03)

        # Try to set CSV type
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
        saved = os.path.exists(out_path)
    except WindowNotFoundError:
        _pg_press("escape")

    if not saved:
        raise MCUIError(f"Failed to export CSV to {out_path}")

    logger.info("Results exported: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Load and clean CSV
# ---------------------------------------------------------------------------

def load_results_csv(csv_path: str, cfg: StrategyConfig) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", thousands=",")

    rename = {}
    for col in df.columns:
        stripped = col.strip()
        rename[col] = MC_COLUMN_MAP.get(stripped, stripped)
    df.rename(columns=rename, inplace=True)

    for required in ["NetProfit", "MaxDrawdown"]:
        if required not in df.columns:
            raise ValueError(
                f"Column '{required}' not found in {csv_path}. "
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
    try:
        ensure_chart_ready(conn, cfg)
        configure_optimization(conn, cfg, date_range=cfg.insample)
        completed = wait_for_optimization_complete(conn)
        if not completed:
            raise OptimizationFailedError(f"{cfg.name}: timed out")
        return export_optimization_results(conn, cfg, output_dir)
    except Exception:
        _save_screenshot(cfg.name, output_dir)
        raise
