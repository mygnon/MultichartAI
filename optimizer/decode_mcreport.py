"""
Decode the MCReport binary Inp field to extract optimization results.
The Inp field is base64-encoded gzip-compressed binary data.
"""
import base64
import gzip
import io
import struct
import os
import csv
import re

MCREPORT_PATH = r"C:\Users\Tim\MultichartAI\results\Breakout_Daily_raw.csv"
OUT_PATH = r"C:\Users\Tim\MultichartAI\results\Breakout_Daily_decoded.csv"


def find_opt_report_section(lines):
    """Find the top-level [MCReport\\OptimizationReport\\OptReport] section."""
    in_section = False
    for i, line in enumerate(lines):
        if line.strip() == r"[MCReport\OptimizationReport\OptReport]":
            in_section = True
            continue
        if in_section and line.startswith("["):
            return None  # ended without finding Inp
        if in_section and "Inp" in line and "=" in line:
            return i
    return None


def extract_inp_base64(lines, start_idx):
    """Extract the full base64 string from Inp = '...' (may span multiple lines)."""
    full = ""
    # First line: Inp = 'H4sI...'
    first = lines[start_idx]
    m = re.match(r"\s*Inp\s*=\s*'(.*)", first)
    if m:
        full = m.group(1)
    else:
        return None

    # Continue collecting until closing quote
    if full.endswith("'"):
        return full[:-1]

    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if line.endswith("'"):
            full += line[:-1]
            break
        elif line.startswith("["):
            break
        else:
            full += line.strip()
    return full


def decode_and_decompress(b64_str):
    """Base64 decode + gzip decompress."""
    data = base64.b64decode(b64_str)
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        return gz.read()


def try_parse_as_text(raw_bytes):
    """Try UTF-16 LE, then UTF-8."""
    for enc in ("utf-16-le", "utf-16", "utf-8", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            if any(c.isalnum() for c in text[:200]):
                return text, enc
        except Exception:
            pass
    return None, None


def try_parse_fixed_binary(raw_bytes, num_inputs=2, num_cols=19):
    """
    Try to parse as a fixed-width binary table.
    Each row: num_inputs doubles (params) + num_cols doubles (stats) = (num_inputs+num_cols)*8 bytes
    Total rows should be ~2500 for LE=1..50, SE=1..50.
    """
    row_size = (num_inputs + num_cols) * 8
    n_rows = len(raw_bytes) // row_size
    print(f"  Binary parse attempt: {len(raw_bytes)} bytes / {row_size} bytes/row = {n_rows} rows")

    if n_rows < 100 or n_rows > 100000:
        return None

    rows = []
    for i in range(n_rows):
        chunk = raw_bytes[i * row_size:(i + 1) * row_size]
        try:
            vals = struct.unpack(f"<{num_inputs + num_cols}d", chunk)
            rows.append(vals)
        except Exception:
            return None
    return rows


def try_parse_variable_binary(raw_bytes):
    """Look for repeating float patterns that match 2500-row expectation."""
    # Try different column counts with 2 input params
    for ncols in range(10, 25):
        row_size = (2 + ncols) * 8
        n_rows = len(raw_bytes) // row_size
        if abs(n_rows - 2500) < 200:
            print(f"  Trying ncols={ncols}: {n_rows} rows @ {row_size} bytes/row")
            rows = []
            ok = True
            for i in range(n_rows):
                chunk = raw_bytes[i * row_size:(i + 1) * row_size]
                try:
                    vals = struct.unpack(f"<{2 + ncols}d", chunk)
                    rows.append(vals)
                except Exception:
                    ok = False
                    break
            if ok and rows:
                # Sanity check: first 2 cols should be params in 1-50 range
                params0 = [r[0] for r in rows[:20]]
                params1 = [r[1] for r in rows[:20]]
                p0_ok = all(1 <= v <= 50 for v in params0)
                p1_ok = all(1 <= v <= 50 for v in params1)
                print(f"    First params col0: {params0[:5]}, col1: {params1[:5]}, p0_ok={p0_ok}, p1_ok={p1_ok}")
                if p0_ok and p1_ok:
                    return rows, ncols
    return None, None


def main():
    print(f"Reading: {MCREPORT_PATH}")
    with open(MCREPORT_PATH, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    lines = [l.rstrip("\n").rstrip("\r") for l in lines]
    print(f"Total lines: {len(lines)}")

    inp_idx = find_opt_report_section(lines)
    if inp_idx is None:
        print("ERROR: Could not find [MCReport\\OptimizationReport\\OptReport] Inp= field")
        return

    print(f"Found Inp= at line {inp_idx + 1}")
    b64 = extract_inp_base64(lines, inp_idx)
    if not b64:
        print("ERROR: Could not extract base64 string")
        return

    print(f"Base64 length: {len(b64)} chars")

    print("Decoding base64 + gzip...")
    try:
        raw = decode_and_decompress(b64)
    except Exception as e:
        print(f"ERROR decompressing: {e}")
        return

    print(f"Decompressed size: {len(raw)} bytes")

    # Save raw bytes for inspection
    raw_out = OUT_PATH.replace(".csv", "_raw.bin")
    with open(raw_out, "wb") as f:
        f.write(raw)
    print(f"Saved raw binary to: {raw_out}")

    # Try text decode
    text, enc = try_parse_as_text(raw)
    if text:
        print(f"Decoded as text ({enc}), first 500 chars:")
        print(repr(text[:500]))
        # Save as text for inspection
        txt_out = OUT_PATH.replace(".csv", "_decoded.txt")
        with open(txt_out, "w", encoding="utf-8", errors="replace") as f:
            f.write(text)
        print(f"Saved decoded text to: {txt_out}")

        # Check if it's CSV-like
        lines_text = text.split("\n")
        print(f"\nFirst 5 text lines:")
        for l in lines_text[:5]:
            print(f"  {repr(l[:120])}")
        return

    print("Not readable as text — trying binary table parse...")

    # Try variable column count
    rows, ncols = try_parse_variable_binary(raw)
    if rows:
        print(f"\nParsed {len(rows)} rows with {ncols} stat columns")
        headers = ["LE", "SE"]
        # Standard MC optimization report columns (from Caption_0..Caption_18)
        stat_names = [
            "Net Profit", "Gross Profit", "Gross Loss", "Total Trades",
            "Pct Profitable", "Winning Trades", "Losing Trades",
            "Avg Trade", "Avg Winning Trade", "Avg Losing Trade",
            "Win/Loss Ratio", "Max Consecutive Winners", "Max Consecutive Losers",
            "Avg Bars in Winner", "Avg Bars in Loser",
            "Max Intraday Drawdown", "Profit Factor",
            "Return on Account", "Custom Fitness Value"
        ]
        headers += stat_names[:ncols]

        with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for row in rows:
                w.writerow([f"{v:.6g}" for v in row])
        print(f"Written CSV: {OUT_PATH}")
        print(f"\nFirst 5 rows:")
        for r in rows[:5]:
            print(f"  {[f'{v:.4g}' for v in r]}")
        return

    # Try fixed 19-col binary
    rows = try_parse_fixed_binary(raw, num_inputs=2, num_cols=19)
    if rows:
        print(f"\nParsed {len(rows)} rows (fixed 19-col)")
        stat_names = [
            "Net Profit", "Gross Profit", "Gross Loss", "Total Trades",
            "Pct Profitable", "Winning Trades", "Losing Trades",
            "Avg Trade", "Avg Winning Trade", "Avg Losing Trade",
            "Win/Loss Ratio", "Max Consecutive Winners", "Max Consecutive Losers",
            "Avg Bars in Winner", "Avg Bars in Loser",
            "Max Intraday Drawdown", "Profit Factor",
            "Return on Account", "Custom Fitness Value"
        ]
        headers = ["LE", "SE"] + stat_names
        with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for row in rows:
                w.writerow([f"{v:.6g}" for v in row])
        print(f"Written CSV: {OUT_PATH}")
        return

    print("Could not parse binary — check _raw.bin file manually")
    print(f"First 64 bytes (hex): {raw[:64].hex()}")


if __name__ == "__main__":
    main()
