"""Decode the Vals field from MCReport to get optimization results."""
import base64
import gzip
import io
import struct
import csv
import re

MCREPORT_PATH = r"C:\Users\Tim\MultichartAI\results\Breakout_Daily_raw.csv"
OUT_PATH = r"C:\Users\Tim\MultichartAI\results\Breakout_Daily_decoded.csv"

def extract_field_b64(lines, field_name, start_search=0):
    """Extract a multi-line base64 field: FieldName = 'H4s...' """
    for i in range(start_search, len(lines)):
        m = re.match(rf"^\s*{re.escape(field_name)}\s*=\s*'(.*)", lines[i])
        if m:
            accumulated = m.group(1)
            if accumulated.endswith("'"):
                return accumulated[:-1], i
            for j in range(i + 1, len(lines)):
                line = lines[j].strip()
                if line.endswith("'"):
                    accumulated += line[:-1]
                    break
                elif lines[j].startswith("[") or re.match(r"^\s*\w+\s*=", lines[j]):
                    break
                else:
                    accumulated += line
            return accumulated, i
    return None, -1


def decode_gzip(b64):
    data = base64.b64decode(b64)
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        return gz.read()


def main():
    with open(MCREPORT_PATH, "r", encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip("\r\n") for l in f]

    print(f"Lines: {len(lines)}")

    # Decode Inp (input params) and Vals (output metrics)
    inp_b64, inp_idx = extract_field_b64(lines, "Inp")
    vals_b64, vals_idx = extract_field_b64(lines, "Vals")
    inp_ex_b64, inp_ex_idx = extract_field_b64(lines, "Inp_Ex")

    print(f"Inp at line {inp_idx+1}: {len(inp_b64)} chars")
    print(f"Inp_Ex at line {inp_ex_idx+1}: {len(inp_ex_b64)} chars")
    print(f"Vals at line {vals_idx+1}: {len(vals_b64)} chars")

    print("\nDecoding Inp (input params)...")
    inp_raw = decode_gzip(inp_b64)
    print(f"  Decompressed: {len(inp_raw)} bytes")
    n_params = len(inp_raw) // 8
    params = list(struct.unpack(f"<{n_params}d", inp_raw))
    print(f"  {n_params} doubles, first 10: {params[:10]}")

    print("\nDecoding Inp_Ex...")
    inp_ex_raw = decode_gzip(inp_ex_b64)
    print(f"  Decompressed: {len(inp_ex_raw)} bytes")
    n_ex = len(inp_ex_raw) // 8
    if n_ex * 8 == len(inp_ex_raw):
        ex_vals = list(struct.unpack(f"<{n_ex}d", inp_ex_raw))
        print(f"  {n_ex} doubles, first 10: {ex_vals[:10]}")
    else:
        print(f"  Not aligned to 8 bytes, first 64 hex: {inp_ex_raw[:64].hex()}")

    print("\nDecoding Vals (performance metrics)...")
    vals_raw = decode_gzip(vals_b64)
    print(f"  Decompressed: {len(vals_raw)} bytes")
    n_vals = len(vals_raw) // 8
    print(f"  = {n_vals} doubles total")

    # From the section header: Cols = '19', OptInputsCnt = '2', rows = n_params // 2
    n_runs = n_params // 2  # LE and SE per run
    print(f"  n_runs (from Inp): {n_runs}")
    n_cols = n_vals // n_runs if n_runs > 0 else 0
    print(f"  cols per run: {n_cols}")

    # Parse Inp as (LE, SE) pairs
    le_se = [(params[i], params[i+1]) for i in range(0, len(params), 2)]

    # Parse Vals as rows of n_cols doubles
    col_names = [
        "Net Profit", "Gross Profit", "Gross Loss", "Total Trades",
        "Pct Profitable", "Winning Trades", "Losing Trades",
        "Avg Trade", "Avg Winning Trade", "Avg Losing Trade",
        "Win/Loss Ratio", "Max Consecutive Winners", "Max Consecutive Losers",
        "Avg Bars in Winner", "Avg Bars in Loser",
        "Max Intraday Drawdown", "Profit Factor",
        "Return on Account", "Custom Fitness Value"
    ]

    rows_out = []
    for i in range(n_runs):
        le, se = le_se[i]
        metrics = struct.unpack(f"<{n_cols}d", vals_raw[i*n_cols*8:(i+1)*n_cols*8])
        rows_out.append((le, se) + metrics)

    print(f"\nFirst 5 rows preview:")
    for r in rows_out[:5]:
        le, se = int(r[0]), int(r[1])
        net_p = r[2]
        max_dd = r[17] if n_cols >= 17 else "?"
        print(f"  LE={le:2d} SE={se:2d}  NetProfit={net_p:>12.2f}  MaxDD={max_dd}")

    # Find column index for Max Intraday Drawdown (index 15 in 0-based col_names)
    # Cols present: 19 columns indexed 0..18 corresponding to Caption_0..Caption_18
    header = ["LE", "SE"] + col_names[:n_cols]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows_out:
            w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])

    print(f"\nWritten {len(rows_out)} rows → {OUT_PATH}")

    # Find best by NetProfit^2 / |MaxDrawdown|
    print("\n=== TOP PARAMS BY NetProfit² / |MaxDrawdown| ===")
    net_p_idx = 2  # col index in row (0=LE, 1=SE, 2=Net Profit, ...)
    max_dd_col_name = "Max Intraday Drawdown"
    max_dd_idx = header.index(max_dd_col_name) if max_dd_col_name in header else -1
    print(f"  Net Profit col idx: {net_p_idx}, MaxDD col idx: {max_dd_idx}")

    scored = []
    for row in rows_out:
        le, se = int(row[0]), int(row[1])
        net_p = row[net_p_idx]
        max_dd = row[max_dd_idx] if max_dd_idx >= 0 else 0
        if net_p > 0 and max_dd < 0:
            obj = (net_p * net_p) / abs(max_dd)
        else:
            obj = 0
        scored.append((obj, le, se, net_p, max_dd))

    scored.sort(reverse=True)
    print(f"\nTop 10:")
    print(f"  {'LE':>4} {'SE':>4}  {'NetProfit':>12}  {'MaxDD':>12}  {'Objective':>14}")
    for obj, le, se, np_, dd in scored[:10]:
        print(f"  {le:>4} {se:>4}  {np_:>12.2f}  {dd:>12.2f}  {obj:>14.2f}")


if __name__ == "__main__":
    main()
