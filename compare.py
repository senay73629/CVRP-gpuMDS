"""
compare.py
----------
Compare two results files (file1 vs file2) from gpuMDS / parMDS runs.

Features (choose with the -n / --number argument):
  1 - % error report (.txt)
  2 - % error comparison graph
  3 - total time comparison graph
  4 - loop time comparison graph
  5 - MST time comparison graph

Usage:
    python compare.py file1.txt file2.txt          # run all 5
    python compare.py file1.txt file2.txt -n 1     # report only
    python compare.py file1.txt file2.txt -n 3     # total-time graph only
    python compare.py file1.txt file2.txt -n 2 4   # error graph + loop graph
"""

import os
import re
import sys
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── CLI Arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Compare two results files across cost / time metrics."
)
parser.add_argument("file1", help="Path to first results file")
parser.add_argument("file2", help="Path to second results file")
parser.add_argument(
    "-n", "--number",
    type=int,
    nargs="+",
    choices=[1, 2, 3, 4, 5],
    default=[1, 2, 3, 4, 5],
    metavar="N",
    help=(
        "Which feature(s) to run (1-5). Default: all.\n"
        "  1 = %% error report (.txt)\n"
        "  2 = %% error graph\n"
        "  3 = total time graph\n"
        "  4 = loop time graph\n"
        "  5 = MST time graph"
    ),
)
parser.add_argument(
    "-l", "--location",
    default="local",
    help="Input file location label (e.g., local, remote, etc.)"
)
args = parser.parse_args()

FILE1    = os.path.abspath(args.file1)
FILE2    = os.path.abspath(args.file2)
FEATURES = set(args.number)

if not os.path.isfile(FILE1):
    sys.exit(f"[ERROR] file1 not found: {FILE1}")
if not os.path.isfile(FILE2):
    sys.exit(f"[ERROR] file2 not found: {FILE2}")

# Derive label names from the filenames (strip extension, take basename)
LABEL1 = os.path.splitext(os.path.basename(FILE1))[0]
LABEL2 = os.path.splitext(os.path.basename(FILE2))[0]

# Output directory = {LABEL1}_vs_{LABEL2}_on_{LOCATION}
LOCATION = args.location
OUT_DIR  = f"{LABEL1}_vs_{LABEL2}_on_{LOCATION}"
os.makedirs(OUT_DIR, exist_ok=True)

# BKS file lives next to compare.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BKS_FILE   = os.path.join(SCRIPT_DIR, "bks_costs.txt")

OUT_REPORT = os.path.join(OUT_DIR, f"{LABEL1}_vs_{LABEL2}_error_report.txt")
OUT_ERR_G  = os.path.join(OUT_DIR, f"{LABEL1}_vs_{LABEL2}_error_comparison.png")
OUT_TOTAL  = os.path.join(OUT_DIR, f"{LABEL1}_vs_{LABEL2}_total_time.png")
OUT_LOOP   = os.path.join(OUT_DIR, f"{LABEL1}_vs_{LABEL2}_loop_time.png")
OUT_MST    = os.path.join(OUT_DIR, f"{LABEL1}_vs_{LABEL2}_mst_time.png")

# ── Parse BKS file ───────────────────────────────────────────────────────────
def parse_bks(path):
    """Returns dict: instance_name -> best known cost (float)"""
    bks = {}
    if not os.path.isfile(path):
        print(f"[WARN] BKS file not found: {path}  (% errors will be skipped)")
        return bks
    skip = ("=", "-", "Best", "(", "Instance", "Total", "Source")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or any(line.startswith(p) for p in skip):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    bks[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return bks

bks = parse_bks(BKS_FILE)

# ── Dark theme colours ────────────────────────────────────────────────────────
DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
GRID_COL  = "#30363d"
TEXT_COL  = "#c9d1d9"
TITLE_COL = "#e6edf3"
COL1      = "#f78166"   # file1 colour (warm red)
COL2      = "#58a6ff"   # file2 colour (cool blue)

# ── Parse a results file ──────────────────────────────────────────────────────
def parse_results(path):
    """
    Returns dict: instance_name -> {cost, mst, loop, post, total, valid}

    Supports two log formats:
      gpuMDS:  inputs/X-n101-k25.vrp  MinCost: 30663  TimeMST: ...  TimeLoop: ...  TimePostProcess: ...  TimeTotal: ...  VALID
      parMDS:  inputs/X-n101-k25.vrp  Cost 30000  TimeMST 0.1  TimeLoop 0.2  TimePost 0.3  TimeTotal 0.4 VALID
    """
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m_path = re.search(r"inputs[/\\](.+?)\.vrp", line)
            if not m_path:
                continue
            inst = m_path.group(1)

            # -- cost --
            cost = None
            m = re.search(r"MinCost:\s*([0-9.eE+\-]+)", line)
            if not m:
                m = re.search(r"\bCost\s+([0-9.eE+\-]+)", line)
            if m:
                cost = float(m.group(1))

            def get_time(pattern):
                m2 = re.search(pattern + r":?\s*([0-9.eE+\-]+)", line)
                return float(m2.group(1)) if m2 else None

            mst   = get_time(r"TimeMST")
            loop  = get_time(r"TimeLoop")
            post  = get_time(r"TimePost(?:Process)?")
            total = get_time(r"TimeTotal")
            valid = "VALID" in line.upper()

            if cost is not None:
                data[inst] = dict(cost=cost, mst=mst, loop=loop,
                                  post=post, total=total, valid=valid)
    return data

# ── Dark axis helper ──────────────────────────────────────────────────────────
def dark_ax(ax, ytitle="", xtitle="", title=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.set_ylabel(ytitle, color=TEXT_COL, fontsize=10)
    ax.set_xlabel(xtitle, color=TEXT_COL, fontsize=10)
    if title:
        ax.set_title(title, color=TITLE_COL, fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", color=GRID_COL, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.grid(axis="x", color=GRID_COL, linestyle=":",  linewidth=0.3, alpha=0.4)

# ══════════════════════════════════════════════════════════════════════════════
# Load both result files
# ══════════════════════════════════════════════════════════════════════════════
data1 = parse_results(FILE1)
data2 = parse_results(FILE2)

# Only keep instances that appear in both files AND in BKS
all_common = sorted(set(data1) & set(data2))
if not all_common:
    sys.exit("[ERROR] No common instances found between the two files.")

if bks:
    common = [i for i in all_common if i in bks]
    if not common:
        print("[WARN] No common instances found in BKS — falling back to all shared instances.")
        common = all_common
else:
    common = all_common

print(f"File 1 ({LABEL1}): {len(data1)} instances")
print(f"File 2 ({LABEL2}): {len(data2)} instances")
print(f"BKS entries      : {len(bks)}")
print(f"Matched instances: {len(common)}")

x = np.arange(len(common))

# ── Collect arrays ────────────────────────────────────────────────────────────
costs1  = np.array([data1[i]["cost"]                       for i in common])
costs2  = np.array([data2[i]["cost"]                       for i in common])
totals1 = np.array([data1[i].get("total") or np.nan        for i in common])
totals2 = np.array([data2[i].get("total") or np.nan        for i in common])
loops1  = np.array([data1[i].get("loop")  or np.nan        for i in common])
loops2  = np.array([data2[i].get("loop")  or np.nan        for i in common])
msts1   = np.array([data1[i].get("mst")   or np.nan        for i in common])
msts2   = np.array([data2[i].get("mst")   or np.nan        for i in common])

# Percentage error: (cost - BKS) / BKS * 100
bks_vals = np.array([bks.get(i, np.nan) for i in common])
err1     = (costs1 - bks_vals) / bks_vals * 100.0
err2     = (costs2 - bks_vals) / bks_vals * 100.0

# ══════════════════════════════════════════════════════════════════════════════
# Feature 1 – % error report (.txt)
# ══════════════════════════════════════════════════════════════════════════════
if 1 in FEATURES:
    C1, C2, C3 = 24, 14, 10
    HDR = (f"{'Instance':<{C1}} "
           f"{f'{LABEL1} Cost':>{C2}} {f'{LABEL1} Err%':>{C3}} "
           f"{f'{LABEL2} Cost':>{C2}} {f'{LABEL2} Err%':>{C3}}")
    SEP = "-" * len(HDR)

    def fmtc(v): return f"{v:>{C2}.2f}" if not np.isnan(v) else f"{'N/A':>{C2}}"
    def fmtp(v): return f"{v:>{C3}.2f}%" if not np.isnan(v) else f"{'N/A':>{C3}}"

    with open(OUT_REPORT, "w") as f:
        f.write(f"CVRP Results Comparison: {LABEL1} vs {LABEL2}\n")
        f.write("% error is relative to the Best Known Solution (BKS).\n")
        f.write("=" * len(HDR) + "\n")
        f.write(HDR + "\n")
        f.write(SEP + "\n")
        for i, inst in enumerate(common):
            f.write(f"{inst:<{C1}} {fmtc(costs1[i])} {fmtp(err1[i])} "
                    f"{fmtc(costs2[i])} {fmtp(err2[i])}\n")
        f.write(SEP + "\n")

        def sumline(errs, label):
            v = errs[~np.isnan(errs)]
            if len(v) == 0:
                return f"  {label}: no data\n"
            return (f"  {label}  avg={np.mean(v):.2f}%  "
                    f"min={np.min(v):.2f}%  max={np.max(v):.2f}%\n")

        f.write(f"\nSummary ({len(common)} matched instances):\n")
        f.write(sumline(err1, LABEL1))
        f.write(sumline(err2, LABEL2))

        wins1 = int(np.sum(costs1 < costs2))
        wins2 = int(np.sum(costs2 < costs1))
        ties  = int(np.sum(costs1 == costs2))
        f.write(f"\n  {LABEL1} has lower cost on: {wins1} instances\n")
        f.write(f"  {LABEL2} has lower cost on: {wins2} instances\n")
        f.write(f"  Tied on: {ties} instances\n")

    with open(OUT_REPORT) as f:
        print(f.read())
    print(f"Report  → {OUT_REPORT}")

# ══════════════════════════════════════════════════════════════════════════════
# Feature 2 – % error comparison graph
# ══════════════════════════════════════════════════════════════════════════════
if 2 in FEATURES:
    diff = err1 - err2   # positive = file1 worse

    fig, axes = plt.subplots(2, 1, figsize=(26, 14),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor(DARK_BG)

    # Top: error lines
    ax1 = axes[0]
    dark_ax(ax1,
            title=f"% Error Comparison: {LABEL1} vs {LABEL2}",
            ytitle="% Error (vs best of the two)")
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax1.plot(x, err1, color=COL1, lw=1.5, marker="o",
             markersize=3, label=f"{LABEL1}  (avg {np.nanmean(err1):.2f}%)", zorder=3)
    ax1.plot(x, err2, color=COL2, lw=1.5, marker="s",
             markersize=3, label=f"{LABEL2}  (avg {np.nanmean(err2):.2f}%)", zorder=3)
    ax1.fill_between(x, err1, err2, alpha=0.12, color="#ffffff")
    ax1.axhline(np.nanmean(err1), color=COL1, lw=0.8, linestyle="--", alpha=0.6)
    ax1.axhline(np.nanmean(err2), color=COL2, lw=0.8, linestyle="--", alpha=0.6)
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax1.tick_params(which="minor", axis="y", color=GRID_COL, length=2)
    ax1.legend(facecolor="#21262d", edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)

    # Bottom: difference bar
    ax2 = axes[1]
    dark_ax(ax2,
            title=f"Difference ({LABEL1} − {LABEL2}) per Instance",
            ytitle="Δ % Error", xtitle="Instance")
    bar_colors = ["#f85149" if d > 0 else "#3fb950" for d in diff]
    ax2.bar(x, diff, color=bar_colors, width=0.8, zorder=3)
    ax2.axhline(0, color=TEXT_COL, lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(common, rotation=90, ha="center", fontsize=5.2)
    p1 = plt.matplotlib.patches.Patch(color="#3fb950", label=f"{LABEL2} better")
    p2 = plt.matplotlib.patches.Patch(color="#f85149", label=f"{LABEL1} better")
    ax2.legend(handles=[p1, p2], facecolor="#21262d",
               edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)

    plt.tight_layout(pad=2)
    plt.savefig(OUT_ERR_G, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Graph   → {OUT_ERR_G}")

# ══════════════════════════════════════════════════════════════════════════════
# Helper: single-time-metric comparison (used by features 3, 4, 5)
# ══════════════════════════════════════════════════════════════════════════════
def plot_time_metric(vals1, vals2, metric_name, out_path):
    """Two-panel plot: time lines on top, speedup ratio bar on bottom."""
    speedup = vals2 / vals1   # > 1 means file2 is slower (file1 is faster)

    fig, axes = plt.subplots(2, 1, figsize=(24, 13),
                             gridspec_kw={"height_ratios": [2.5, 1]})
    fig.patch.set_facecolor(DARK_BG)

    # Top: time lines
    ax1 = axes[0]
    dark_ax(ax1,
            title=f"{metric_name} Comparison: {LABEL1} vs {LABEL2}",
            ytitle=f"{metric_name} (seconds, log scale)")
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax1.plot(x, vals1, color=COL1, lw=1.6, marker="s", markersize=3.5,
             label=f"{LABEL1} (avg {np.nanmean(vals1):.4f} s)", zorder=3)
    ax1.plot(x, vals2, color=COL2, lw=1.6, marker="o", markersize=3.5,
             label=f"{LABEL2} (avg {np.nanmean(vals2):.4f} s)", zorder=3)
    ax1.fill_between(x, vals1, vals2,
                     where=vals2 >= vals1, alpha=0.12, color=COL2,
                     label=f"{LABEL2} slower region")
    ax1.fill_between(x, vals1, vals2,
                     where=vals1 > vals2, alpha=0.12, color=COL1,
                     label=f"{LABEL1} slower region")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3g}s"))
    ax1.legend(facecolor="#21262d", edgecolor=GRID_COL,
               labelcolor=TEXT_COL, fontsize=9)

    # Bottom: speedup / ratio bar  (file2 / file1)
    ax2 = axes[1]
    dark_ax(ax2,
            title=f"Time Ratio ({LABEL2} / {LABEL1})  — >1 means {LABEL1} is faster",
            ytitle="Ratio", xtitle="Instance")
    bar_col = ["#3fb950" if s >= 1 else "#f85149" for s in speedup]
    ax2.bar(x, speedup, color=bar_col, width=0.8, zorder=3)
    ax2.axhline(1.0, color=TEXT_COL, lw=0.9, linestyle="--")

    valid_sp = speedup[~np.isnan(speedup)]
    if len(valid_sp):
        s_lo = max(0, np.nanmin(valid_sp) - 0.2)
        s_hi = np.nanmax(valid_sp) + 0.2
        ax2.set_ylim(s_lo, s_hi)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax2.tick_params(which="minor", axis="y", color=GRID_COL, length=2)
    ax2.grid(axis="y", which="minor", color=GRID_COL,
             linestyle=":", lw=0.4, alpha=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(common, rotation=90, ha="center", fontsize=5.2)

    green_p = plt.matplotlib.patches.Patch(color="#3fb950", label=f"{LABEL1} faster")
    red_p   = plt.matplotlib.patches.Patch(color="#f85149", label=f"{LABEL2} faster")
    ax2.legend(handles=[green_p, red_p], facecolor="#21262d",
               edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)

    plt.tight_layout(pad=2)
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Graph   → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Feature 3 – total time comparison graph
# ══════════════════════════════════════════════════════════════════════════════
if 3 in FEATURES:
    plot_time_metric(totals1, totals2, "Total Time", OUT_TOTAL)

# ══════════════════════════════════════════════════════════════════════════════
# Feature 4 – loop time comparison graph
# ══════════════════════════════════════════════════════════════════════════════
if 4 in FEATURES:
    plot_time_metric(loops1, loops2, "Loop Time", OUT_LOOP)

# ══════════════════════════════════════════════════════════════════════════════
# Feature 5 – MST time comparison graph
# ══════════════════════════════════════════════════════════════════════════════
if 5 in FEATURES:
    plot_time_metric(msts1, msts2, "MST Time", OUT_MST)

# Transfer input files to output directory
import shutil
try:
    shutil.move(FILE1, os.path.join(OUT_DIR, os.path.basename(FILE1)))
    shutil.move(FILE2, os.path.join(OUT_DIR, os.path.basename(FILE2)))
    print(f"Transferred {os.path.basename(FILE1)} and {os.path.basename(FILE2)} to {OUT_DIR}")
except Exception as e:
    print(f"[WARN] Could not move input files: {e}")

print("\nDone.")
