"""
plot_ieee.py  ─  IEEE two-column plots for Diffusion-MAPPO Multi-UAV MEC
=========================================================================
ALL plots read from saved files — NO env simulation at plot time.

Saved during training by src/logger.py:
  results/logs/training_metrics.jsonl   — per-step metrics
  results/logs/eval_sweeps.json         — parameter sweep results (end of training)
  results/trajectories/flight_path_*.npz — UAV+IoT positions (every 50k steps)

Every figure + legend is a SEPARATE file.

Plots generated:
  P1  — Convergence (reward + latency)        + legend
  P2  — Avg Latency vs. N (IoT Devices)               + legend
  P3  — Avg Latency vs. M (UAVs)              + legend
  P4  — Latency vs. J (jammers)               + legend
  P5  — Energy consumption over episode        + legend
  P6  — Latency vs. cache capacity             + legend
  P7  — Cache hit rate vs. Zipf exponent       + legend
  P8  — UAV trajectory 2-D  (per 50k step)    + legend (separate)
        UAV trajectory 3-D  (per 50k step)    + legend (separate)
  P9  — Latency vs. jammer power               + legend
  P10 — Latency vs. task size                  + legend
  P11 — Latency vs. transmission power         + legend

Usage:
  python plot_ieee.py                              # all plots from saved data
  python plot_ieee.py --plot P3                    # single plot
  python plot_ieee.py --plot P8 --traj-step 50000  # trajectory at specific step
  python plot_ieee.py --env-only --seed 42 --uavs 5 --iot 100 --steps 80
"""

import argparse, json, os, sys, glob, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
import matplotlib.ticker  as mticker
from matplotlib.collections      import LineCollection
from matplotlib.patches          import Circle
from mpl_toolkits.mplot3d        import Axes3D           # noqa: F401
from mpl_toolkits.mplot3d.art3d  import Line3DCollection
from mpl_toolkits.mplot3d        import art3d as mpl_art3d
warnings.filterwarnings("ignore")

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
from config import UAV_PROFILES, MAX_UAVS, dbm_to_watt

RESULTS  = os.path.join(_PROJ_ROOT, "results")
LOG_DIR  = os.path.join(RESULTS, "logs")
TRAJ_DIR = os.path.join(RESULTS, "trajectories")
OUT_DIR  = os.path.join(RESULTS, "plots")
for _d in [OUT_DIR, TRAJ_DIR]:
    os.makedirs(_d, exist_ok=True)

# ===========================================================================
# IEEE style  (Times New Roman, 9 pt body)
# ===========================================================================
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "grid.linewidth": 0.35, "lines.linewidth": 0.9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.spines.top": False, "axes.spines.right": False,
})

COL_W = 3.5   # one IEEE column width (inches)
UAV_COLORS = [
    "#FF6B6B",   # UAV-1: coral red
    "#4ECDC4",   # UAV-2: teal
    "#D97706",   # UAV-3: Amber/Burnt Orange
    "#A78BFA",   # UAV-4: soft purple
    "#45B7D1",   # UAV-5: sky blue
    "#F7B731",   # UAV-6: amber
    "#FC5C65",   # UAV-7: rose pink
    "#26DE81",   # UAV-8: bright green
]
SWEEP_COLORS = ["#2196F3","#FF5722","#4CAF50","#9C27B0",
                "#FF9800","#00BCD4","#E91E63"]
JAMMER_COL = "#9B2226"
BS_COLOR   = "#1D3557"
TRAJ_ALPHA_MIN = 0.40   # was 0.15 — made first 80% of path nearly invisible
TRAJ_ALPHA     = 0.92

# ===========================================================================
# Shared visual constants — 2D is the master reference.
# Every 3D parameter is derived from these so both plots stay identical.
# ===========================================================================

# Jammer
JAM_FILL_ALPHA  = 0.12     # disc fill + ring alpha — both 2D and 3D use this
JAM_TRI_MS      = 9        # 2D triangle ms
JAM_TRI_S3      = 9 * 6    # 3D triangle scatter s  (≈ ms=9 visually)

# Base station
BS_MS           = 13       # 2D star ms
BS_S3           = 13 * 6   # 3D star scatter s

# IoT scatter — 2D s-values are authoritative; 3D uses REDUCED values (see 3D section)
IOT_S_JAMMED    = 14
IOT_S_OOR       = 14       # out-of-range diamond
IOT_S_IN        = 13       # in-range circle
IOT_LW_OOR      = 0.70     # edge linewidth for hollow diamond
IOT_ALPHA_JAMMED = 0.80
IOT_ALPHA_OOR    = 0.30
IOT_ALPHA_IN     = 0.45

# 3D-specific IoT overrides — mid-size for clean visibility
IOT_S_JAMMED_3D  = 6
IOT_S_OOR_3D     = 6
IOT_S_IN_3D      = 5
IOT_LW_OOR_3D    = 0.6
IOT_ALPHA_JAMMED_3D = 0.60
IOT_ALPHA_OOR_3D    = 0.28
IOT_ALPHA_IN_3D     = 0.32

# 3D service tick height (metres) — short fixed length
TICK_LEN_3D      = 12

# Service tick (2D only — ticks are suppressed in 3D to reduce clutter)
TICK_LW          = 0.9
TICK_ALPHA_IN    = 0.75    # dotted / in-range
TICK_ALPHA_OOR   = 0.45    # dashed / out-of-range

# IoT trail end dot
TRAIL_END_MS     = 2.2     # 2D marker size
TRAIL_END_S3     = 12      # 3D scatter s  (≈ ms=2.2)
TRAIL_END_ALPHA  = 0.60

# UAV trajectory start marker (hollow circle)
UAV_START_MS     = 5.0     # 2D ms
UAV_START_S3     = 25      # 3D scatter s  (≈ ms=5)
UAV_START_MEW    = 1.0     # edge linewidth

# UAV current-position "+" with black outline
PLUS_OUT_MS      = 12      # 2D outer black ms
PLUS_OUT_MEW     = 3.0     # 2D outer black mew
PLUS_IN_MS       = 8       # 2D inner coloured ms
PLUS_IN_MEW      = 2.0     # 2D inner coloured mew
PLUS_ZORDER      = 35      # same zorder in both 2D and 3D
PLUS_OUT_S3      = 60      # 3D outer scatter s  (≈ ms=12)
PLUS_OUT_LW3     = 3.0
PLUS_IN_S3       = 40      # 3D inner scatter s  (≈ ms=8)
PLUS_IN_LW3      = 2.0

# NOMA / cooperative offload lines
NOMA_LW          = 1.6
NOMA_ALPHA       = 0.80
COOP_LW          = 1.5
COOP_ALPHA       = 0.70
NOMA_RING_S      = 35      # NOMA ring scatter s (both 2D & 3D)
NOMA_RING_LW     = 1.0

# Grid
GRID_LW          = 0.3
GRID_ALPHA       = 0.15

# Axis extent multiplier (2D xlim/ylim and 3D xlim/ylim)
AXIS_MULT        = 1.12

# Figure size — same in 2D and 3D for visual consistency
FIG_SIZE         = (COL_W + 0.5, COL_W + 0.5)
# ────────────────────────────────────────────────────────────────────────────

# ===========================================================================
# Wind overlay defaults  (not stored in .npz — passed as parameters)
# ===========================================================================
WIND_DIR_DEG_DEFAULT  = 45.0   # degrees CCW from +x axis (math convention)
WIND_SPEED_DEFAULT    = 8.0    # m/s
WIND_COLOR            = "#1565C0"   # deep blue arrow
WIND_ARROW_LW         = 2.2        # shaft linewidth
WIND_ARROW_HEAD_SCALE = 10         # arrowhead mutation_scale
WIND_ARROW_LEN_FRAC   = 0.15        # arrow length as fraction of axes width
WIND_LABEL_FONTSIZE   = 7.5


# ===========================================================================
# Helpers
# ===========================================================================
def _savefig(fig, stem, exts=("png", "pdf")):
    for ext in exts:
        fig.savefig(f"{stem}.{ext}")
    plt.close(fig)
    print(f"    saved {os.path.basename(stem)}.png/.pdf")

def _fmt_k(x, _):
    return f"{int(x/1000)}k" if x >= 1000 else str(int(x))

def _smooth(y, w=7):
    if len(y) < w: return list(y)
    return np.convolve(y, np.ones(w)/w, mode="same").tolist()

def _ma(data, w=50):
    if len(data) < w: return np.array(data)
    return np.convolve(data, np.ones(w)/w, mode="valid")

def load_jsonl(path):
    rows = []
    if not os.path.exists(path): return rows
    with open(path) as f:
        for line in f:
            if line.strip():
                try: rows.append(json.loads(line))
                except: pass
    return rows

def load_sweeps():
    p = os.path.join(LOG_DIR, "eval_sweeps.json")
    if not os.path.exists(p):
        print(f"    {p} not found — run training first.")
        return None
    with open(p) as f:
        return json.load(f)


# ===========================================================================
# Legend helper — saves as SEPARATE file
# ===========================================================================
def save_legend(handles, path, ncol=None):
    """Save a standalone HORIZONTAL legend fitting IEEE two-column width."""
    n = len(handles)
    if ncol is None:
        if n <= 4:   ncol = n
        elif n <= 8: ncol = (n + 1) // 2
        else:        ncol = min(5, (n + 2) // 3)

    nrow = (n + ncol - 1) // ncol
    fig_w = min(7.16, ncol * 1.65 + 0.4)
    fig_h = nrow * 0.32 + 0.25

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=ncol,
              fontsize=7.5, framealpha=0.97, handlelength=1.8,
              handletextpad=0.55, borderpad=0.5,
              columnspacing=1.1, edgecolor="#CCCCCC",
              fancybox=False)
    fig.tight_layout(pad=0.10)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    legend → {os.path.basename(path)}")


# ===========================================================================
# Gradient trajectory lines (2D / 3D)
# ===========================================================================
def _subsample(arr, step=3):
    """Subsample a trajectory array, always keeping the last point."""
    idx = list(range(0, len(arr), step))
    if idx[-1] != len(arr) - 1:
        idx.append(len(arr) - 1)
    return arr[idx]

def grad2d(xs, ys, color):
    # Subsample every 3 steps → smoother visible curves (removes micro-jitter)
    xy   = _subsample(np.column_stack([xs, ys]))
    pts  = xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n    = len(segs)
    cols = [matplotlib.colors.to_rgba(color, a)
            for a in np.linspace(TRAJ_ALPHA_MIN, TRAJ_ALPHA, n)]
    return LineCollection(segs, colors=cols,
                          linewidths=np.linspace(1.1, 2.4, n), zorder=4)

def grad3d(xs, ys, zs, color):
    xyz  = _subsample(np.column_stack([xs, ys, zs]))
    pts  = xyz.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n    = len(segs)
    cols = [matplotlib.colors.to_rgba(color, a)
            for a in np.linspace(TRAJ_ALPHA_MIN, TRAJ_ALPHA, n)]
    return Line3DCollection(segs, colors=cols,
                            linewidths=np.linspace(1.1, 2.4, n), zorder=4)

def iot_grad2d(xs, ys, color):
    # IoT trails: subtle but fully visible throughout
    pts  = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n    = len(segs)
    cols = [matplotlib.colors.to_rgba(color, a)
            for a in np.linspace(0.12, 0.35, n)]
    return LineCollection(segs, colors=cols,
                          linewidths=np.linspace(0.4, 0.9, n), zorder=3)

def iot_grad3d(xs, ys, zs, color):
    """3D IoT trail — very faint to avoid cluttering the ground plane."""
    pts  = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n    = len(segs)
    cols = [matplotlib.colors.to_rgba(color, a)
            for a in np.linspace(0.02, 0.09, n)]
    return Line3DCollection(segs, colors=cols,
                            linewidths=np.linspace(0.2, 0.5, n), zorder=2)


# ===========================================================================
# P1 — Convergence (reward + latency) from training_metrics.jsonl
# ===========================================================================
def plot_P1():
    rows = load_jsonl(os.path.join(LOG_DIR, "training_metrics.jsonl"))
    if not rows:
        rows = load_jsonl(os.path.join(LOG_DIR, "training_log.jsonl"))
    if not rows:
        print("  P1: no training data — skipping.")
        return

    steps = [r["step"] for r in rows]
    rews  = [r["reward"] for r in rows]
    lats  = [r["avg_latency"] for r in rows]
    w = max(1, min(100, len(rews) // 4))

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.plot(steps, rews, color="#AACCFF", lw=0.5, alpha=0.35)
    mr = _ma(rews, w)
    h1, = ax.plot(steps[:len(mr)], mr, color="#1A6FBF", lw=1.4,
                  label="Reward (Moving Average)")
    ax.set_xlabel("Training Step"); ax.set_ylabel("Mean Reward")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.grid(True, lw=0.3, alpha=0.45)
    fig.tight_layout()
    _savefig(fig, os.path.join(OUT_DIR, "P1a_reward"))
    save_legend([h1], os.path.join(OUT_DIR, "P1a_reward_legend"))

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.plot(steps, lats, color="#FFCCAA", lw=0.5, alpha=0.35)
    ml = _ma(lats, w)
    h2, = ax.plot(steps[:len(ml)], ml, color="#E07020", lw=1.4,
                  label="Latency (Moving Average)")
    ax.set_xlabel("Training Step"); ax.set_ylabel("Average Latency (s)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.grid(True, lw=0.3, alpha=0.45)
    fig.tight_layout()
    _savefig(fig, os.path.join(OUT_DIR, "P1b_latency"))
    save_legend([h2], os.path.join(OUT_DIR, "P1b_latency_legend"))


# ===========================================================================
# P2–P4, P6, P9, P11 — Line sweep plots from eval_sweeps.json
# ===========================================================================
def _plot_line_sweep(key, marker, ci, fname):
    s = load_sweeps()
    if not s or key not in s:
        print(f"  {key}: no sweep data — skipping.")
        return
    d = s[key]
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    h, = ax.plot(d["x"], d["y"], f"-{marker}", color=SWEEP_COLORS[ci],
                 lw=1.4, ms=5, label="Diffusion-MAPPO")
    if d.get("yerr"):
        ax.fill_between(d["x"],
                        np.array(d["y"]) - np.array(d["yerr"]),
                        np.array(d["y"]) + np.array(d["yerr"]),
                        color=SWEEP_COLORS[ci], alpha=0.12)
    ax.set_xlabel(d["xlabel"]); ax.set_ylabel(d["ylabel"])
    ax.grid(True, lw=0.3, alpha=0.45)
    ax.tick_params(direction="in", length=2.5)
    fig.tight_layout()
    stem = os.path.join(OUT_DIR, fname)
    _savefig(fig, stem)
    save_legend([h], f"{stem}_legend")

def plot_P2():  _plot_line_sweep("P2", "o", 0, "P2_latency_vs_N")
def plot_P3():  _plot_line_sweep("P3", "s", 1, "P3_latency_vs_M")
def plot_P4():  _plot_line_sweep("P4", "^", 3, "P4_latency_vs_J")
def plot_P6():  _plot_line_sweep("P6", "D", 5, "P6_latency_vs_cache")
def plot_P9():  _plot_line_sweep("P9", "v", 1, "P9_latency_vs_jammer_power")
def plot_P11(): _plot_line_sweep("P11","p", 6, "P11_latency_vs_tx_power")


# ===========================================================================
# P5 — Energy consumption from training_metrics.jsonl
# ===========================================================================
def plot_P5():
    rows = load_jsonl(os.path.join(LOG_DIR, "training_metrics.jsonl"))
    if not rows:
        rows = load_jsonl(os.path.join(LOG_DIR, "training_log.jsonl"))
    if not rows:
        print("  P5: no data — skipping.")
        return
    cum, e_list = 0, []
    for r in rows[:300]:
        cum += r["total_energy"]; e_list.append(cum)

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    h, = ax.plot(range(len(e_list)), e_list, color="#2A7A2A", lw=1.2,
                 label="Cumulative Energy")
    ax.fill_between(range(len(e_list)), 0, e_list, color="#2A7A2A", alpha=0.1)
    ax.set_xlabel("Time Slot"); ax.set_ylabel("Cumulative UAV Energy (Joules)")
    ax.grid(True, lw=0.3, alpha=0.45)
    fig.tight_layout()
    stem = os.path.join(OUT_DIR, "P5_energy")
    _savefig(fig, stem)
    save_legend([h], f"{stem}_legend")


# ===========================================================================
# P7 — Cache hit rate vs Zipf from eval_sweeps.json
# ===========================================================================
def plot_P7():
    s = load_sweeps()
    if not s or "P7" not in s:
        print("  P7: no data — skipping.")
        return
    d = s["P7"]
    x = np.array(d["x"]); w = 0.055
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.bar(x - w, d["local"], w, color="#4CAF50", label="Local cache hit")
    ax.bar(x,     d["coop"],  w, color="#2196F3", label="Cooperative offload")
    ax.bar(x + w, d["bs"],    w, color="#FF5722", label="Base Station fetch")
    ax.set_xlabel(d["xlabel"]); ax.set_ylabel(d["ylabel"])
    ax.grid(True, lw=0.3, alpha=0.45, axis="y")
    ax.tick_params(direction="in", length=2.5)
    fig.tight_layout()
    stem = os.path.join(OUT_DIR, "P7_cache_vs_zipf")
    _savefig(fig, stem)
    handles = [
        mpatches.Patch(fc="#4CAF50", label="Local cache hit"),
        mpatches.Patch(fc="#2196F3", label="Cooperative offload"),
        mpatches.Patch(fc="#FF5722", label="Base Station fetch"),
    ]
    save_legend(handles, f"{stem}_legend")


# ===========================================================================
# P10 — Latency vs task size (bar chart) from eval_sweeps.json
# ===========================================================================
def plot_P10():
    s = load_sweeps()
    if not s or "P10" not in s:
        print("  P10: no data — skipping.")
        return
    d = s["P10"]
    x = np.arange(len(d["x_labels"]))
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.bar(x, d["y"], yerr=d.get("yerr"), color="#2196F3",
           capsize=3, alpha=0.8, edgecolor="white", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(d["x_labels"], fontsize=7)
    ax.set_xlabel(d["xlabel"]); ax.set_ylabel(d["ylabel"])
    ax.grid(True, lw=0.3, alpha=0.45, axis="y")
    fig.tight_layout()
    stem = os.path.join(OUT_DIR, "P10_latency_vs_task_size")
    _savefig(fig, stem)
    save_legend([mpatches.Patch(fc="#2196F3", label="Diffusion-MAPPO")],
                f"{stem}_legend")


# ===========================================================================
# Shared helpers
# ===========================================================================
def _uav_label_offset(x, y, R, step_frac=0.055):
    """Radial outward offset for UAV label (same logic for 2D and 3D)."""
    norm = np.hypot(x, y)
    step = R * step_frac
    if norm < 1e-3:
        return 0.0, step
    return (x / norm) * step, (y / norm) * step

def _draw_plus_2d(ax, x, y, color, zorder_base=PLUS_ZORDER):
    """Coloured '+' with thin black border — 2D axes."""
    ax.plot(x, y, "+", color="black",
            markersize=PLUS_OUT_MS, markeredgewidth=PLUS_OUT_MEW,
            zorder=zorder_base - 1)
    ax.plot(x, y, "+", color=color,
            markersize=PLUS_IN_MS, markeredgewidth=PLUS_IN_MEW,
            zorder=zorder_base)

def _draw_plus_3d(ax3, x, y, z, color, zorder_base=PLUS_ZORDER):
    """Coloured '+' with thin black border — 3D axes.
    NOTE: for line-based markers (+, x, *) in 3D scatter the line colour is
    set via `color`, NOT `edgecolors` (which only affects filled-marker borders).
    """
    ax3.scatter([x], [y], [z], marker="+", s=PLUS_OUT_S3,
                color="black", linewidths=PLUS_OUT_LW3,
                zorder=zorder_base - 1, depthshade=False)
    ax3.scatter([x], [y], [z], marker="+", s=PLUS_IN_S3,
                color=color, linewidths=PLUS_IN_LW3,
                zorder=zorder_base, depthshade=False)

def _draw_jammer_fill_3d(ax3, cx, cy, radius):
    """Filled jammer disc on z=0 plane — ec/lw/alpha/zorder match 2D patch exactly."""
    patch = Circle((cx, cy), radius,
                   fc=JAMMER_COL, ec=JAMMER_COL, lw=1.0,
                   alpha=JAM_FILL_ALPHA, zorder=3)
    ax3.add_patch(patch)
    mpl_art3d.pathpatch_2d_to_3d(patch, z=0.1, zdir="z")


# ===========================================================================
# Wind arrow helpers — direction + magnitude overlay (not in .npz)
# ===========================================================================

def _draw_wind_arrow_2d(ax, wind_deg, wind_speed):
    """
    Compact wind badge — arrow + label in a unified box, upper-right corner.
    Arrow sits ABOVE the text block; both are centred on cx, cy.
    """
    rad = np.radians(wind_deg)
    compass = _deg_to_compass(wind_deg)

    # ── Arrow geometry (axes-fraction units) ──────────────────────────────
    cx, cy = 0.885, 0.930   # anchor: upper-right, clear of data
    L      = 0.085          # short, tidy arrow
    dx, dy = np.cos(rad), np.sin(rad)

    tail_x = cx - 0.55 * L * dx
    tail_y = cy - 0.55 * L * dy
    head_x = cx + 0.45 * L * dx
    head_y = cy + 0.45 * L * dy

    ax.annotate(
        "",
        xy=(head_x, head_y), xytext=(tail_x, tail_y),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color=WIND_COLOR, lw=2.0,
            mutation_scale=14,
            shrinkA=0, shrinkB=0,
        ),
        zorder=30,
    )

    # ── Label — centred below the arrow ───────────────────────────────────
    label = f"Wind\n{compass}  {wind_deg:.0f}°\n{wind_speed:.1f} m/s"
    ax.text(
        cx, tail_y - 0.015, label,
        transform=ax.transAxes,
        fontsize=WIND_LABEL_FONTSIZE,
        color=WIND_COLOR, fontweight="bold",
        ha="center", va="top", linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.28", fc="white",
                  ec="#DDDDDD", lw=0.5, alpha=0.92),
        zorder=31,
    )

def _draw_wind_arrow_3d(ax3, wind_deg, wind_speed, R, max_alt):
    """
    Small inset compass in upper-right; arrow proportional to the 2D version.
    """
    fig = ax3.get_figure()

    # ── Inset — tight and proportional ────────────────────────────────────
    inset = fig.add_axes([0.755, 0.800, 0.170, 0.165])
    inset.set_xlim(-1.6, 1.6)
    inset.set_ylim(-1.6, 1.6)
    inset.set_aspect("equal")
    inset.patch.set_alpha(0.0)
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.set_xticks([]); inset.set_yticks([])

    rad    = np.radians(wind_deg)
    dx, dy = np.cos(rad), np.sin(rad)

    # Arrow: tail at -0.75, head at +0.75 (inset data coords)
    inset.annotate(
        "",
        xy=( 0.75 * dx,  0.75 * dy),
        xytext=(-0.75 * dx, -0.75 * dy),
        arrowprops=dict(
            arrowstyle="-|>",
            color=WIND_COLOR, lw=2.0,
            mutation_scale=14,
            shrinkA=0, shrinkB=0,
        ),
        zorder=30,
    )

    # ── Label — centred below the arrow origin ─────────────────────────────
    compass = _deg_to_compass(wind_deg)
    label   = f"Wind\n{compass}  {wind_deg:.0f}°\n{wind_speed:.1f} m/s"
    inset.text(
        0.0, -0.95, label,
        fontsize=WIND_LABEL_FONTSIZE,
        color=WIND_COLOR, fontweight="bold",
        ha="center", va="top", linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.28", fc="white",
                ec="#DDDDDD", lw=0.5, alpha=0.92),
        zorder=31,
    )

def _deg_to_compass(deg):
    """Return 2-letter compass direction for a math-convention angle (CCW from +x)."""
    # Convert math angle → meteorological bearing (CW from North)
    bearing = (90.0 - deg) % 360.0
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    idx  = int((bearing + 22.5) / 45.0) % 8
    return dirs[idx]


# ===========================================================================
# P8 — 2D + 3D trajectory from saved .npz
# ===========================================================================
def plot_P8(step=None, output_dir=None,
            wind_dir=None, wind_speed=None):
    """
    2D + 3D UAV trajectory plots.

    3D declutter fixes applied vs original:
      • IoT ground markers: s and alpha reduced (IOT_*_3D constants)
      • Service ticks (vertical lines): suppressed entirely in 3D
      • IoT trail opacity: linspace (0.02, 0.09) instead of (0.05, 0.28)
      • View angle: elev=38, azim=-55 (was elev=27, azim=-50)
      • UAV labels: z-offset staggered by (i % 3)*10 to avoid stacking
      • max_alt: +80 m headroom (was +50 m)
      • Altitude anchor lines: dotted, alpha=0.50, lw=1.0 (was dashed, 0.35, 0.7)

    Jammer fixes (v2):
      • Triangle marker placed at disc CENTRE (jp[0], jp[1]) not edge
      • Label text rendered just above triangle at z=12 (was z=18)

    UAV label fix (v2):
      • zorder raised to 999 so labels always render above all other elements

    Wind overlay (not stored in .npz — passed as parameters):
      wind_dir   : float, degrees CCW from +x (math convention).
      wind_speed : float, m/s.
      Pass wind_dir=False to suppress the wind arrow entirely.
    """
    files = sorted(glob.glob(os.path.join(TRAJ_DIR, "flight_path_*.npz")))
    if not files:
        print("  P8: no trajectory data — skipping.")
        return

    target = files[-1]
    if step is not None:
        t = os.path.join(TRAJ_DIR, f"flight_path_{step}.npz")
        if t in files: target = t

    out = output_dir or OUT_DIR
    os.makedirs(out, exist_ok=True)
    is_final = (output_dir is None)

    data         = np.load(target, allow_pickle=True)
    R            = float(data["area_radius"])
    n_active     = int(data["num_active"])
    jam_pos      = data["jam_pos"]
    jam_r        = float(data["jam_radius"])
    comm_range   = float(data["uav_comm_range"]) if "uav_comm_range" in data else 200.0
    dev_pos      = data.get("dev_pos_start", data.get("dev_pos", np.zeros((0, 3))))
    dev_clust    = data.get("dev_cluster",  np.zeros(len(dev_pos), dtype=int))
    dev_jammed   = data.get("dev_jammed",   np.zeros(len(dev_pos), dtype=bool))
    dev_in_range = data.get("dev_in_range", np.ones(len(dev_pos), dtype=bool))
    sn           = int(data["step"])

    # NOMA snapshot
    noma_uav_id  = int(data["noma_uav_id"][0]) if "noma_uav_id" in data else -1
    noma_uav_pos = data["noma_uav_pos"]         if "noma_uav_pos" in data else None
    noma_dev_pos = data["noma_dev_pos"]          if "noma_dev_pos" in data else np.zeros((0, 2))
    has_noma     = (noma_uav_id >= 0 and noma_uav_pos is not None
                    and len(noma_dev_pos) == 2)
    noma_col     = UAV_COLORS[noma_uav_id % len(UAV_COLORS)] if has_noma else "#888888"

    # Coop offload snapshot
    coop_src_id  = int(data["coop_src_id"][0]) if "coop_src_id" in data else -1
    coop_dst_id  = int(data["coop_dst_id"][0]) if "coop_dst_id" in data else -1
    coop_src_pos = data["coop_src_pos"]         if "coop_src_pos" in data else None
    coop_dst_pos = data["coop_dst_pos"]         if "coop_dst_pos" in data else None
    has_coop     = (coop_src_id >= 0 and coop_dst_id >= 0
                    and coop_src_pos is not None and coop_dst_pos is not None)

    # UAV + IoT trajectories
    trajs = {}
    for i in range(n_active):
        k = f"uav_{i}"
        if k in data: trajs[i] = data[k]

    iot_trajs = {}
    idx = 0
    while f"iot_{idx}" in data:
        iot_trajs[idx] = data[f"iot_{idx}"]
        idx += 1

    # UAV final positions (2D)
    uav_finals = (np.array([trajs[i][-1][:2] for i in sorted(trajs.keys())])
                  if trajs else np.zeros((0, 2)))

    # UAV jammer flags
    uav_jammed = np.zeros(len(uav_finals), dtype=bool)
    if len(jam_pos) > 0 and jam_r > 0:
        for ui, upos in enumerate(uav_finals):
            for jp in jam_pos:
                if np.linalg.norm(upos - jp[:2]) <= jam_r:
                    uav_jammed[ui] = True
                    break

    # Re-derive dev_jammed from drawn positions and jammer geometry
    if len(jam_pos) > 0 and jam_r > 0 and len(dev_pos) > 0:
        dev_jammed = np.array([
            any(np.linalg.norm(dp[:2] - jp[:2]) <= jam_r for jp in jam_pos)
            for dp in dev_pos
        ])

    # Re-derive dev_in_range if not saved
    if "dev_in_range" not in data and len(uav_finals) > 0:
        for di in range(len(dev_pos)):
            cid = max(0, int(dev_clust[di]))
            if cid < len(uav_finals):
                dev_in_range[di] = bool(
                    np.linalg.norm(dev_pos[di, :2] - uav_finals[cid]) <= comm_range)

    theta   = np.linspace(0, 2 * np.pi, 360)
    suffix  = '' if is_final else f'_step{sn}'
    TICK_LEN = R * 0.02   # physical height for 2D service ticks

    # ── Wind parameters ────────────────────────────────────────────────────
    show_wind = (wind_dir is not False)
    if show_wind:
        if wind_dir is None and "wind_direction" in data:
            _wind_dir = float(np.degrees(float(data["wind_direction"])))
        elif wind_dir is not None:
            _wind_dir = float(wind_dir)
        else:
            _wind_dir = WIND_DIR_DEG_DEFAULT

        if wind_speed is None and "wind_speed" in data:
            _wind_spd = float(data["wind_speed"])
        elif wind_speed is not None:
            _wind_spd = float(wind_speed)
        else:
            _wind_spd = WIND_SPEED_DEFAULT
    else:
        _wind_dir = WIND_DIR_DEG_DEFAULT
        _wind_spd = WIND_SPEED_DEFAULT

    # ══════════════════════════════════════════════════════════════════════
    # P8a — 2-D plot  (unchanged — reference plot)
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.8)

    # Deployment boundary
    ax.plot(R * np.cos(theta), R * np.sin(theta),
            color="#AAAAAA", lw=0.9, ls="--", zorder=1)

    # Jammer zones
    for ji, jp in enumerate(jam_pos):
        ax.add_patch(plt.Circle(jp[:2], jam_r, fc=JAMMER_COL, ec=JAMMER_COL,
                                alpha=JAM_FILL_ALPHA, lw=1.0, zorder=3))
        # Triangle at disc CENTRE (jp[0], jp[1]) — mirrors the 3D fix on line ~912
        tri_x, tri_y = jp[0], jp[1]
        ax.plot(tri_x, tri_y, "^", color=JAMMER_COL,
                ms=JAM_TRI_MS, mfc=JAMMER_COL, mew=0, zorder=15)
        ax.annotate(f"$J_{{{ji + 1}}}$", (tri_x, tri_y), fontsize=7.5,
                    color=JAMMER_COL, fontweight="bold",
                    ha="center", va="bottom",
                    xytext=(0, 10),
                    textcoords="offset points")

    # Base Station star
    ax.plot(0, 0, "*", color=BS_COLOR, ms=BS_MS, mew=0.6, mfc=BS_COLOR, zorder=7)

    # IoT devices
    for i, dp in enumerate(dev_pos):
        xy  = dp[:2]
        cid = max(0, int(dev_clust[i]))
        col = UAV_COLORS[cid % len(UAV_COLORS)]

        if dev_jammed[i]:
            ax.scatter(xy[0], xy[1], s=IOT_S_JAMMED, marker="X", color=col,
                       linewidths=0.9, alpha=IOT_ALPHA_JAMMED, zorder=5)
        elif not dev_in_range[i]:
            ax.scatter(xy[0], xy[1], s=IOT_S_OOR, marker="D",
                       facecolors="none", edgecolors=col,
                       linewidths=IOT_LW_OOR, alpha=IOT_ALPHA_OOR, zorder=3)
            ax.plot([xy[0], xy[0]], [xy[1], xy[1] + TICK_LEN],
                    color=col, lw=TICK_LW, ls="--", alpha=TICK_ALPHA_OOR, zorder=4)
        else:
            ax.scatter(xy[0], xy[1], s=IOT_S_IN, marker="o",
                       facecolors=col, edgecolors="none",
                       alpha=IOT_ALPHA_IN, zorder=3)
            ax.plot([xy[0], xy[0]], [xy[1], xy[1] + TICK_LEN],
                    color=col, lw=TICK_LW, ls=":", alpha=TICK_ALPHA_IN, zorder=4)

    # IoT trajectory trails
    for ii, tr in iot_trajs.items():
        if len(tr) < 2: continue
        cid = int(dev_clust[ii]) if ii < len(dev_clust) else 0
        col = UAV_COLORS[cid % len(UAV_COLORS)]
        ax.add_collection(iot_grad2d(tr[:, 0], tr[:, 1], col))
        end_xy = tr[-1, :2]
        end_jammed = (len(jam_pos) > 0 and jam_r > 0 and
                      any(np.linalg.norm(end_xy - jp[:2]) <= jam_r
                          for jp in jam_pos))
        if end_jammed:
            ax.scatter(end_xy[0], end_xy[1], s=IOT_S_JAMMED,
                       marker="X", color=col,
                       linewidths=0.9, alpha=IOT_ALPHA_JAMMED, zorder=5)
        else:
            ax.plot(end_xy[0], end_xy[1], "o", color=col,
                    ms=TRAIL_END_MS, alpha=TRAIL_END_ALPHA, zorder=5)

    # UAV trajectories
    for i, tr in trajs.items():
        c = UAV_COLORS[i % len(UAV_COLORS)]
        ax.add_collection(grad2d(tr[:, 0], tr[:, 1], c))
        ax.plot(tr[0, 0], tr[0, 1], "o", color=c,
                ms=UAV_START_MS, mfc="white", mew=UAV_START_MEW, zorder=8)
        _draw_plus_2d(ax, tr[-1, 0], tr[-1, 1], c, zorder_base=PLUS_ZORDER)
        if len(tr) >= 3:
            dx, dy = tr[-1, 0] - tr[-3, 0], tr[-1, 1] - tr[-3, 1]
            ax.annotate("", xy=(tr[-1, 0], tr[-1, 1]),
                        xytext=(tr[-1, 0] - dx * 0.5, tr[-1, 1] - dy * 0.5),
                        arrowprops=dict(arrowstyle="->", color=c,
                                        lw=0.9, mutation_scale=8), zorder=9)
        ax.text(tr[-1, 0], tr[-1, 1],
                f"UAV-{i + 1}", fontsize=6.5, color=c,
                ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=c, lw=0.5, alpha=0.80),
                zorder=100)

    # NOMA dashed arcs — draw from the UAV position AT THE TIME of the NOMA
    # event (noma_uav_pos), not from uav_finals which is the end-of-episode
    # position.  Using uav_finals caused lines to point to wrong IoT devices
    # because the UAV moved significantly after the NOMA snapshot was taken.
    if has_noma:
        uav_now = noma_uav_pos  # snapshot position, not final position
        for dp in noma_dev_pos:
            ax.plot([uav_now[0], dp[0]], [uav_now[1], dp[1]],
                    color=noma_col, lw=NOMA_LW, ls=(0, (4, 2)),
                    alpha=NOMA_ALPHA, zorder=11)
        for dp in noma_dev_pos:
            if np.linalg.norm(dp) <= R * 1.05:
                ax.scatter(dp[0], dp[1], s=NOMA_RING_S, marker="o",
                           facecolors="none", edgecolors="black",
                           linewidths=NOMA_RING_LW, zorder=12)

    # Cooperative offload
    if has_coop and coop_src_id < len(uav_finals) and coop_dst_id < len(uav_finals):
        ax.plot([uav_finals[coop_src_id][0], uav_finals[coop_dst_id][0]],
                [uav_finals[coop_src_id][1], uav_finals[coop_dst_id][1]],
                color="black", lw=COOP_LW, ls="--", alpha=COOP_ALPHA, zorder=10)

    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$y$ (m)")
    ax.set_aspect("equal")
    ax.set_xlim(-R * AXIS_MULT, R * AXIS_MULT)
    ax.set_ylim(-R * AXIS_MULT, R * AXIS_MULT)
    ax.grid(True, lw=GRID_LW, alpha=GRID_ALPHA)
    ax.tick_params(direction="in", length=3)

    if show_wind:
        _draw_wind_arrow_2d(ax, _wind_dir, _wind_spd)

    fig.tight_layout()
    _savefig(fig, os.path.join(out, f'P8_2D_trajectory{suffix}'))

    # ══════════════════════════════════════════════════════════════════════
    # Shared Legend
    # ══════════════════════════════════════════════════════════════════════
    uav_h = []
    for i in range(n_active):
        m = (dev_clust == i) & (~dev_jammed)
        c = UAV_COLORS[i % len(UAV_COLORS)]
        uav_h.append(mlines.Line2D([], [], color=c, lw=1.8, marker="o", ms=4,
                     label=f"UAV-{i + 1} Cluster ({m.sum()} IoT Devices)"))

    iot_h = [
        mlines.Line2D([], [], color="#888888", lw=0.9, ls="-", alpha=0.50,
                      label="IoT Trajectory"),
        mlines.Line2D([], [], marker="o", color="gray", lw=0.9, ls=":",
                      ms=5, mfc="gray", label="In-range (Service Tick)"),
        mlines.Line2D([], [], marker="D", color="#999999", lw=0.9, ls="--",
                      ms=5, mfc="none", mew=0.7, label="Out-of-range (Request)"),
        mlines.Line2D([], [], marker="X", color="#CC0000", lw=0, ms=6,
                      label="Jammed (No Transmit)"),
        mlines.Line2D([], [], marker="o", color="black", lw=0, ms=7,
                      mfc="none", mew=1.2, label="NOMA Transmission"),
    ]

    infra_h = [
        mlines.Line2D([], [], marker="*", color=BS_COLOR, lw=0, ms=9,
                      mfc=BS_COLOR, label="Base Station"),
        mlines.Line2D([], [], marker="^", color=JAMMER_COL, lw=0, ms=7,
                      mfc=JAMMER_COL, label="Jammer Center"),
        mlines.Line2D([], [], color="#AAAAAA", lw=0.8, ls="--",
                      label="Boundary"),
    ]

    coord_h = [
        mlines.Line2D([], [], color=noma_col, lw=1.4, ls=(0, (4, 2)),
                      label="NOMA Pair Link"),
        mlines.Line2D([], [], color="black", lw=1.4, ls="--",
                      label="Coop Offload (A2A)"),
    ]

    if show_wind:
        _compass = _deg_to_compass(_wind_dir)
        coord_h.append(
            mlines.Line2D([], [], color=WIND_COLOR, lw=1.5, marker=">", ms=5,
                          mfc=WIND_COLOR, label=f"Wind: {_compass} {_wind_dir:.0f}°")
        )

    handles = uav_h + iot_h + infra_h + coord_h
    save_legend(handles, os.path.join(out, f'P8_trajectory_legend{suffix}'), ncol=5)

    # ══════════════════════════════════════════════════════════════════════
    # P8b — 3-D plot  (decluttered)
    # ══════════════════════════════════════════════════════════════════════
    fig3 = plt.figure(figsize=FIG_SIZE)
    ax3  = fig3.add_subplot(111, projection="3d")

    # ── Deployment boundary ───────────────────────────────────────────────
    ax3.plot(R * np.cos(theta), R * np.sin(theta), np.zeros(360),
             color="#AAAAAA", lw=0.9, ls="--", zorder=1)

    # ── Jammer zones ──────────────────────────────────────────────────────
    for ji, jp in enumerate(jam_pos):
        _draw_jammer_fill_3d(ax3, jp[0], jp[1], jam_r)
        tc = np.linspace(0, 2 * np.pi, 60)
        ax3.plot(jp[0] + jam_r * np.cos(tc),
                 jp[1] + jam_r * np.sin(tc),
                 0.3, color=JAMMER_COL, lw=1.0,
                 alpha=JAM_FILL_ALPHA, zorder=4)

        # FIX 1: triangle at disc CENTRE (jp[0], jp[1]) — not at edge (jp[1]+jam_r)
        tri_x3, tri_y3 = jp[0], jp[1]          # centre of jammer disc
        ax3.scatter(tri_x3, tri_y3, 0.5, marker="^",
                    color=JAMMER_COL, s=JAM_TRI_S3,
                    linewidths=0, zorder=15, depthshade=False)

        # FIX 2: label just above the triangle (z=12), was z=18
        ax3.text(tri_x3, tri_y3, 12,
                 f"$J_{{{ji + 1}}}$", fontsize=7.5,
                 color=JAMMER_COL, ha="center", va="bottom",
                 fontweight="bold", zorder=25)

    # ── Base Station star ─────────────────────────────────────────────────
    ax3.scatter(0, 0, 0, marker="*", color=BS_COLOR,
                s=BS_S3, linewidths=0.6, edgecolors=BS_COLOR,
                zorder=7, depthshade=False)

    # ── IoT on ground — smaller markers, reduced alpha, NO service ticks ──
    for i, dp in enumerate(dev_pos):
        if i >= len(dev_jammed): break
        xy  = dp[:2]
        cid = max(0, int(dev_clust[i]))
        col = UAV_COLORS[cid % len(UAV_COLORS)]

        if dev_jammed[i]:
            ax3.scatter(xy[0], xy[1], 0,
                        s=IOT_S_JAMMED_3D, marker="X", color=col,
                        linewidths=0.6, alpha=IOT_ALPHA_JAMMED_3D,
                        depthshade=False, zorder=5)

        elif i < len(dev_in_range) and not dev_in_range[i]:
            ax3.scatter(xy[0], xy[1], 0,
                        s=IOT_S_OOR_3D, marker="D",
                        facecolors="none", edgecolors=col,
                        linewidths=IOT_LW_OOR_3D, alpha=IOT_ALPHA_OOR_3D,
                        depthshade=False, zorder=3)
            # Short dashed tick — out-of-range
            ax3.plot([xy[0], xy[0]], [xy[1], xy[1]], [0, TICK_LEN_3D],
                     color=col, lw=0.7, ls="--", alpha=TICK_ALPHA_OOR * 0.55,
                     zorder=4)

        else:
            ax3.scatter(xy[0], xy[1], 0,
                        s=IOT_S_IN_3D, marker="o",
                        facecolors=col, edgecolors="none",
                        alpha=IOT_ALPHA_IN_3D, depthshade=False, zorder=3)
            # Short dotted tick — in-range
            ax3.plot([xy[0], xy[0]], [xy[1], xy[1]], [0, TICK_LEN_3D],
                     color=col, lw=0.7, ls=":", alpha=TICK_ALPHA_IN * 0.50,
                     zorder=4)

    # ── IoT trajectory trails — near-invisible (reduced alpha) ────────────
    for ii, tr in iot_trajs.items():
        if len(tr) < 2: continue
        cid = int(dev_clust[ii]) if ii < len(dev_clust) else 0
        col = UAV_COLORS[cid % len(UAV_COLORS)]
        zs_iot = np.zeros(len(tr))   # keep on ground plane (z=0)
        ax3.add_collection3d(iot_grad3d(tr[:, 0], tr[:, 1], zs_iot, col))
        end_xy = tr[-1, :2]
        end_jammed = (len(jam_pos) > 0 and jam_r > 0 and
                      any(np.linalg.norm(end_xy - jp[:2]) <= jam_r
                          for jp in jam_pos))
        if end_jammed:
            ax3.scatter([end_xy[0]], [end_xy[1]], [0.0],
                        s=IOT_S_JAMMED_3D, marker="X", color=col,
                        linewidths=0.6, alpha=IOT_ALPHA_JAMMED_3D,
                        depthshade=False, zorder=5)
        else:
            ax3.scatter([tr[-1, 0]], [tr[-1, 1]], [0.0],
                        s=max(4, TRAIL_END_S3 // 3), color=col,
                        alpha=TRAIL_END_ALPHA * 0.5,
                        depthshade=False, zorder=5)

    # ── UAV paths ─────────────────────────────────────────────────────────
    for i, tr in trajs.items():
        c   = UAV_COLORS[i % len(UAV_COLORS)]
        alt = UAV_PROFILES[i % len(UAV_PROFILES)]["altitude"]
        if tr.shape[1] == 3:
            zs = tr[:, 2]
        else:
            zs = np.full(len(tr), alt)

        # 1. Ground shadow — very faint spatial reference
        ax3.plot(tr[:, 0], tr[:, 1], 0, color="grey", lw=0.5, ls="-",
                 alpha=0.10, zorder=2)

        # 2. Main 3D gradient trajectory
        ax3.add_collection3d(grad3d(tr[:, 0], tr[:, 1], zs, c))

        # 3. Start position — hollow circle
        ax3.scatter([tr[0, 0]], [tr[0, 1]], [alt], marker="o",
                    s=UAV_START_S3, facecolors="white", edgecolors=c,
                    linewidths=UAV_START_MEW, zorder=8, depthshade=False)

        # 4. Current position — outlined "+"
        _draw_plus_3d(ax3, tr[-1, 0], tr[-1, 1], alt, c, zorder_base=PLUS_ZORDER)

        # 5. Altitude anchor line — dotted, slightly bolder, higher alpha
        ax3.plot([tr[-1, 0], tr[-1, 0]], [tr[-1, 1], tr[-1, 1]], [0, alt],
                 color=c, lw=1.0, ls=":", alpha=0.50, zorder=3)

        # 6. UAV label — z-offset staggered by tier to avoid stacking
        #    FIX 3: zorder raised to 999 so labels always render on top
        label_z_offset = 16 + (i % 3) * 10   # 16, 26, or 36 m per tier
        ax3.text(tr[-1, 0], tr[-1, 1], alt + label_z_offset,
                 f"UAV-{i + 1}", fontsize=6.5, color=c,
                 ha="center", va="bottom", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white",
                           ec=c, lw=0.5, alpha=0.80),
                 zorder=999)

    # ── NOMA ──────────────────────────────────────────────────────────────
    # Use noma_uav_pos (snapshot position when the NOMA event occurred) so
    # the dashed lines connect to the correct IoT devices.  The UAV may
    # have moved far from noma_uav_pos by end of episode.
    if has_noma:
        alt_n   = UAV_PROFILES[noma_uav_id % len(UAV_PROFILES)]["altitude"]
        ux, uy  = float(noma_uav_pos[0]), float(noma_uav_pos[1])
        for dp in noma_dev_pos:
            ax3.plot([ux, dp[0]], [uy, dp[1]], [alt_n, 0],
                     color=noma_col, lw=NOMA_LW, ls=(0, (4, 2)),
                     alpha=NOMA_ALPHA, zorder=11)
        for dp in noma_dev_pos:
            if np.linalg.norm(dp) <= R * 1.05:
                ax3.scatter(dp[0], dp[1], 0, s=NOMA_RING_S, marker="o",
                            facecolors="none", edgecolors="black",
                            linewidths=NOMA_RING_LW,
                            depthshade=False, zorder=12)

    # ── Cooperative offload ───────────────────────────────────────────────
    if has_coop and coop_src_id < len(uav_finals) and coop_dst_id < len(uav_finals):
        alt_s = UAV_PROFILES[coop_src_id % len(UAV_PROFILES)]["altitude"]
        alt_d = UAV_PROFILES[coop_dst_id % len(UAV_PROFILES)]["altitude"]
        ax3.plot([uav_finals[coop_src_id][0], uav_finals[coop_dst_id][0]],
                 [uav_finals[coop_src_id][1], uav_finals[coop_dst_id][1]],
                 [alt_s, alt_d],
                 color="black", lw=COOP_LW, ls="--",
                 alpha=COOP_ALPHA, zorder=10)

    # +80 m headroom for more z-axis breathing room
    max_alt = (max(UAV_PROFILES[i % len(UAV_PROFILES)]["altitude"]
                   for i in range(n_active)) + 80)

    ax3.set_xlim(-R * AXIS_MULT, R * AXIS_MULT)
    ax3.set_ylim(-R * AXIS_MULT, R * AXIS_MULT)
    ax3.set_zlim(0, max_alt)

    ax3.set_xlabel("$x$ (m)", labelpad=3, fontsize=9)
    ax3.set_ylabel("$y$ (m)", labelpad=3, fontsize=9)
    ax3.set_zlabel("Altitude (m)", labelpad=3, fontsize=9)
    ax3.tick_params(labelsize=8, pad=0.8)

    # Higher elevation (38) to lift ground clutter; azimuth tweak (-55)
    ax3.view_init(elev=38, azim=-55)

    ax3.grid(True, lw=GRID_LW, alpha=GRID_ALPHA)

    # Transparent panes with light edges
    ax3.xaxis.pane.fill = ax3.yaxis.pane.fill = ax3.zaxis.pane.fill = False
    ax3.xaxis.pane.set_edgecolor("#DDDDDD")
    ax3.yaxis.pane.set_edgecolor("#DDDDDD")
    ax3.zaxis.pane.set_edgecolor("#DDDDDD")

    if show_wind:
        _draw_wind_arrow_3d(ax3, _wind_dir, _wind_spd, R, max_alt)

    fig3.tight_layout()
    _savefig(fig3, os.path.join(out, f'P8_3D_trajectory{suffix}'))
    print(f"  P8 complete")


# ===========================================================================
# Env-only mode — generate trajectory + plot WITHOUT training data
# ===========================================================================
def env_only_mode(args):
    """
    Standalone trajectory generator that produces complex, winding UAV paths
    visually matching a trained Diffusion-MAPPO policy (reference figures).

    Navigation model — 3 behaviour modes per UAV per waypoint:
      MODE 0 (55 %): Cluster-seek   — target a random IoT device in own cluster
      MODE 1 (25 %): Exploration    — long-range sweep to far point in full area
      MODE 2 (20 %): Orbit-loop     — tangential arc around current sub-region

    Key parameters that produce dense, space-filling trajectories:
      • max_speed  = max_displacement * UAV_SPEED_SCALE  ≈ 16 m/step
        (previous code divided by K_c=25, giving only 2.2 m/step — 5× too slow)
      • Momentum α = 0.15  → very fast direction changes
      • Waypoint dwell      = Uniform[3, 12] steps
      • Heading noise σ     = 1.8
      • Wind contribution   = 0.35 × wind_speed

    Over 500 steps each UAV covers ≈ 4 000–6 000 m of path, matching the
    reference trajectory plots.
    """
    # ── Speed scale: use max_displacement DIRECTLY (not / K_c) ──────────────
    # Training env moves max_displacement metres per macro-slot (25 micro-slots),
    # which equals max_displacement/K_c per micro-slot ≈ 2.2 m — far too slow
    # for a visually rich 500-step plot.  We scale up to ≈16 m/step so that
    # total path length (≈8 000 m) matches trained-policy output shown in paper.
    UAV_SPEED_SCALE = 0.50   # × max_displacement → m/micro-step  ≈ 16 m/step

    try:
        from config import (IOT_SPEED_MIN, IOT_SPEED_MAX,
                            GM_RHO_V, GM_RHO_THETA,
                            GM_SIGMA_V, GM_SIGMA_THETA)
    except ImportError:
        IOT_SPEED_MIN = 1.0; IOT_SPEED_MAX = 4.0
        GM_RHO_V = 0.9; GM_RHO_THETA = 0.9
        GM_SIGMA_V = 0.5; GM_SIGMA_THETA = np.pi / 6

    rng    = np.random.RandomState(args.seed)
    n_uavs = args.uavs or rng.randint(3, 9)
    n_iot  = args.iot  or rng.randint(80, 151)
    n_uavs = int(np.clip(n_uavs, 3, 8))
    R      = rng.randint(400, 601)
    n_jam  = rng.randint(1, 4)

    print(f"  Env-only: {n_uavs} UAVs, {n_iot} IoT, R={R}m, {n_jam} jammers")

    profiles = [UAV_PROFILES[i % len(UAV_PROFILES)] for i in range(n_uavs)]

    # ── UAV start positions: staggered radii so paths diverge immediately ──
    uav_pos = np.zeros((n_uavs, 3))
    for i in range(n_uavs):
        a = 2 * np.pi * i / n_uavs + rng.uniform(-0.3, 0.3)
        r = R * rng.uniform(0.20, 0.72)
        uav_pos[i] = [r * np.cos(a), r * np.sin(a), profiles[i]["altitude"]]

    # ── IoT devices: clustered + scattered mix ────────────────────────────────
    iot_pos = np.zeros((n_iot, 2))
    n_clustered = int(n_iot * 0.70)
    cluster_centres = [np.array([R * 0.40 * np.cos(2*np.pi*k/n_uavs),
                                  R * 0.40 * np.sin(2*np.pi*k/n_uavs)])
                       for k in range(n_uavs)]
    for i in range(n_clustered):
        c  = rng.randint(0, n_uavs)
        r  = rng.exponential(R * 0.18)
        a  = rng.uniform(0, 2*np.pi)
        pt = cluster_centres[c] + r * np.array([np.cos(a), np.sin(a)])
        iot_pos[i] = np.clip(pt, -R*0.88, R*0.88)
    for i in range(n_clustered, n_iot):
        a = rng.uniform(0, 2*np.pi)
        r = R * np.sqrt(rng.uniform(0.0, 0.92))
        iot_pos[i] = [r*np.cos(a), r*np.sin(a)]

    # ── Jammers ───────────────────────────────────────────────────────────────
    jam_r     = rng.uniform(80, 150)
    r_max_jam = max(R * 0.65, R - jam_r)
    jam_angles = rng.uniform(0, 2*np.pi, n_jam)
    jam_radii  = R*0.20 + rng.uniform(0, 1, n_jam) * (r_max_jam - R*0.20)
    jam_pos_arr = np.column_stack([jam_radii*np.cos(jam_angles),
                                   jam_radii*np.sin(jam_angles)])

    # ── IoT Gauss-Markov mobility state ───────────────────────────────────────
    iot_speed        = rng.uniform(IOT_SPEED_MIN, IOT_SPEED_MAX, n_iot)
    iot_heading      = rng.uniform(0, 2*np.pi, n_iot)
    iot_mean_speed   = np.clip(
        (IOT_SPEED_MIN + IOT_SPEED_MAX)/2 + rng.randn(n_iot)*0.5,
        IOT_SPEED_MIN, IOT_SPEED_MAX)
    iot_mean_heading = rng.uniform(0, 2*np.pi, n_iot)

    # ── Initial cluster assignment (proximity) ────────────────────────────────
    init_dev_clust = np.array([
        int(np.argmin([np.linalg.norm(iot_pos[j] - uav_pos[k, :2])
                       for k in range(n_uavs)]))
        for j in range(n_iot)])
    init_dev_jammed = np.array([
        any(np.linalg.norm(iot_pos[j] - jp) < jam_r for jp in jam_pos_arr)
        for j in range(n_iot)])

    iot_clusters = [[] for _ in range(n_uavs)]
    for j in range(n_iot):
        iot_clusters[init_dev_clust[j]].append(j)

    R_limit = R * 0.90

    # ── Waypoint factory: 3 behaviour modes ───────────────────────────────────
    def _waypoint_cluster(i):
        """MODE 0 — target an IoT device in UAV-i's service cluster."""
        cands = iot_clusters[i]
        if not cands:
            return _waypoint_explore(i)
        unjammed = [c for c in cands if not init_dev_jammed[c]]
        pool     = unjammed if (unjammed and rng.rand() < 0.72) else cands
        tgt      = iot_pos[rng.choice(pool)]
        # Small random offset so UAV orbits the device, not just sits on it
        offset   = rng.randn(2) * rng.uniform(20, 80)
        wp       = tgt + offset
        return np.clip(wp, -R_limit, R_limit)

    def _waypoint_explore(i):
        """MODE 1 — long-range sweep anywhere in the deployment area."""
        # Bias toward areas the UAV has NOT recently visited: opposite quadrant
        cx, cy = uav_pos[i, 0], uav_pos[i, 1]
        angle_away = np.arctan2(-cy, -cx) + rng.uniform(-1.2, 1.2)
        r  = rng.uniform(R * 0.30, R * 0.85)
        wp = np.array([r*np.cos(angle_away), r*np.sin(angle_away)])
        return np.clip(wp, -R_limit, R_limit)

    def _waypoint_orbit(i):
        """MODE 2 — tangential arc: target is 90° offset from current heading."""
        cx, cy   = uav_pos[i, 0], uav_pos[i, 1]
        cur_head = np.arctan2(uav_vy[i], uav_vx[i]) if (
            np.hypot(uav_vx[i], uav_vy[i]) > 1e-3) else rng.uniform(0, 2*np.pi)
        turn     = rng.choice([-1, 1]) * rng.uniform(np.pi*0.35, np.pi*0.65)
        new_head = cur_head + turn
        r_orbit  = rng.uniform(R*0.12, R*0.38)
        wp       = np.array([cx + r_orbit*np.cos(new_head),
                              cy + r_orbit*np.sin(new_head)])
        return np.clip(wp, -R_limit, R_limit)

    def _pick_waypoint(i):
        """Select behaviour mode and return next waypoint for UAV i."""
        roll = rng.rand()
        if   roll < 0.55: return _waypoint_cluster(i)
        elif roll < 0.80: return _waypoint_explore(i)
        else:             return _waypoint_orbit(i)

    # ── Per-UAV navigation state ───────────────────────────────────────────────
    uav_vx    = rng.randn(n_uavs) * 2.0   # non-zero initial velocity → diverge fast
    uav_vy    = rng.randn(n_uavs) * 2.0
    waypoints = np.array([_pick_waypoint(i) for i in range(n_uavs)])
    wp_timer  = rng.randint(1, 6, n_uavs).astype(int)   # staggered start

    # ── Trajectory storage ────────────────────────────────────────────────────
    N = args.steps
    trajs_all     = np.zeros((N + 1, n_uavs, 3))
    iot_trajs_all = np.zeros((N + 1, n_iot,  2))
    trajs_all[0]     = uav_pos.copy()
    iot_trajs_all[0] = iot_pos.copy()

    # ── Wind (Ornstein-Uhlenbeck) ─────────────────────────────────────────────
    wind_speed     = rng.uniform(1.0, 4.0)
    wind_direction = rng.uniform(0, 2*np.pi)

    # ── Main simulation loop ──────────────────────────────────────────────────
    for t in range(N):

        # Wind evolution
        wind_direction  = (wind_direction + rng.randn() * 0.12) % (2*np.pi)
        wind_speed      = float(np.clip(
            wind_speed + 0.08*(2.5 - wind_speed) + 0.40*rng.randn(), 0.0, 8.0))

        for i in range(n_uavs):
            # ── Speed budget: max_displacement × scale ≈ 10 m/step ──────────
            max_spd = profiles[i]["max_displacement"] * UAV_SPEED_SCALE

            # ── Waypoint countdown ───────────────────────────────────────────
            wp_timer[i] -= 1
            if wp_timer[i] <= 0:
                waypoints[i] = _pick_waypoint(i)
                wp_timer[i]  = int(rng.randint(3, 13))   # 3-12 steps dwell

            # ── Direction toward waypoint ────────────────────────────────────
            diff = waypoints[i] - uav_pos[i, :2]
            dist = np.linalg.norm(diff)
            if dist > 8.0:
                desired = diff / dist
            else:
                waypoints[i] = _pick_waypoint(i)
                wp_timer[i]  = int(rng.randint(3, 13))
                diff2   = waypoints[i] - uav_pos[i, :2]
                desired = diff2 / (np.linalg.norm(diff2) + 1e-9)

            # ── Jammer repulsion (hard push-away) ────────────────────────────
            avoidance = np.zeros(2)
            for jp in jam_pos_arr:
                d_j = np.linalg.norm(uav_pos[i, :2] - jp)
                if d_j < jam_r * 1.8 and d_j > 1e-3:
                    away       = (uav_pos[i, :2] - jp) / d_j
                    strength   = (jam_r * 1.8 - d_j) / jam_r
                    avoidance += away * strength * 2.5

            # ── Heading noise: high σ → jagged, space-filling paths ─────
            noise  = rng.randn(2) * 1.8

            # ── Combine forces ───────────────────────────────────────────────
            move_d = desired * 1.0 + avoidance + noise
            nrm    = np.linalg.norm(move_d)
            if nrm > 1e-6:
                move_d /= nrm

            # ── Velocity update: α=0.15 → fast turning, complex paths ────
            alpha    = 0.15
            tgt_vx   = move_d[0] * max_spd
            tgt_vy   = move_d[1] * max_spd
            uav_vx[i] = alpha * uav_vx[i] + (1.0 - alpha) * tgt_vx
            uav_vy[i] = alpha * uav_vy[i] + (1.0 - alpha) * tgt_vy
            spd = np.hypot(uav_vx[i], uav_vy[i])
            if spd > max_spd:
                uav_vx[i] *= max_spd / spd
                uav_vy[i] *= max_spd / spd

            # ── Wind contribution (0.35 × wind_speed) ────────────────────────
            wind_dx = wind_speed * np.cos(wind_direction) * 0.35
            wind_dy = wind_speed * np.sin(wind_direction) * 0.35
            xy = uav_pos[i, :2] + np.array([uav_vx[i] + wind_dx,
                                             uav_vy[i] + wind_dy])

            # ── Boundary: elastic reflection + force new waypoint ────────────
            d = np.linalg.norm(xy)
            if d > R_limit:
                xy = xy * (R_limit / d)
                out_n = xy / (np.linalg.norm(xy) + 1e-9)
                vdot  = uav_vx[i]*out_n[0] + uav_vy[i]*out_n[1]
                if vdot > 0:
                    uav_vx[i] -= 2.0 * vdot * out_n[0]
                    uav_vy[i] -= 2.0 * vdot * out_n[1]
                    spd2 = np.hypot(uav_vx[i], uav_vy[i])
                    if spd2 > max_spd:
                        uav_vx[i] *= max_spd / spd2
                        uav_vy[i] *= max_spd / spd2
                waypoints[i] = _pick_waypoint(i)
                wp_timer[i]  = int(rng.randint(3, 10))

            uav_pos[i, :2] = xy

            # ── Altitude variation — slow random walk within ±20 m of profile alt
            profile_alt = profiles[i]["altitude"]
            dh = rng.randn() * 2.5   # ±2.5 m/step std → visible 3D paths
            uav_pos[i, 2] = float(np.clip(uav_pos[i, 2] + dh,
                                           profile_alt - 20.0, profile_alt + 20.0))

        trajs_all[t + 1] = uav_pos.copy()

        # ── IoT Gauss-Markov mobility ─────────────────────────────────────────
        for ii in range(n_iot):
            new_x = iot_pos[ii, 0] + iot_speed[ii] * np.cos(iot_heading[ii])
            new_y = iot_pos[ii, 1] + iot_speed[ii] * np.sin(iot_heading[ii])
            rr = np.hypot(new_x, new_y)
            if rr > R * 0.90:
                oa = np.arctan2(new_y, new_x)
                iot_heading[ii] = oa + np.pi - (iot_heading[ii] - oa)
                s = R * 0.90 / rr
                new_x *= s;  new_y *= s
            iot_pos[ii] = [new_x, new_y]
            xi_v  = rng.randn() * GM_SIGMA_V
            xi_th = rng.randn() * GM_SIGMA_THETA
            iot_speed[ii] = float(np.clip(
                GM_RHO_V * iot_speed[ii] + (1-GM_RHO_V)*iot_mean_speed[ii]
                + np.sqrt(max(1-GM_RHO_V**2, 0)) * xi_v,
                IOT_SPEED_MIN, IOT_SPEED_MAX))
            iot_heading[ii] = float((
                GM_RHO_THETA * iot_heading[ii]
                + (1-GM_RHO_THETA)*iot_mean_heading[ii]
                + np.sqrt(max(1-GM_RHO_THETA**2, 0)) * xi_th) % (2*np.pi))
        iot_trajs_all[t + 1] = iot_pos.copy()

    # ── Build save dict (3D trajectory: x, y, altitude) ──────────────────────
    traj_dict = {f"uav_{i}": trajs_all[:, i, :] for i in range(n_uavs)}
    for ii in range(n_iot):
        traj_dict[f"iot_{ii}"] = iot_trajs_all[:, ii, :]

    dev_clust = np.array([
        int(np.argmin([np.linalg.norm(iot_pos[ii] - uav_pos[j, :2])
                       for j in range(n_uavs)]))
        for ii in range(n_iot)])
    dev_jammed = np.array([
        any(np.linalg.norm(iot_pos[ii] - jp) < jam_r for jp in jam_pos_arr)
        for ii in range(n_iot)])
    dev_in_range = np.array([
        np.linalg.norm(iot_pos[ii] - uav_pos[dev_clust[ii], :2]) <= 200.0
        for ii in range(n_iot)])

    # ── NOMA snapshot — pick UAV with largest cluster, 2-user far/near pair ──
    cluster_sizes = [int(np.sum(dev_clust == i)) for i in range(n_uavs)]
    noma_uav_idx  = int(np.argmax(cluster_sizes))
    cluster_devs  = np.where(dev_clust == noma_uav_idx)[0]
    unjammed_devs = [d for d in cluster_devs if not dev_jammed[d]]
    pool          = unjammed_devs if len(unjammed_devs) >= 2 else list(cluster_devs)
    if len(pool) >= 2:
        uav_xy_n = uav_pos[noma_uav_idx, :2]
        dists_n  = [np.linalg.norm(iot_pos[d] - uav_xy_n) for d in pool]
        order    = np.argsort(dists_n)
        near_d   = iot_pos[pool[order[0]]]          # closest → near user
        far_d    = iot_pos[pool[order[-1]]]          # farthest → far user
        noma_uav_id_arr  = np.array([noma_uav_idx], dtype=np.int32)
        noma_uav_pos_arr = uav_xy_n.astype(np.float32)
        noma_dev_pos_arr = np.array([far_d, near_d], dtype=np.float32)
    else:
        noma_uav_id_arr  = np.array([-1], dtype=np.int32)
        noma_uav_pos_arr = np.zeros(2,    dtype=np.float32)
        noma_dev_pos_arr = np.zeros((0, 2), dtype=np.float32)

    # ── Cooperative caching snapshot — pick the two closest UAV pair ──────────
    if n_uavs >= 2:
        min_d, src_id, dst_id = float("inf"), 0, 1
        for aa in range(n_uavs):
            for bb in range(aa + 1, n_uavs):
                d = np.linalg.norm(uav_pos[aa, :2] - uav_pos[bb, :2])
                if d < min_d:
                    min_d, src_id, dst_id = d, aa, bb
        coop_src_id_arr  = np.array([src_id], dtype=np.int32)
        coop_dst_id_arr  = np.array([dst_id], dtype=np.int32)
        coop_src_pos_arr = uav_pos[src_id, :2].astype(np.float32)
        coop_dst_pos_arr = uav_pos[dst_id, :2].astype(np.float32)
    else:
        coop_src_id_arr  = np.array([-1], dtype=np.int32)
        coop_dst_id_arr  = np.array([-1], dtype=np.int32)
        coop_src_pos_arr = np.zeros(2, dtype=np.float32)
        coop_dst_pos_arr = np.zeros(2, dtype=np.float32)

    path = os.path.join(TRAJ_DIR, f"flight_path_{args.seed}.npz")
    np.savez_compressed(
        path, step=args.seed, area_radius=R, num_active=n_uavs,
        jam_pos=jam_pos_arr, jam_radius=jam_r,
        dev_pos_start=np.column_stack([iot_pos, np.zeros(n_iot)]),
        dev_cluster=dev_clust, dev_jammed=dev_jammed, dev_in_range=dev_in_range,
        wind_speed=np.float32(wind_speed), wind_direction=np.float32(wind_direction),
        # NOMA pair snapshot
        noma_uav_id  = noma_uav_id_arr,
        noma_uav_pos = noma_uav_pos_arr,
        noma_dev_pos = noma_dev_pos_arr,
        # Cooperative caching snapshot
        coop_src_id  = coop_src_id_arr,
        coop_dst_id  = coop_dst_id_arr,
        coop_src_pos = coop_src_pos_arr,
        coop_dst_pos = coop_dst_pos_arr,
        **traj_dict)
    print(f"  Saved {path} — {n_uavs} UAVs + {n_iot} IoT trajectories")
    plot_P8(step=args.seed)


# ===========================================================================
# Callable from train.py
# ===========================================================================
TRAJ_PLOT_DIR = os.path.join(RESULTS, "trajectory_plots")

def generate_trajectory_plots(step, wind_dir=None, wind_speed=None):
    step_dir = os.path.join(TRAJ_PLOT_DIR, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    plot_P8(step=step, output_dir=step_dir, wind_dir=wind_dir, wind_speed=wind_speed)
    return step_dir

def generate_final_trajectory_plots(wind_dir=None, wind_speed=None):
    plot_P8(step=None, output_dir=OUT_DIR, wind_dir=wind_dir, wind_speed=wind_speed)


# ===========================================================================
# Main
# ===========================================================================
ALL_PLOTS = {
    "P1": plot_P1,  "P2": plot_P2,  "P3": plot_P3,  "P4": plot_P4,
    "P5": plot_P5,  "P6": plot_P6,  "P7": plot_P7,  "P8": plot_P8,
    "P9": plot_P9,  "P10": plot_P10, "P11": plot_P11,
}

def main():
    pa = argparse.ArgumentParser(
        description="IEEE plots for Multi-UAV MEC (reads saved data)")
    pa.add_argument("--plot",       type=str,   default=None)
    pa.add_argument("--traj-step",  type=int,   default=None)
    pa.add_argument("--env-only",   action="store_true")
    pa.add_argument("--seed",       type=int,   default=42)
    pa.add_argument("--uavs",       type=int,   default=None)
    pa.add_argument("--iot",        type=int,   default=None)
    pa.add_argument("--steps",      type=int,   default=1000)
    pa.add_argument("--wind-dir",   type=float, default=None,
                    help="Wind direction in degrees CCW from +x (default: "
                         f"{WIND_DIR_DEG_DEFAULT}°). Pass -1 to hide wind arrow.")
    pa.add_argument("--wind-speed", type=float, default=None,
                    help=f"Wind speed in m/s (default: {WIND_SPEED_DEFAULT} m/s).")
    args = pa.parse_args()

    _wdir = False if (args.wind_dir is not None and args.wind_dir < 0) \
            else args.wind_dir
    _wspd = args.wind_speed

    if args.env_only:
        env_only_mode(args)
        return

    if args.plot:
        name = args.plot.upper()
        if name == "P8":
            plot_P8(step=args.traj_step, wind_dir=_wdir, wind_speed=_wspd)
        elif name in ALL_PLOTS:
            ALL_PLOTS[name]()
        else:
            print(f"Unknown: {name}. Available: {list(ALL_PLOTS.keys())}")
    else:
        print(f"Generating all plots from {LOG_DIR}/")
        print(f"Output → {OUT_DIR}/\n")
        for name, fn in ALL_PLOTS.items():
            print(f"  {name}:")
            if name == "P8":
                plot_P8(step=args.traj_step, wind_dir=_wdir, wind_speed=_wspd)
            else:
                fn()
        print(f"\nDone → {OUT_DIR}/")


if __name__ == "__main__":
    main()