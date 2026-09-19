"""
Training Pipeline — Diffusion-MAPPO  (Multi-method comparison edition)
========================================================================
Supports all six Table I comparison methods via --method flag or by
editing METHOD in src/agent_factory.py:

    python train.py                                  # DMJO (default)
    python train.py --method MAPPO                   # Row 2
    python train.py --method MADDPG                  # Row 3
    python train.py --method MATD3                   # Row 4
    python train.py --method GreedyOffloading        # Row 5
    python train.py --method LocalExecution          # Row 6

All other flags work unchanged for every method:
    python train.py --method MATD3 --total-steps 5000000 --device cuda
    python train.py --method DMJO  --eval --checkpoint results/checkpoints/DMJO/best_model.pt

Video speed controls:
    python train.py --video-frames 960 --video-fps 24    <- 40 s (default, cinema quality)
    python train.py --video-frames 500 --video-fps 24    <- 21 s (shorter)
    python train.py --video-frames 960 --video-fps 30    <- 32 s (30fps)

Disable periodic training videos (speeds up training):
    python train.py --no-training-video
    # OR set ENABLE_TRAINING_VIDEO = False in config.py

    Either approach skips the 3D video at every VIDEO_INTERVAL steps.
    Trajectory plots (2D + 3D) are still generated at every interval.
    Final evaluation videos after training are NEVER skipped.

Smooth motion:
    VIDEO_RENDER_SUBSTEPS (config) sub-frames are rendered per env step.
    Positions are linearly interpolated so UAVs and IoT devices glide
    instead of jumping slot-to-slot.
"""

import argparse
import time
import os
import sys
import io
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from config import CACHE_UPDATE_INTERVAL   # for two-timescale macro-step logging
from src.env import MultiUAVMECEnv, DomainRandomiser
from src.agent_factory import make_agent, is_off_policy, is_rule_based, describe
from src.buffer import RolloutBuffer
from src.metrics import MetricsTracker
from src.logger import TrainingLogger
from plot_ieee import generate_trajectory_plots, generate_final_trajectory_plots

# Graceful fallback for VIDEO_RENDER_SUBSTEPS
try:
    _SUBSTEPS = int(VIDEO_RENDER_SUBSTEPS)
except NameError:
    _SUBSTEPS = 6

# ================================================================
# COLOUR PALETTE
# ================================================================
_C_BG     = "#0d1117"
_C_PANE   = "#161b22"
_C_GRID   = "#30363d"
_C_TEXT   = "#c9d1d9"
_C_ACCENT = "#58a6ff"
# 8 vivid, distinct UAV colors (works on dark video bg AND white paper)
_C_UAV = [
    "#FF6B6B",   # UAV-1: coral red
    "#4ECDC4",   # UAV-2: teal
    "#FFE66D",   # UAV-3: golden yellow
    "#A78BFA",   # UAV-4: soft purple
    "#45B7D1",   # UAV-5: sky blue
    "#F7B731",   # UAV-6: amber
    "#FC5C65",   # UAV-7: rose pink
    "#26DE81",   # UAV-8: bright green
]
_C_JAMMER  = "#ff4444"
_C_REWARD  = "#bc8cff"
_C_LAT_C   = "#f78166"
_C_ENERGY  = "#79c0ff"
_C_TX_C    = "#56d364"
_C_WIND    = "#88ccff"
_C_OFFLOAD = "#00ffcc"
_C_BS      = "#ffd700"   # gold — base station tower + BS-fetch beams


# ================================================================
# DISPLAY WRAPPERS  — store prev + curr for sub-frame interpolation
# ================================================================

class _SmoothUAV:
    """
    Display wrapper for one UAV.

    Stores the UAV's position at the END of the previous env step
    (prev) and the END of the current env step (curr).  lerp_pos(t)
    interpolates between them so the UAV glides smoothly across
    VIDEO_RENDER_SUBSTEPS frames instead of teleporting each slot.

    The visible trail is appended once per env step (at lerp=1.0) so
    it correctly shows the actual flight path, not sub-frame ghosts.
    """
    TRAIL_LEN = 90

    def __init__(self, idx, env):
        uav = env.uavs[idx]
        self.idx  = idx
        p = np.array([float(uav.position[0]),
                      float(uav.position[1]),
                      float(uav.position[2])])
        self.prev  = p.copy()
        self.curr  = p.copy()
        self.trail = [p.copy()]

    def sync(self, env):
        """Called ONCE per env step, after env.step() returns."""
        self.prev = self.curr.copy()
        uav = env.uavs[self.idx]
        self.curr = np.array([float(uav.position[0]),
                              float(uav.position[1]),
                              float(uav.position[2])])
        self.trail.append(self.curr.copy())
        if len(self.trail) > self.TRAIL_LEN:
            self.trail.pop(0)

    def lerp_pos(self, t: float) -> np.ndarray:
        """
        Smoothstep-interpolated 3-D position for sub-frame t in [0, 1].

        Smoothstep: s(t) = t² * (3 - 2t)
        Compared to linear lerp this eases in and out — the UAV
        accelerates from its previous position and decelerates into
        the new one, giving a natural gliding motion.
        """
        s = t * t * (3.0 - 2.0 * t)
        return self.prev + s * (self.curr - self.prev)


class _DisplayIoT:
    """
    Display wrapper for one mobile IoT device.

    env.step() moves every device (Random Waypoint Model) and saves
    dev.prev_position before the move.  sync() reads both, so
    lerp_pos(t) animates the device walking between its pre-slot and
    post-slot positions — perfectly synchronised with the simulation.

    A persistent `trail` list (capped at TRAIL_LEN) records every
    post-step position so the device's full flight path is visible in
    the video as a fading Catmull-Rom spline, exactly like UAV trails.
    """
    TRAIL_LEN = 55   # one full episode worth of positions

    def __init__(self, dev):
        self.dev_id    = dev.id
        p = np.array([float(dev.position[0]),
                      float(dev.position[1]), 0.0])
        self.prev      = p.copy()
        self.curr      = p.copy()
        self.trail     = [p.copy()]
        self.active_tx = False
        self.noma_tx   = False

    def sync(self, env):
        """Called ONCE per env step, after env.step() returns."""
        dev = env.devices[self.dev_id]
        self.prev = np.array([float(dev.prev_position[0]),
                              float(dev.prev_position[1]), 0.0])
        self.curr = np.array([float(dev.position[0]),
                              float(dev.position[1]), 0.0])
        self.trail.append(self.curr.copy())
        if len(self.trail) > self.TRAIL_LEN:
            self.trail.pop(0)

    def lerp_pos(self, t: float) -> np.ndarray:
        """
        Smoothstep-interpolated ground position for sub-frame t in [0, 1].
        Uses the same s(t) = t²(3-2t) easing as _SmoothUAV so IoT device
        footsteps accelerate and decelerate naturally.
        """
        s = t * t * (3.0 - 2.0 * t)
        return self.prev + s * (self.curr - self.prev)


def _make_display_iots(env):
    """Build a fresh _DisplayIoT list from the current env device list."""
    return [_DisplayIoT(dev) for dev in env.devices]


# ================================================================
# NOMA PAIR COMPUTATION  (called once per env step, reused for sub-frames)
# ================================================================

def _apply_tx_flags(display_iots, info):
    """
    Set active_tx / noma_tx on display wrappers using the ground-truth TX
    outcome from info['served_dev_ids'] — produced by env.step().

    Root causes of the old color-lag bug (now fixed):
      1. The old geometry heuristic marked devices active by proximity alone,
         ignoring whether TX actually succeeded (bits >= threshold + energy ok).
         Failed transmissions were shown with a white border even though
         dev.latency was NOT reset to 1 — so color (green) and border disagreed.
      2. The heuristic re-picked NOMA pairs by distance, which could differ
         from the env's bisection-optimised pairing.  The actually-served
         device kept its old Latency color for one full slot.
      3. active_tx flags were derived from geometry AFTER env.step(), so they
         could never correctly reflect what happened INSIDE that step.

    Now flags come directly from info['served_dev_ids']:
      - Only devices where TX genuinely succeeded (latency==1) get active_tx=True.
      - noma flag is True when that agent served 2+ devices (real NOMA pair).
    """
    served = info.get('served_dev_ids', {})
    active_pairs = []

    for mi in display_iots:
        entry = served.get(mi.dev_id)
        if entry:
            mi.active_tx = True
            mi.noma_tx   = entry.get('noma', False)
            active_pairs.append((entry['agent'], mi, mi.noma_tx))
        else:
            mi.active_tx = False
            mi.noma_tx   = False

    return active_pairs


def _compute_active_pairs(env, smooth_uavs, display_iots):
    """Legacy geometry heuristic — superseded by _apply_tx_flags. Kept for
    any external scripts that import this symbol."""
    active_pairs = []
    for s, su in enumerate(smooth_uavs):
        uav_xy = np.array([su.curr[0], su.curr[1]])
        cluster = [
            m for m in display_iots
            if env.devices[m.dev_id].cluster_id == s
            and not env.is_in_jammer_zone(env.devices[m.dev_id])
        ]
        if not cluster:
            continue
        cluster.sort(key=lambda m: np.linalg.norm(m.curr[:2] - uav_xy))
        if len(cluster) >= 2:
            near_m = cluster[0]
            far_m  = cluster[-1]
            far_m.active_tx  = True;  far_m.noma_tx  = True
            near_m.active_tx = True;  near_m.noma_tx = True
            active_pairs.append((s, far_m,  True))
            active_pairs.append((s, near_m, False))
        else:
            cluster[0].active_tx = True
            active_pairs.append((s, cluster[0], True))
    return active_pairs


# ================================================================
# 3-D DRAWING PRIMITIVES
# ================================================================

def _ground_grid(ax):
    R     = AREA_RADIUS
    theta = np.linspace(0, 2 * np.pi, 120)
    for i in range(1, 9):
        r = R * i / 8
        ax.plot(r * np.cos(theta), r * np.sin(theta), np.zeros(120),
                color=_C_GRID, lw=0.35, alpha=0.45)
    for i in range(12):
        a = 2 * np.pi * i / 12
        ax.plot([0, R * np.cos(a)], [0, R * np.sin(a)], [0, 0],
                color=_C_GRID, lw=0.35, alpha=0.45)
    ax.plot(R * np.cos(theta), R * np.sin(theta), np.zeros(120),
            color=_C_ACCENT, lw=1.0, alpha=0.35)


def _altitude_line(ax, x, y, z, color, alpha=0.25):
    ax.plot([x, x], [y, y], [0, z], color=color, lw=0.6, ls=':', alpha=alpha)
    t = np.linspace(0, 2 * np.pi, 30)
    ax.plot(x + 18 * np.cos(t), y + 18 * np.sin(t), np.zeros(30),
            color=color, lw=0.5, alpha=alpha * 0.6)


def _draw_beam(ax, p1, p2, color, lw=1.2, alpha=0.55, pulse_t=None):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=color, lw=lw, alpha=alpha, zorder=5)
    if pulse_t is not None:
        ax.scatter(
            p1[0] + pulse_t * (p2[0] - p1[0]),
            p1[1] + pulse_t * (p2[1] - p1[1]),
            p1[2] + pulse_t * (p2[2] - p1[2]),
            s=28, color=color, alpha=0.95, depthshade=False, zorder=6,
        )


def _jammer_cone(ax, jx, jy, height=90, radius_top=110):
    t = np.linspace(0, 2 * np.pi, 36)
    for frac in np.linspace(0.2, 1.0, 6):
        ax.plot(jx + radius_top * frac * np.cos(t),
                jy + radius_top * frac * np.sin(t),
                np.full(36, height * frac),
                color=_C_JAMMER, lw=0.4, alpha=0.10)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.plot([jx, jx + radius_top * np.cos(angle)],
                [jy, jy + radius_top * np.sin(angle)],
                [0, height], color=_C_JAMMER, lw=0.4, alpha=0.14)


def _coverage_ring(ax, x, y, z, r_ground, color):
    t = np.linspace(0, 2 * np.pi, 60)
    ax.plot(x + r_ground * np.cos(t), y + r_ground * np.sin(t),
            np.zeros(60), color=color, lw=0.7, ls='--', alpha=0.30)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.plot([x, x + r_ground * np.cos(angle)],
                [y, y + r_ground * np.sin(angle)],
                [z, 0], color=color, lw=0.3, alpha=0.12)


def _noma_arc(ax, p1, p2, color, n=24):
    xs = np.linspace(p1[0], p2[0], n + 1)
    ys = np.linspace(p1[1], p2[1], n + 1)
    zs = np.array([20.0 * np.sin(np.pi * i / n) for i in range(n + 1)])
    ax.plot(xs, ys, zs, color=color, lw=0.9, alpha=0.55, ls='--', zorder=4)


def _catmull_rom_spline(points, pts_per_seg=8):
    """
    Generate a smooth Catmull-Rom spline through a sequence of 3-D points.

    Each pair of consecutive control points gets pts_per_seg interpolated
    samples, using the neighbouring points as tangent guides.  The result
    is a C1-continuous curve — positions and tangents match at every
    control point — which looks far smoother than raw line segments.

    Returns
    -------
    curve : np.ndarray  shape (N, 3)
    """
    pts = [np.asarray(p, dtype=float) for p in points]
    n   = len(pts)
    if n < 2:
        return np.array(pts)
    if n == 2:
        ts  = np.linspace(0, 1, pts_per_seg)
        return np.array([pts[0] + t * (pts[1] - pts[0]) for t in ts])

    result = []
    for i in range(n - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(i + 2, n - 1)]
        for j in range(pts_per_seg):
            t  = j / pts_per_seg
            t2 = t * t
            t3 = t2 * t
            # Catmull-Rom basis
            pt = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * t2
                + (-p0 + 3.0*p1 - 3.0*p2 + p3) * t3
            )
            result.append(pt)
    result.append(pts[-1])
    return np.array(result)


def _plot_metric(ax, data, title, color, ylabel, window=20):
    ax.clear()
    ax.set_facecolor(_C_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(_C_GRID)
    ax.tick_params(colors=_C_TEXT, labelsize=6)
    ax.xaxis.label.set_color(_C_TEXT)
    ax.yaxis.label.set_color(_C_TEXT)
    ax.grid(True, color=_C_GRID, lw=0.3, ls='--', alpha=0.5)
    if not data:
        return
    ax.plot(range(len(data)), data, color=color, lw=0.7, alpha=0.35)
    if len(data) >= window:
        ma = np.convolve(data, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(data)), ma, color=color, lw=1.6)
    ax.set_title(title, color=_C_TEXT, fontsize=7.5, pad=3)
    ax.set_ylabel(ylabel, fontsize=6.5)


# ================================================================
# FRAME RENDERER  (pure rendering — no env interaction)
# ================================================================

def _render_frame(env, smooth_uavs, display_iots, active_pairs,
                  lerp_t, frame_idx, total_frames,
                  history, info, train_step, method_name="DMJO"):
    """
    Render one video frame as an H x W x 4 ndarray.

    This function only READS state; it never calls env.step() or reset().
    All positions are lerp-interpolated between prev and curr using lerp_t,
    giving smooth sub-frame motion for both UAVs and IoT devices.

    Parameters
    ----------
    lerp_t      : float in [0, 1] — 0 = start of slot, 1 = end of slot
    frame_idx   : global frame counter (for camera rotation + pulse)
    total_frames: total frames in this video (for camera angle range)
    active_pairs: list from _compute_active_pairs() for this slot
    history     : running metric lists (reward, latency, energy, tx, noma)
    info        : env.step() info dict for this slot
    method_name : string shown in the video title bar
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    import imageio.v2 as imageio

    R     = AREA_RADIUS
    Z_MAX = 210.0

    # ── Precompute interpolated positions ─────────────────────────
    uav_pos  = [su.lerp_pos(lerp_t)  for su in smooth_uavs]
    iot_pos  = [mi.lerp_pos(lerp_t)  for mi in display_iots]

    pulse_t = (frame_idx % 20) / 20.0

    # ── Figure layout ─────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=_C_BG)
    gs  = gridspec.GridSpec(
        4, 3, figure=fig,
        width_ratios=[2.6, 1.0, 1.0],
        height_ratios=[1, 1, 1, 1],
        hspace=0.55, wspace=0.32,
        left=0.01, right=0.97,
        top=0.93, bottom=0.05,
    )
    ax3d   = fig.add_subplot(gs[:, 0], projection='3d')
    ax_r   = fig.add_subplot(gs[0, 1])
    ax_a   = fig.add_subplot(gs[1, 1])
    ax_e   = fig.add_subplot(gs[2, 1])
    ax_tx  = fig.add_subplot(gs[3, 1])
    ax_inf = fig.add_subplot(gs[:, 2])

    # ── 3-D axes appearance ───────────────────────────────────────
    ax3d.set_facecolor(_C_BG)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(_C_GRID)
    ax3d.tick_params(colors=_C_TEXT, labelsize=5.5, pad=0)
    for lbl in [ax3d.xaxis.label, ax3d.yaxis.label, ax3d.zaxis.label]:
        lbl.set_color(_C_TEXT)
    ax3d.set_xlim(-R, R); ax3d.set_ylim(-R, R); ax3d.set_zlim(0, Z_MAX)
    ax3d.set_xlabel("X (m)", fontsize=6.5, labelpad=1)
    ax3d.set_ylabel("Y (m)", fontsize=6.5, labelpad=1)
    ax3d.set_zlabel("Z (m)", fontsize=6.5, labelpad=1)

    # Camera: gentle elevation bob + half-rotation over the full video
    elev = 28 + 5 * np.sin(2 * np.pi * frame_idx / total_frames * 1.5)
    azim = -50 + 180 * frame_idx / total_frames
    ax3d.view_init(elev=elev, azim=azim)

    _ground_grid(ax3d)

    # ── Jammers + interference zones ──────────────────────────────
    jam_radius = getattr(env, 'jammer_interference_radius',
                         JAMMER_INTERFERENCE_RADIUS)
    for jammer in env.jammers:
        jx, jy = float(jammer.position[0]), float(jammer.position[1])
        _jammer_cone(ax3d, jx, jy)
        ax3d.scatter(jx, jy, 0, s=130, color=_C_JAMMER,
                     marker='X', depthshade=False, zorder=9)
        ax3d.text(jx + 22, jy + 22, 4, "JAM",
                  color=_C_JAMMER, fontsize=6, fontweight='bold')
        t_zone = np.linspace(0, 2 * np.pi, 80)
        ax3d.plot(jx + jam_radius * np.cos(t_zone),
                  jy + jam_radius * np.sin(t_zone), np.zeros(80),
                  color=_C_JAMMER, lw=2.0, alpha=0.70, zorder=3)
        for frac in np.linspace(0.1, 1.0, 12):
            ax3d.plot(jx + jam_radius * frac * np.cos(t_zone),
                      jy + jam_radius * frac * np.sin(t_zone), np.zeros(80),
                      color=_C_JAMMER, lw=0.6, alpha=0.06, zorder=2)
        ax3d.text(jx, jy - jam_radius - 15, 0, "NO-COMM ZONE",
                  color=_C_JAMMER, fontsize=5.5, ha='center',
                  fontweight='bold', alpha=0.85)

    # ── Wind arrow ────────────────────────────────────────────────
    wind_spd = getattr(env, 'wind_speed', 0.0)
    wind_dir = getattr(env, 'wind_direction', 0.0)
    shaft_col = (_C_WIND if wind_spd < 4
                 else "#ffaa44" if wind_spd < 6 else '#ff6644')

    wx_c, wy_c, wz_c = R * 0.72, R * 0.72, 180
    shaft_len = 40 + wind_spd * 12
    shaft_lw  = 2.5 + wind_spd * 0.5
    head_len  = 18 + wind_spd * 3

    tail_x = wx_c - shaft_len * 0.5 * np.cos(wind_dir)
    tail_y = wy_c - shaft_len * 0.5 * np.sin(wind_dir)
    tip_x  = wx_c + shaft_len * 0.5 * np.cos(wind_dir)
    tip_y  = wy_c + shaft_len * 0.5 * np.sin(wind_dir)
    ax3d.plot([tail_x, tip_x], [tail_y, tip_y], [wz_c, wz_c],
              color=shaft_col, lw=shaft_lw, alpha=0.90, zorder=10,
              solid_capstyle='round')
    for spread in np.linspace(-1, 1, 7):
        ba = wind_dir + np.pi + spread * 0.45
        ax3d.plot([tip_x, tip_x + head_len * np.cos(ba)],
                  [tip_y, tip_y + head_len * np.sin(ba)],
                  [wz_c, wz_c + 12 * spread * 0.3],
                  color=shaft_col, lw=max(shaft_lw * 0.6, 1.2),
                  alpha=0.80, zorder=10)
    ax3d.scatter(tip_x, tip_y, wz_c, s=50, color=shaft_col,
                 alpha=0.95, depthshade=False, zorder=11)

    ring_r = 22
    ring_t = np.linspace(0, 2 * np.pi, 48)
    ax3d.plot(wx_c + ring_r * np.cos(ring_t),
              wy_c + ring_r * np.sin(ring_t),
              np.full(48, wz_c), color=_C_WIND, lw=1.0, alpha=0.40, zorder=9)
    for clbl, cang in [('N', np.pi/2), ('E', 0), ('S', -np.pi/2), ('W', np.pi)]:
        ax3d.text(wx_c + (ring_r + 8) * np.cos(cang),
                  wy_c + (ring_r + 8) * np.sin(cang),
                  wz_c, clbl, color=_C_WIND, fontsize=5,
                  ha='center', va='center', alpha=0.55, zorder=9)

    deg = np.degrees(wind_dir) % 360
    compass_lbl = ['N','NE','E','SE','S','SW','W','NW'][int((deg+22.5)/45)%8]
    ax3d.text(wx_c, wy_c, wz_c - 22,
              f"{wind_spd:.1f} m/s  {compass_lbl}",
              color=shaft_col, fontsize=7.5, ha='center',
              fontweight='bold', alpha=0.95, zorder=10)

    bar_half = 30
    bar_fill = min(wind_spd / 8.0, 1.0) * bar_half * 2
    bz_bar   = wz_c - 32
    ax3d.plot([wx_c - bar_half, wx_c + bar_half],
              [wy_c, wy_c], [bz_bar, bz_bar],
              color=_C_WIND, lw=4, alpha=0.15, solid_capstyle='round', zorder=9)
    if bar_fill > 1:
        ax3d.plot([wx_c - bar_half, wx_c - bar_half + bar_fill],
                  [wy_c, wy_c], [bz_bar, bz_bar],
                  color=shaft_col, lw=4, alpha=0.70,
                  solid_capstyle='round', zorder=9)

    # Animated wind streamers
    for i in range(6):
        sx   = R * 0.35 * np.cos(i * np.pi / 3)
        sy   = R * 0.35 * np.sin(i * np.pi / 3)
        slen = 25 + wind_spd * 8
        fade = 0.2 + 0.5 * np.sin((frame_idx * 0.12 + i * 0.5) % 1.0 * np.pi)
        ex   = sx + slen * np.cos(wind_dir)
        ey   = sy + slen * np.sin(wind_dir)
        ax3d.plot([sx, ex], [sy, ey], [3, 3],
                  color=_C_WIND, lw=0.9, alpha=fade * 0.4, zorder=2)
        for da in [0.5, -0.5]:
            ax3d.plot([ex, ex - 8 * np.cos(wind_dir + da)],
                      [ey, ey - 8 * np.sin(wind_dir + da)],
                      [3, 3],
                      color=_C_WIND, lw=0.7, alpha=fade * 0.3, zorder=2)

    # ── IoT trajectory trails (ground-level Catmull-Rom splines) ─────
    for mi in display_iots:
        trail = mi.trail
        if len(trail) < 2:
            continue
        dev = env.devices[mi.dev_id]
        if env.is_in_jammer_zone(dev):
            continue
        cid = dev.cluster_id
        col = _C_UAV[cid % len(_C_UAV)] if 0 <= cid < len(_C_UAV) else '#888888'
        ctrl = trail[::max(len(trail) // 20, 1)]
        if len(ctrl) < 2:
            ctrl = trail[-2:]
        spline = _catmull_rom_spline(ctrl, pts_per_seg=4)
        m = len(spline)
        if m < 2:
            continue
        for k in range(m - 1):
            frac      = k / m
            alpha_val = 0.04 + 0.22 * frac
            lw_val    = 0.5  + 0.5  * frac
            ax3d.plot(
                [spline[k][0], spline[k+1][0]],
                [spline[k][1], spline[k+1][1]],
                [0.5, 0.5],
                color=col, lw=lw_val, alpha=alpha_val, zorder=3,
            )

    # ── IoT devices ───────────────────────────────────────────────
    for mi, ip in zip(display_iots, iot_pos):
        dev = env.devices[mi.dev_id]
        in_jammer = env.is_in_jammer_zone(dev)

        if in_jammer:
            ax3d.scatter(ip[0], ip[1], 0, s=8, color='#666666',
                         alpha=0.25, depthshade=False, zorder=2)
        else:
            cid = dev.cluster_id
            col = _C_UAV[cid] if 0 <= cid < len(_C_UAV) else '#888888'
            if mi.active_tx:
                size  = 60 if mi.noma_tx else 42
                alpha = 0.95
                edge  = 'white'
                ew    = 0.9 if mi.noma_tx else 0.5
            else:
                size  = 16
                alpha = 0.45
                edge  = 'none'
                ew    = 0
            ax3d.scatter(ip[0], ip[1], 0, s=size, color=col,
                         edgecolors=edge, linewidths=ew,
                         alpha=alpha, depthshade=False, zorder=4)

    # ── Base Station ──────────────────────────────────────────────
    BS_HEIGHT = 30.0
    bs_x, bs_y = 0.0, 0.0
    ax3d.plot([bs_x, bs_x], [bs_y, bs_y], [0, BS_HEIGHT],
              color=_C_BS, lw=3.5, alpha=0.95, zorder=9, solid_capstyle='round')
    arm = 13
    ax3d.plot([bs_x - arm, bs_x + arm], [bs_y, bs_y],
              [BS_HEIGHT, BS_HEIGHT], color=_C_BS, lw=2.2, alpha=0.95, zorder=9)
    ax3d.plot([bs_x, bs_x], [bs_y - arm, bs_y + arm],
              [BS_HEIGHT, BS_HEIGHT], color=_C_BS, lw=2.2, alpha=0.95, zorder=9)
    beacon_s = 90 + 45 * np.sin(2 * np.pi * pulse_t)
    ax3d.scatter(bs_x, bs_y, BS_HEIGHT, s=beacon_s, color=_C_BS,
                 marker='*', depthshade=False, alpha=0.98, zorder=10)
    halo_t = np.linspace(0, 2 * np.pi, 64)
    ax3d.plot(bs_x + 30 * np.cos(halo_t), bs_y + 30 * np.sin(halo_t),
              np.zeros(64), color=_C_BS, lw=0.9, ls=':', alpha=0.38, zorder=3)
    ax3d.text(bs_x + 20, bs_y + 20, BS_HEIGHT + 10, "BS",
              color=_C_BS, fontsize=9, fontweight='bold', alpha=0.98, zorder=10)
    for uid in info.get('bs_fetch_uavs', []):
        up = uav_pos[uid]
        ax3d.plot([bs_x, up[0]], [bs_y, up[1]], [BS_HEIGHT, up[2]],
                  color=_C_BS, lw=1.5, ls='--', alpha=0.70, zorder=6)
        pt_bs = (frame_idx % 18) / 18.0
        ax3d.scatter(
            bs_x + pt_bs * (up[0] - bs_x),
            bs_y + pt_bs * (up[1] - bs_y),
            BS_HEIGHT + pt_bs * (up[2] - BS_HEIGHT),
            s=30, color=_C_BS, alpha=0.92, depthshade=False, zorder=8,
        )

    # ── UAVs ──────────────────────────────────────────────────────
    for s, (su, up) in enumerate(zip(smooth_uavs, uav_pos)):
        col = _C_UAV[s % len(_C_UAV)]
        px, py, pz = up[0], up[1], up[2]

        _coverage_ring(ax3d, px, py, pz, r_ground=pz * 0.85, color=col)
        _altitude_line(ax3d, px, py, pz, col, alpha=0.30)

        # Catmull-Rom trail
        trail = su.trail
        if len(trail) >= 2:
            ctrl   = trail[::max(len(trail) // 25, 1)]
            if len(ctrl) < 2:
                ctrl = trail[-2:]
            spline = _catmull_rom_spline(ctrl, pts_per_seg=6)
            m      = len(spline)
            for k in range(m - 1):
                frac = k / m
                ax3d.plot(
                    [spline[k][0], spline[k+1][0]],
                    [spline[k][1], spline[k+1][1]],
                    [spline[k][2], spline[k+1][2]],
                    color=col,
                    lw=0.6 + 1.4 * frac,
                    alpha=0.08 + 0.52 * frac,
                    zorder=5,
                )

        ax3d.scatter(px, py, pz, s=200, color=col, marker='^',
                     edgecolors='white', linewidths=0.9,
                     depthshade=False, zorder=10)
        ax3d.text(px + 15, py + 15, pz + 8, f"U{s+1}",
                  color=col, fontsize=8, fontweight='bold', zorder=11)

        # TX beams to active IoT
        for _, mi, is_noma in active_pairs:
            if not mi.active_tx:
                continue
            dev = env.devices[mi.dev_id]
            if dev.cluster_id != s:
                continue
            ip_now = mi.lerp_pos(lerp_t)
            if is_noma:
                _noma_arc(ax3d, [px, py, pz], [ip_now[0], ip_now[1], 0], col)
            else:
                _draw_beam(ax3d, [px, py, pz], [ip_now[0], ip_now[1], 0],
                           col, lw=1.0, alpha=0.50, pulse_t=pulse_t)

    # ── Cooperative offload links (A2A) ───────────────────────────
    drawn_links = set()
    for src_id, dst_id in info.get('coop_offload_links', []):
        key = (min(src_id, dst_id), max(src_id, dst_id))
        if key in drawn_links:
            continue
        drawn_links.add(key)
        up_src = uav_pos[src_id]
        up_dst = uav_pos[dst_id]
        ax3d.plot([up_src[0], up_dst[0]], [up_src[1], up_dst[1]],
                  [up_src[2], up_dst[2]],
                  color=_C_OFFLOAD, lw=2.5, ls='--', alpha=0.80, zorder=7)
        pt_off = (frame_idx % 14) / 14.0
        ax3d.scatter(
            up_src[0] + pt_off * (up_dst[0] - up_src[0]),
            up_src[1] + pt_off * (up_dst[1] - up_src[1]),
            up_src[2] + pt_off * (up_dst[2] - up_src[2]),
            s=40, color=_C_OFFLOAD, alpha=0.95, depthshade=False, zorder=8,
        )
        mx = (up_src[0] + up_dst[0]) / 2
        my = (up_src[1] + up_dst[1]) / 2
        mz = (up_src[2] + up_dst[2]) / 2 + 12
        ax3d.text(mx, my, mz, "OFFLOAD", color=_C_OFFLOAD,
                  fontsize=5.5, ha='center', fontweight='bold', alpha=0.85)

    # ── Title — includes method name ──────────────────────────────
    fig.suptitle(
        f"{method_name}  |  Multi-UAV NOMA-MEC  "
        f"|  Train step {train_step:,}  |  Slot {env.t:>3d}/{NUM_TIME_SLOTS}",
        color=_C_ACCENT, fontsize=10, fontweight='bold', y=0.97
    )

    # ── Metric subplots ───────────────────────────────────────────
    _plot_metric(ax_r,  history['reward'],  "Mean Reward",       _C_REWARD, "R")
    _plot_metric(ax_a,  history['latency'], "Avg Latency (s)",   _C_LAT_C,  "Latency")
    _plot_metric(ax_e,  history['energy'],  "UAV Energy (J)",    _C_ENERGY, "J")
    _plot_metric(ax_tx, history['tx'],      "Successful Tx",     _C_TX_C,   "Tx")

    # ── Info panel ────────────────────────────────────────────────
    ax_inf.set_facecolor(_C_PANE)
    for sp in ax_inf.spines.values():
        sp.set_edgecolor(_C_GRID)
    ax_inf.axis('off')

    total_tx   = sum(history['tx'])
    total_noma = sum(history['noma'])
    avg_r      = float(np.mean(history['reward'][-30:])) if history['reward'] else 0.0
    avg_latency= float(np.mean(history['latency'][-30:])) if history['latency'] else 0.0
    c_local    = info.get('cache_local_hits', 0)
    c_coop     = info.get('cache_coop_hits', 0)
    c_bs       = info.get('cache_bs_fetches', 0)
    c_cov      = info.get('cache_coverage', 0.0)

    lines = [
        ("SYSTEM STATUS",                           _C_ACCENT, 9.5, True),
        ("",                                        _C_TEXT,   2,   False),
        (f"Method       {method_name}",             _C_ACCENT, 7.5, True),
        (f"Train Step   {train_step:>10,}",         _C_TEXT,   8,   False),
        (f"Slot         {env.t:>4d}/{NUM_TIME_SLOTS}", _C_TEXT, 8,  False),
        (f"IoT Devices  {env.num_users:>6d}",       _C_TEXT,   8,   False),
        (f"Successful Tx {total_tx:>5d}",           _C_TX_C,   8,   False),
        (f"NOMA Pairs   {total_noma:>6d}",          _C_ACCENT, 8,   False),
        (f"Avg Reward   {avg_r:>9.3f}",             _C_REWARD, 8,   False),
        (f"Avg Latency (30) {avg_latency:>8.2f}",   _C_LAT_C,  8,   False),
        ("",                                        _C_TEXT,   2,   False),
        ("CACHING (this slot)",                     _C_OFFLOAD, 9,  True),
        ("",                                        _C_TEXT,   2,   False),
        (f"Local hits   {c_local:>6d}",             _C_OFFLOAD, 8,  False),
        (f"Coop offloads {c_coop:>5d}",             _C_OFFLOAD, 8,  False),
        (f"BS fetch     {c_bs:>7d}",                _C_BS,      8,  False),
        (f"Coverage     {c_cov:>7.0%}",             _C_TEXT,   8,   False),
        ("",                                        _C_TEXT,   2,   False),
        ("HETEROGENEOUS UAVS",                      _C_ACCENT, 9,   True),
        ("",                                        _C_TEXT,   2,   False),
    ]
    for i, u in enumerate(env.uavs):
        col = _C_UAV[i % len(_C_UAV)]
        lines.append((f"U{i+1} {u.cpu_freq/1e9:.1f}G "
                       f"{u.bandwidth/1e6:.0f}MHz "
                       f"c={u.cache_capacity}", col, 7, False))
    lines += [
        ("",                                        _C_TEXT,   2,   False),
        ("ENVIRONMENT",                             _C_ACCENT, 9,   True),
        ("",                                        _C_TEXT,   2,   False),
        ("BS @ (0,0)  central tower",               _C_BS,     8,   False),
        (f"Jam Zone R   {jam_radius:.0f} m",        _C_JAMMER, 8,   False),
        (f"Wind         {wind_spd:.1f} m/s",        _C_WIND,   8,   False),
        ("",                                        _C_TEXT,   2,   False),
        ("LEGEND",                                  _C_ACCENT, 9,   True),
        ("",                                        _C_TEXT,   2,   False),
    ]
    for uid in range(env.num_active):
        c = _C_UAV[uid % len(_C_UAV)]
        lines.append((f"^  UAV-{uid+1}  +  IoT cluster", c, 7.5, False))
    lines += [
        ("*  Base Station (BS)",                    _C_BS,      7.5, False),
        ("·  Jammed IoT (gray, inside zone)",       "#666666",  7.5, False),
        ("O  IoT actively transmitting",            '#ffffff',  7.5, False),
        ("~  IoT path trail (ground, fading)",      "#aaaaaa",  7.5, False),
        ("-- Cooperative offload (A2A)",            _C_OFFLOAD, 7.5, False),
        ("-- BS fetch beam",                        _C_BS,      7.5, False),
        ("~~ NOMA pair arc",                        _C_UAV[1], 7.5, False),
        ("->  Wind direction",                      _C_WIND,    7.5, False),
    ]

    y = 0.98
    for text, col, size, bold in lines:
        ax_inf.text(0.06, y, text, color=col, fontsize=size,
                    transform=ax_inf.transAxes,
                    va='top', fontfamily='monospace',
                    fontweight='bold' if bold else 'normal')
        y -= 0.040 if size >= 8.5 else (0.032 if size >= 7 else 0.013)

    # ── Render to array ───────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor=_C_BG)
    buf.seek(0)
    import imageio.v2 as _iio
    img = _iio.imread(buf)
    plt.close(fig)
    return img


# ================================================================
# VIDEO RECORDER
# ================================================================

class VideoRecorder3D:
    """
    Integrated 3-D video recorder with smooth sub-frame motion.

    For each real env time-slot, VIDEO_RENDER_SUBSTEPS frames are rendered
    with UAV and IoT positions linearly interpolated from the previous
    slot's final position to the new position.  This removes the jarring
    slot-to-slot jump while keeping the simulation physics unchanged.

    IoT device positions are animated directly from dev.prev_position
    (saved by env._update_device_positions() before the move) to
    dev.position (after the move), so the walk is perfectly synced.

    Sole owner of env.reset() — never called inside _render_frame.
    Rebuilds display wrappers after every episode boundary so stale
    device-id references are impossible even when num_users changes.
    """

    def __init__(self, output_dir='results/videos',
                 total_frames=160, fps=10):
        self.output_dir   = output_dir
        self.total_frames = total_frames
        self.fps          = fps
        os.makedirs(output_dir, exist_ok=True)

    def record(self, env, agent, train_step, tag='', method_name="DMJO"):
        """
        Record one video.

        Parameters
        ----------
        env         : MultiUAVMECEnv (reset inside this method)
        agent       : any agent from agent_factory
        train_step  : int — shown in title and filename
        tag         : optional suffix for filename
        method_name : shown in video title bar
        """
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("[VideoRecorder3D] imageio not available — skipping.")
            return None

        suffix   = f"_{tag}" if tag else ""
        filepath = os.path.join(
            self.output_dir,
            f"3d_step{train_step}{suffix}.mp4"
        )

        env_steps           = VIDEO_PLOT_STEPS
        total_render_frames = env_steps * _SUBSTEPS

        obs, state   = env.reset()
        smooth_uavs  = [_SmoothUAV(i, env)  for i in range(env.num_uavs)]
        display_iots = _make_display_iots(env)
        history = {'reward': [], 'latency': [], 'energy': [], 'tx': [], 'noma': []}
        done    = False

        print(f"  [3D-Video] {env_steps} env steps x {_SUBSTEPS} sub-frames "
              f"= {total_render_frames} frames "
              f"({total_render_frames / self.fps:.0f}s @ {self.fps}fps) ...")

        writer    = None
        frame_idx = 0
        output_H  = None
        output_W  = None

        for step in range(env_steps):
            if done:
                obs, state   = env.reset()
                smooth_uavs  = [_SmoothUAV(i, env)  for i in range(env.num_uavs)]
                display_iots = _make_display_iots(env)

            with torch.no_grad():
                actions, _, _ = agent.get_actions(obs, state, num_active=env.num_active)

            obs, state, rewards, done, info = env.step(actions)

            for su in smooth_uavs:
                su.sync(env)
            for mi in display_iots:
                mi.sync(env)

            history['reward'].append(float(np.mean(rewards)))
            history['latency'].append(float(info['avg_latency']))
            history['energy'].append(float(info['total_energy_uav']))
            history['tx'].append(int(info['successful_tx']))
            history['noma'].append(int(info['noma_pairs_formed']))

            active_pairs = _apply_tx_flags(display_iots, info)

            for sub in range(_SUBSTEPS):
                lerp_t = (sub + 1) / _SUBSTEPS

                frame = _render_frame(
                    env, smooth_uavs, display_iots, active_pairs,
                    lerp_t, frame_idx, total_render_frames,
                    history, info, train_step, method_name=method_name,
                )

                if writer is None:
                    H, W     = frame.shape[:2]
                    output_H = H - (H % 2)
                    output_W = W - (W % 2)
                    writer   = imageio.get_writer(
                        filepath, fps=self.fps, macro_block_size=1)

                writer.append_data(frame[:output_H, :output_W])
                frame_idx += 1

        if writer is not None:
            writer.close()

        print(f"  [3D-Video] Saved -> {filepath}")
        return filepath


# ================================================================
# TRAINING LOOP
# ================================================================
def train(args):
    import config as _cfg

    method     = args.method
    device     = torch.device(args.device)
    method_desc = describe(method)

    enable_training_video = ENABLE_TRAINING_VIDEO and not args.no_training_video

    print(f"{'='*60}")
    print(f"  Multi-UAV MEC Training")
    print(f"  Method      : {method}  —  {method_desc}")
    print(f"  Ablation    : {_cfg.ABLATION_MODE}")
    print(f"  Device      : {device}")
    print(f"  Steps       : {args.total_steps:,}")
    if enable_training_video:
        print(f"  Video every : {VIDEO_INTERVAL:,} steps  "
              f"({args.video_frames} frames @ {args.video_fps}fps "
              f"= {args.video_frames/args.video_fps:.0f}s, "
              f"{_SUBSTEPS} sub-frames/step)")
    else:
        print(f"  Training videos : DISABLED")
        print(f"  Trajectory plots: ENABLED (2D + 3D every {VIDEO_INTERVAL:,} steps)")
    print(f"  Domain rand : every {DOMAIN_RAND_INTERVAL:,} steps")
    print(f"{'='*60}\n")

    randomiser = DomainRandomiser(np.random.RandomState(args.seed))
    env        = MultiUAVMECEnv(seed=args.seed)
    obs, state = env.reset()

    # ── Instantiate the chosen agent ────────────────────────────────────
    agent = make_agent(
        method     = method,
        obs_dim    = env.local_obs_dim,
        state_dim  = env.global_state_dim,
        action_dim = env.agent_action_dim,
        n_agents   = env.n_agents,
        device     = device,
        share_actor= True,
    )

    # ── Per-method capability flags ───────────────────────────────────
    _off_policy  = is_off_policy(agent)   # MADDPG / MATD3 → need push_transition
    _rule_based  = is_rule_based(agent)   # Greedy / Local  → no gradient updates
    _has_diffusion = hasattr(agent, 'actor') and hasattr(
        getattr(agent, 'actor', None), 'diffusion_blend')
    _has_anneal    = hasattr(agent, 'actor') and hasattr(
        getattr(agent, 'actor', None), 'anneal_blend')

    print(f"Agent: {agent.METHOD}  |  off_policy={_off_policy}  "
          f"rule_based={_rule_based}  diffusion={_has_diffusion}\n")

    # ── Rollout buffer (used by on-policy agents) ──────────────────────
    buffer = RolloutBuffer(
        rollout_length = MAPPO_ROLLOUT_LENGTH,
        n_agents       = env.n_agents,
        obs_dim        = env.local_obs_dim,
        state_dim      = env.global_state_dim,
        action_dim     = env.agent_action_dim,
    )

    video_recorder = VideoRecorder3D(
        output_dir   = 'results/videos',
        total_frames = args.video_frames,
        fps          = args.video_fps,
    )

    tracker  = MetricsTracker()
    logger   = TrainingLogger()

    log_path = f'{_cfg.LOG_DIR}/training_log.jsonl'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, 'w')

    global_step         = 0
    _next_macro_log     = 10_000
    episode_count       = 0
    ep_rewards          = np.zeros(env.n_agents)
    ep_length           = 0
    best_reward         = -float('inf')
    domain_change_count = 0
    start_time          = time.time()

    print(f"Env: obs_dim={env.local_obs_dim}, "
          f"state_dim={env.global_state_dim}, "
          f"action_dim={env.agent_action_dim}, "
          f"agents={env.n_agents}\n")

    # ── Saved transition for off-policy push ──────────────────────────
    _prev_obs   = obs.copy()
    _prev_state = state.copy()

    while global_step < args.total_steps:

        buffer.reset()
        domain_rand_triggered = False

        for t in range(MAPPO_ROLLOUT_LENGTH):
            with torch.no_grad():
                actions, log_probs, values = agent.get_actions(
                    obs, state, num_active=env.num_active)

            next_obs, next_state, rewards, done, info = env.step(actions)



            # ── On-policy buffer insert ────────────────────────────────
            buffer.insert(obs, state, actions, log_probs, rewards,
                          float(done), values)

            # ── Off-policy replay push (MADDPG / MATD3) ───────────────
            if _off_policy:
                agent.push_transition(
                    obs, state, actions, rewards,
                    next_obs, next_state, float(done))

            obs   = next_obs
            state = next_state
            ep_rewards += rewards
            ep_length  += 1
            global_step += 1

            # ── Two-timescale: log every 10,000 steps ────────────────
            if info.get('is_macro_step', False) and global_step >= _next_macro_log:
                print(f"\n    [Macro-{info['macro_t']:>4d}] step={global_step:,} | UAV positions updated | caches refreshed | K_c={info['macro_interval_len']}", flush=True)
                _next_macro_log += 10_000

            tracker.log_step(
                reward      = rewards.mean(),
                avg_latency = info['avg_latency'],
                uav_energy  = info['total_energy_uav'],
            )

            logger.log_step(global_step, info, float(rewards.mean()))

            # ── Progress log ──────────────────────────────────────────
            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                sps     = global_step / max(elapsed, 1)
                pct     = global_step / args.total_steps * 100
                # macro_t is global — read directly from env, not from info snapshot.
                # info snapshot is from the last step which may be end-of-episode.
                macro_t = env.macro_t
                K_c     = info.get('macro_interval_len', CACHE_UPDATE_INTERVAL)
                print(f"  [{global_step:>9,}/{args.total_steps:,}] "
                      f"{pct:5.1f}% | ep={episode_count} | "
                      f"SPS={sps:.0f} | "
                      f"Latency={info['avg_latency']:.1f} | "
                      f"NOMA={info['noma_pairs_formed']} | "
                      f"tx={info['successful_tx']} | "
                      f"macro_t={macro_t} | "
                      f"best_R={best_reward:.1f}")

            # ── Checkpoint ────────────────────────────────────────────
            if global_step % CHECKPOINT_INTERVAL == 0:
                ckpt_path = f'{_cfg.CHECKPOINT_DIR}/model_step{global_step}.pt'
                agent.save(ckpt_path)
                print(f"  [Checkpoint] {ckpt_path}")

            # ── Video + trajectory ────────────────────────────────────
            if global_step % VIDEO_INTERVAL == 0:
                print(f"\n  [Video+Trajectory] Triggered at step {global_step:,}")

                if enable_training_video:
                    vid_env = MultiUAVMECEnv(
                        seed=args.seed + global_step, domain_cfg=env.domain_cfg)
                    video_recorder.record(
                        vid_env, agent, global_step, method_name=method)
                else:
                    print(f"  [Video] Skipped (ENABLE_TRAINING_VIDEO=False)")

                traj_file = logger.save_trajectory_episode(
                    global_step, env, n_steps=VIDEO_PLOT_STEPS, agent=agent)
                print(f"  [Trajectory] Saved {traj_file}")

                try:
                    step_dir = generate_trajectory_plots(global_step)
                    print(f"  [Trajectory Plots] Saved to {step_dir}")
                except Exception as e:
                    print(f"  [Trajectory Plots] Warning: {e}")

            # ── Domain randomisation ──────────────────────────────────
            if global_step % DOMAIN_RAND_INTERVAL == 0:
                domain_cfg = randomiser.sample()
                obs, state = env.reset(domain_cfg=domain_cfg)
                domain_change_count += 1
                ep_rewards = np.zeros(env.n_agents)
                ep_length  = 0
                tracker.reset()
                print(f"\n  [Domain Rand #{domain_change_count}] step={global_step:,}")
                print(f"    users={domain_cfg['num_users']}, "
                      f"jammers={domain_cfg['num_jammers']}, "
                      f"radius={domain_cfg['area_radius']:.0f}m, "
                      f"wind={domain_cfg['wind_speed']:.1f}m/s")
                domain_rand_triggered = True
                break

            if done:
                episode_count  += 1
                mean_ep_reward  = ep_rewards.mean()

                logger.log_episode_end(global_step, episode_count)

                if episode_count % 10 == 0 or mean_ep_reward > best_reward:
                    elapsed   = time.time() - start_time
                    sps       = global_step / max(elapsed, 1)
                    summary   = tracker.get_episode_summary()
                    log_entry = {
                        'step':              global_step,
                        'episode':           episode_count,
                        'reward':            float(mean_ep_reward),
                        'avg_latency':       float(summary['avg_latency']),
                        'total_energy':      float(summary['total_energy']),
                        'sps':               float(sps),
                        'domain_changes':    domain_change_count,
                        'noma_pairs_formed': int(info.get('noma_pairs_formed', 0)),
                        'successful_tx':     int(info.get('successful_tx', 0)),
                        'num_users':         int(env.num_users),
                        'num_active_uavs':   int(env.num_active),
                        'num_jammers':       int(env.num_jammers),
                        'jammer_radius':     float(env.jammer_interference_radius),
                        'wind_speed':        float(env.wind_speed),
                        'local_compute':     int(info.get('local_compute', 0)),
                        'uav_compute':       int(info.get('uav_compute', 0)),
                        'compute_fail':      int(info.get('compute_fail', 0)),
                        'cache_local_hits':  int(info.get('cache_local_hits', 0)),
                        'cache_coop_hits':   int(info.get('cache_coop_hits', 0)),
                        'cache_bs_fetches':  int(info.get('cache_bs_fetches', 0)),
                        'cache_coverage':    float(info.get('cache_coverage', 0)),
                        'jammer_blocked':    int(info.get('jammer_blocked', 0)),
                        # Method tag for per-method analysis
                        'method':            method,
                    }
                    log_file.write(json.dumps(log_entry) + '\n')
                    log_file.flush()

                if mean_ep_reward > best_reward:
                    best_reward = mean_ep_reward
                    agent.save(f'{_cfg.CHECKPOINT_DIR}/best_model.pt')

                ep_rewards = np.zeros(env.n_agents)
                ep_length  = 0
                tracker.reset()
                obs, state = env.reset()

            if global_step >= args.total_steps:
                break

        # ── Skip policy update on domain-rand rollout ─────────────────
        if domain_rand_triggered:
            domain_rand_triggered = False
            continue

        # ── Policy update ─────────────────────────────────────────────
        if not _rule_based:
            if not _off_policy:
                # On-policy (DMJO / MAPPO): GAE + PPO update
                with torch.no_grad():
                    last_values = agent.get_values(state)
                buffer.compute_gae(last_values, float(done))

            update_info = agent.update(buffer)

            # ── Anneal diffusion blend (DMJO only) ────────────────────
            if _has_anneal:
                progress = global_step / args.total_steps
                if agent.share_actor:
                    agent.actor.anneal_blend(progress)
                else:
                    for a in agent.actors:
                        a.anneal_blend(progress)

            # ── Update log ────────────────────────────────────────────
            if global_step % LOG_INTERVAL < MAPPO_ROLLOUT_LENGTH:
                # Build log line; include diffusion_loss only when present
                diff_str = ""
                if _has_diffusion and 'diffusion_loss' in update_info:
                    diff_str = f"diff={update_info['diffusion_loss']:.4f} "

                blend_str = ""
                if _has_diffusion:
                    blend = (agent.actor.diffusion_blend
                             if agent.share_actor
                             else agent.actors[0].diffusion_blend)
                    blend_str = f"blend={blend:.2f}"

                print(f"    [Update/{method}] "
                      f"actor={update_info['actor_loss']:.4f} "
                      f"critic={update_info['critic_loss']:.4f} "
                      f"entropy={update_info['entropy']:.4f} "
                      f"{diff_str}"
                      f"kl={update_info['approx_kl']:.4f} "
                      f"{blend_str}".rstrip())

    agent.save(f'{_cfg.CHECKPOINT_DIR}/final_model.pt')
    log_file.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete!  [{method}]")
    print(f"  Total steps    : {global_step:,}")
    print(f"  Episodes       : {episode_count}")
    print(f"  Time           : {elapsed/3600:.1f}h")
    print(f"  Best reward    : {best_reward:.2f}")
    print(f"  Domain changes : {domain_change_count}")
    print(f"{'='*60}")

    if getattr(args, 'smoke_test', False):
        print("\n  [smoke-test] Skipping final videos, plots, and sweeps.")
        return

    print("\n  Recording final evaluation video (default env) ...")
    eval_env = MultiUAVMECEnv(seed=0)
    video_recorder.record(eval_env, agent, global_step,
                          tag='final_default', method_name=method)

    print("  Recording final evaluation video (randomised env) ...")
    rand_cfg      = randomiser.sample()
    eval_env_rand = MultiUAVMECEnv(seed=1, domain_cfg=rand_cfg)
    video_recorder.record(eval_env_rand, agent, global_step,
                          tag='final_random', method_name=method)

    print("  Saving final trajectory snapshot ...")
    logger.save_trajectory_episode(global_step, env,
                                   n_steps=VIDEO_PLOT_STEPS, agent=agent)

    print("  Generating final 2D + 3D trajectory plots ...")
    try:
        generate_final_trajectory_plots()
        print("  Final trajectory plots saved to results/plots/")
    except Exception as e:
        print(f"  Warning: {e}")

    print("\n  Running parameter sweeps for all plots (P2-P11) ...")
    logger.run_and_save_sweeps(MultiUAVMECEnv,
                               n_episodes=3, n_steps=VIDEO_PLOT_STEPS)
    print("  Sweeps complete. Run 'python plot_ieee.py' to generate all plots.")


# ================================================================
# EVALUATION
# ================================================================
def evaluate(args):
    """Load checkpoint and produce evaluation videos."""
    method = args.method
    device = torch.device(args.device)
    env    = MultiUAVMECEnv(seed=0)

    agent = make_agent(
        method     = method,
        obs_dim    = env.local_obs_dim,
        state_dim  = env.global_state_dim,
        action_dim = env.agent_action_dim,
        n_agents   = env.n_agents,
        device     = device,
    )
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}  [{method}]")

    recorder = VideoRecorder3D(
        output_dir   = 'results/videos',
        total_frames = args.video_frames,
        fps          = args.video_fps,
    )
    recorder.record(env, agent, train_step=0,
                    tag='eval_default', method_name=method)

    randomiser = DomainRandomiser()
    for i in range(3):
        cfg      = randomiser.sample()
        eval_env = MultiUAVMECEnv(seed=100 + i, domain_cfg=cfg)
        recorder.record(eval_env, agent, train_step=i + 1,
                        tag=f'eval_rand{i}', method_name=method)

    print("Evaluation complete — videos saved to results/videos/")


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-UAV MEC Training — Table I comparison methods')

    # ── Method selection ────────────────────────────────────────────
    parser.add_argument(
        '--method', type=str, default='DMJO',
        choices=['DMJO', 'MAPPO', 'MADDPG', 'MATD3',
                 'GreedyOffloading', 'LocalExecution'],
        help=(
            'Comparison method (Table I of the paper):\n'
            '  DMJO             — Diffusion-enhanced joint offloading (Ours, Row 1)\n'
            '  MAPPO            — Multi-agent PPO, no diffusion        (Row 2)\n'
            '  MADDPG           — Deterministic multi-agent AC          (Row 3)\n'
            '  MATD3            — Double-Q deterministic MARL           (Row 4)\n'
            '  GreedyOffloading — Nearest-UAV, cache-unaware            (Row 5)\n'
            '  LocalExecution   — All tasks executed locally at WD      (Row 6)'
        ),
    )

    # ── Standard training args ──────────────────────────────────────
    parser.add_argument('--total-steps', type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument('--device',      type=str, default='cpu')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--eval',        action='store_true')
    parser.add_argument('--checkpoint',  type=str,
                        default='results/checkpoints/best_model.pt')
    parser.add_argument('--video-frames', type=int, default=960,
                        help='Total rendered frames per video')
    parser.add_argument('--video-fps',   type=int, default=24)
    parser.add_argument('--no-training-video', action='store_true',
                        help='Disable periodic 3D videos during training')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Skip post-training videos, plots, and sweeps')

    # ── Ablation args (DMJO internal ablations — Table V) ───────────
    parser.add_argument(
        '--ablation', type=str, default='full',
        choices=['full', 'single_timescale', 'no_diffusion',
                 'no_mappo_traj', 'no_coop_cache', 'minimal'],
        help=(
            'DMJO ablation variant (Table V). Only meaningful with --method DMJO:\n'
            '  full             — complete DMJO framework (default)\n'
            '  single_timescale — K_C=1, cache updated every slot (row 2)\n'
            '  no_diffusion     — Gaussian-only actor, no denoiser (row 3)\n'
            '  no_mappo_traj    — fixed circular UAV orbits (row 4)\n'
            '  no_coop_cache    — local + BS only, no A2A tier (row 5)\n'
            '  minimal          — no diffusion AND no coop cache (row 6)'
        ),
    )

    args = parser.parse_args()

    # ── Ablation resolver ────────────────────────────────────────────
    import config as _cfg

    _ABLATION_MAP = {
        'full':             (True,  True,  True,  _cfg.CACHE_UPDATE_INTERVAL),
        'single_timescale': (True,  True,  True,  1),
        'no_diffusion':     (False, True,  True,  _cfg.CACHE_UPDATE_INTERVAL),
        'no_mappo_traj':    (True,  True,  False, _cfg.CACHE_UPDATE_INTERVAL),
        'no_coop_cache':    (True,  False, True,  _cfg.CACHE_UPDATE_INTERVAL),
        'minimal':          (False, False, True,  _cfg.CACHE_UPDATE_INTERVAL),
    }

    (_cfg.USE_DIFFUSION,
     _cfg.USE_COOP_CACHE,
     _cfg.USE_LEARNED_TRAJ,
     _cfg.CACHE_TIMESCALE_K_C) = _ABLATION_MAP[args.ablation]
    _cfg.ABLATION_MODE = args.ablation

    print(f"[Ablation] mode={args.ablation} | "
          f"diffusion={_cfg.USE_DIFFUSION} | "
          f"coop_cache={_cfg.USE_COOP_CACHE} | "
          f"learned_traj={_cfg.USE_LEARNED_TRAJ} | "
          f"K_C={_cfg.CACHE_TIMESCALE_K_C}")

    # ── Per-method output directories ────────────────────────────────
    # Each method + ablation combo gets its own subdir so parallel runs
    # don't overwrite each other and results are self-labelled.
    _method_tag = args.method
    _ablation_tag = args.ablation
    _tag = f"{_method_tag}/{_ablation_tag}"
    _cfg.CHECKPOINT_DIR = f'results/checkpoints/{_tag}'
    _cfg.LOG_DIR        = f'results/logs/{_tag}'
    os.makedirs(_cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(_cfg.LOG_DIR,        exist_ok=True)

    if args.eval:
        evaluate(args)
    else:
        train(args)