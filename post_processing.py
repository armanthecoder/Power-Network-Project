"""
Post-Processing and Report Generation for Multi-Sink Network Formation Simulation
Core Version - No additive leg.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pygame.math import Vector2 as Vec2

from constants import (
    Robot, Sink, Role, NodeType,
    R, COMM_RANGE, SPEED_MAX, SINK_TOUCH_DIST,
    K_R, K_THETA, POS_THRESH, PIVOT_THRESHOLD,
    SINK_TOUCH_SETTLEMENT_THRESH, MAX_ROBOTS, SPAWN_INTERVAL,
    GUIDANCE_STICK_FRAMES, NUM_SINKS, SINK_SPACE_WIDTH, SINK_SPACE_HEIGHT,
    SINK_MIN_SEP, FPS, WIDTH, HEIGHT,
    PIVOT_MARKER_SIZE, ROBOT_MARKER_SIZE, SINK_MARKER_SIZE, SOURCE_MARKER_SIZE,
)


def generate_report(
    robots:               List[Robot],
    sinks:                List[Sink],
    touched:              List[bool],
    sim_time:             float,
    network_complete_time: Optional[float],
    source_pos:           Vec2,
):
    """
    Write a text report + JSON data + two matplotlib plots:
      1. Full network topology.
      2. Steiner skeleton (Source ↔ Pivots ↔ Sinks).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Statistics ----
    network_robots = [r for r in robots if r.role == Role.NETWORK]
    pivot_robots   = [r for r in robots if r.node_type == NodeType.PIVOT]
    straight_robots= [r for r in network_robots if r.node_type == NodeType.STRAIGHT]
    moving_robots  = [r for r in robots if r.role == Role.MOVING]
    traitor_robots = [r for r in robots if r.role == Role.TRAITOR]
    touched_count  = sum(touched)

    branch_ids      = {r.branch_id for r in network_robots}
    max_branch_depth = max((len(r.branch_id) for r in network_robots), default=1)

    # Total robot-edge length
    total_network_length = sum(
        (r.pos - robots[r.parent_id].pos).length()
        for r in network_robots
        if r.parent_id is not None
    )

    # Path lengths Source → touching robot → sink
    sink_touching_robots = [
        (r, s)
        for r in network_robots
        for s in sinks
        if (s.pos - r.pos).length() <= SINK_TOUCH_DIST
    ]
    # deduplicate: one entry per sink (first match)
    seen_sids: set = set()
    unique_sink_robots = []
    for r, s in sink_touching_robots:
        if s.sid not in seen_sids:
            seen_sids.add(s.sid)
            unique_sink_robots.append((r, s))

    path_lengths: Dict[int, float] = {}
    for r, s in unique_sink_robots:
        pl      = (s.pos - r.pos).length()
        current = r
        while current.parent_id is not None:
            parent = robots[current.parent_id]
            pl    += (current.pos - parent.pos).length()
            current = parent
        path_lengths[s.sid] = pl

    # ---- Steiner skeleton (Source + Pivots + Sinks) ----
    sink_to_branch: Dict[int, str] = {s.sid: r.branch_id for r, s in unique_sink_robots}

    steiner_edges: set = set()
    for sid, _ in sink_to_branch.items():
        sink_pos = sinks[sid].pos
        touching = next((r for r, s in unique_sink_robots if s.sid == sid), None)
        if touching is None:
            continue

        current       = touching
        last_key_pos  = sink_pos
        while current is not None:
            if current.node_type == NodeType.PIVOT or current.role == Role.SOURCE:
                edge = tuple(sorted([
                    (round(last_key_pos.x, 2), round(last_key_pos.y, 2)),
                    (round(current.pos.x, 2),  round(current.pos.y, 2)),
                ]))
                steiner_edges.add(edge)
                last_key_pos = current.pos
            current = robots[current.parent_id] if current.parent_id is not None else None

    steiner_edge_list = []
    steiner_network_length = 0.0
    for p1, p2 in steiner_edges:
        dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        steiner_network_length += dist
        steiner_edge_list.append((p1, p2, dist))

    # ---- MST (Prim's) over source + all sinks ----
    mst_nodes = [source_pos] + [s.pos for s in sinks]
    n         = len(mst_nodes)
    in_mst    = [False] * n
    min_edge  = [float("inf")] * n
    parent_mst = [-1] * n
    min_edge[0] = 0.0
    mst_length  = 0.0
    mst_edges   = []

    for _ in range(n):
        u = min((v for v in range(n) if not in_mst[v]), key=lambda v: min_edge[v], default=-1)
        if u == -1 or min_edge[u] == float("inf"):
            break
        in_mst[u]     = True
        mst_length    += min_edge[u]
        if parent_mst[u] != -1:
            p1 = mst_nodes[parent_mst[u]]
            p2 = mst_nodes[u]
            mst_edges.append(((p1.x, p1.y), (p2.x, p2.y), min_edge[u]))
        for v in range(n):
            if not in_mst[v]:
                d = (mst_nodes[u] - mst_nodes[v]).length()
                if d < min_edge[v]:
                    min_edge[v]    = d
                    parent_mst[v]  = u

    # ================================================================
    # TEXT REPORT
    # ================================================================
    lines = []
    lines.append("=" * 70)
    lines.append("     NETWORK FORMATION SIMULATION — CORE REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("CONFIGURATION")
    lines.append("-" * 70)
    for name, val, desc in [
        ("R",                         R,                          "void radius"),
        ("COMM_RANGE",                COMM_RANGE,                 "communication radius"),
        ("SPEED_MAX",                 SPEED_MAX,                  "max speed"),
        ("SINK_TOUCH_DIST",           SINK_TOUCH_DIST,            "touch threshold"),
        ("K_R",                       K_R,                        "radial gain"),
        ("K_THETA",                   K_THETA,                    "angular gain"),
        ("POS_THRESH",                POS_THRESH,                 "pivot settlement thresh"),
        ("PIVOT_THRESHOLD",           PIVOT_THRESHOLD,            "max pivot metric"),
        ("SINK_TOUCH_SETTLEMENT_THRESH", SINK_TOUCH_SETTLEMENT_THRESH, "void settlement thresh"),
        ("MAX_ROBOTS",                MAX_ROBOTS,                 "max robots"),
        ("SPAWN_INTERVAL",            SPAWN_INTERVAL,             "spawn cadence"),
        ("NUM_SINKS",                 NUM_SINKS,                  "sink count"),
        ("FPS",                       FPS,                        "frames per second"),
    ]:
        lines.append(f"  {name:<38} {str(val):<12} {desc}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("SOURCE AND SINK POSITIONS")
    lines.append("-" * 70)
    lines.append(f"  Source: ({source_pos.x:.1f}, {source_pos.y:.1f})")
    for s in sinks:
        status = "TOUCHED" if touched[s.sid] else "UNTOUCHED"
        lines.append(f"  S{s.sid:<4} ({s.pos.x:7.1f}, {s.pos.y:7.1f})  {status}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("OUTPUT METRICS")
    lines.append("-" * 70)
    lines.append(f"  {'Total sim time':<40} {sim_time:.2f} s")
    lines.append(f"  {'Completion time':<40} "
                 f"{network_complete_time:.2f} s" if network_complete_time else
                 f"  {'Completion time':<40} NOT COMPLETED")
    lines.append(f"  {'Total robots deployed':<40} {len(robots)}")
    lines.append(f"  {'Network robots':<40} {len(network_robots)}")
    lines.append(f"  {'Pivot robots':<40} {len(pivot_robots)}")
    lines.append(f"  {'Straight robots':<40} {len(straight_robots)}")
    lines.append(f"  {'Moving robots (free)':<40} {len(moving_robots)}")
    lines.append(f"  {'Traitor robots':<40} {len(traitor_robots)}")
    lines.append(f"  {'Sinks touched':<40} {touched_count}/{len(sinks)}")
    lines.append(f"  {'Unique branches':<40} {len(branch_ids)}")
    lines.append(f"  {'Max branch depth':<40} {max_branch_depth}")
    lines.append(f"  {'Total network length (edges)':<40} {total_network_length:.2f}")
    lines.append(f"  {'Steiner network length':<40} {steiner_network_length:.2f}")
    lines.append(f"  {'MST length':<40} {mst_length:.2f}")
    if mst_length > 0:
        lines.append(f"  {'Steiner / MST ratio':<40} {steiner_network_length/mst_length:.3f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("PATH LENGTHS (Source → Sink)")
    lines.append("-" * 70)
    for sid in range(len(sinks)):
        if sid in path_lengths:
            lines.append(f"  S{sid:<4} {path_lengths[sid]:.2f}")
        else:
            lines.append(f"  S{sid:<4} NOT REACHED")
    if path_lengths:
        vals = list(path_lengths.values())
        lines.append(f"\n  avg={sum(vals)/len(vals):.2f}  "
                     f"min={min(vals):.2f}  max={max(vals):.2f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("PIVOT DETAILS")
    lines.append("-" * 70)
    if pivot_robots:
        for p in pivot_robots:
            lines.append(f"  R{p.rid:<4} pos=({p.pos.x:.1f},{p.pos.y:.1f})  "
                         f"branch={p.branch_id}  children={p.children_ids}  "
                         f"sinks={p.sink_indices[:6]}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    # ---- Save files ----
    folder    = f"{NUM_SINKS}sinks_{R}R_{timestamp}"
    os.makedirs(folder, exist_ok=True)
    base      = f"{NUM_SINKS}sinks_{R}R_{timestamp}"

    with open(os.path.join(folder, f"{base}_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    # JSON
    data = {
        "metadata": {"timestamp": timestamp, "num_sinks": NUM_SINKS, "R": R},
        "config":   {"R": R, "COMM_RANGE": COMM_RANGE, "SPEED_MAX": SPEED_MAX,
                     "SINK_TOUCH_DIST": SINK_TOUCH_DIST, "K_R": K_R, "K_THETA": K_THETA,
                     "PIVOT_THRESHOLD": PIVOT_THRESHOLD, "MAX_ROBOTS": MAX_ROBOTS,
                     "SPAWN_INTERVAL": SPAWN_INTERVAL, "NUM_SINKS": NUM_SINKS, "FPS": FPS},
        "metrics":  {"sim_time": sim_time, "network_complete_time": network_complete_time,
                     "total_robots": len(robots), "network_robots": len(network_robots),
                     "pivot_robots": len(pivot_robots), "touched_count": touched_count,
                     "total_network_length": total_network_length,
                     "steiner_network_length": steiner_network_length,
                     "mst_length": mst_length},
        "sinks":    [{"sid": s.sid, "x": float(s.pos.x), "y": float(s.pos.y),
                      "touched": touched[s.sid]} for s in sinks],
        "robots":   [{"rid": r.rid, "x": float(r.pos.x), "y": float(r.pos.y),
                      "role": r.role, "node_type": r.node_type,
                      "parent_id": r.parent_id, "children_ids": r.children_ids,
                      "branch_id": r.branch_id, "sink_indices": r.sink_indices} for r in robots],
    }
    with open(os.path.join(folder, f"{base}_data.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to: {folder}/")

    # ================================================================
    # PLOT 1: Full network topology
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.invert_yaxis()

    for s in sinks:
        col = "green" if touched[s.sid] else "red"
        ax.scatter(s.pos.x, s.pos.y, c=col, s=SINK_MARKER_SIZE,
                   marker="s", edgecolors="black", linewidths=1, zorder=5)
        ax.annotate(f"S{s.sid}", (s.pos.x+10, s.pos.y), fontsize=7, zorder=6)

    ax.scatter(source_pos.x, source_pos.y, c="lime", s=SOURCE_MARKER_SIZE,
               marker="*", edgecolors="black", linewidths=1, zorder=5)
    ax.annotate("SOURCE", (source_pos.x-25, source_pos.y+20),
                fontsize=9, fontweight="bold", zorder=6)

    for r in robots:
        if r.parent_id is not None and r.role == Role.NETWORK:
            p = robots[r.parent_id]
            ax.plot([p.pos.x, r.pos.x], [p.pos.y, r.pos.y],
                    "b-", linewidth=1.5, alpha=0.6, zorder=2)

    for r in robots:
        if r.role == Role.NETWORK:
            for s in sinks:
                if (s.pos - r.pos).length() <= SINK_TOUCH_DIST:
                    ax.plot([r.pos.x, s.pos.x], [r.pos.y, s.pos.y],
                            "g--", linewidth=2, alpha=0.8, zorder=3)

    for r in robots:
        if r.role == Role.SOURCE:
            continue
        if r.node_type == NodeType.PIVOT:
            ax.scatter(r.pos.x, r.pos.y, c="magenta", s=PIVOT_MARKER_SIZE,
                       marker="o", edgecolors="black", linewidths=2, zorder=4)
            ax.annotate(f"P{r.rid}", (r.pos.x+10, r.pos.y-10),
                        fontsize=7, color="magenta", zorder=6)
        elif r.role == Role.NETWORK:
            ax.scatter(r.pos.x, r.pos.y, c="royalblue", s=ROBOT_MARKER_SIZE,
                       marker="o", edgecolors="black", linewidths=1, zorder=3)
            ax.annotate(f"{r.rid}", (r.pos.x+8, r.pos.y-5),
                        fontsize=6, color="blue", alpha=0.7, zorder=6)
        elif r.role == Role.TRAITOR:
            ax.scatter(r.pos.x, r.pos.y, c="orange", s=int(ROBOT_MARKER_SIZE*1.5),
                       marker="^", edgecolors="black", linewidths=1, zorder=4)
            ax.annotate(f"T{r.rid}", (r.pos.x+8, r.pos.y-10),
                        fontsize=7, color="darkorange", zorder=6)
        else:
            ax.scatter(r.pos.x, r.pos.y, c="cyan",
                       s=int(ROBOT_MARKER_SIZE*1.5), marker="o", alpha=0.5, zorder=2)

    legend = [
        mpatches.Patch(color="lime",      label="Source"),
        mpatches.Patch(color="magenta",   label=f"Pivots ({len(pivot_robots)})"),
        mpatches.Patch(color="royalblue", label=f"Straight ({len(straight_robots)})"),
        mpatches.Patch(color="cyan",      label=f"Moving ({len(moving_robots)})"),
        mpatches.Patch(color="orange",    label=f"Traitors ({len(traitor_robots)})"),
        mpatches.Patch(color="green",     label=f"Touched ({touched_count})"),
        mpatches.Patch(color="red",       label=f"Untouched ({len(sinks)-touched_count})"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)

    title = (f"Network Topology — {len(network_robots)} network robots, "
             f"{len(pivot_robots)} pivots")
    if network_complete_time:
        title += f"\nCompleted in {network_complete_time:.2f} s"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    all_x = [s.pos.x for s in sinks] + [r.pos.x for r in robots] + [source_pos.x]
    all_y = [s.pos.y for s in sinks] + [r.pos.y for r in robots] + [source_pos.y]
    cx, cy = (min(all_x)+max(all_x))/2, (min(all_y)+max(all_y))/2
    W  = max(max(all_x)-min(all_x)+800, 2000)
    H  = max(max(all_y)-min(all_y)+800, 3000)
    ax.set_xlim(cx-W/2, cx+W/2)
    ax.set_ylim(cy+H/2, cy-H/2)

    plt.tight_layout()
    plot1 = os.path.join(folder, f"{base}_topology.png")
    plt.savefig(plot1, dpi=150, bbox_inches="tight")
    print(f"Topology plot: {plot1}")

    # ================================================================
    # PLOT 2: Steiner skeleton
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    ax2.invert_yaxis()

    for p1, p2, dist in steiner_edge_list:
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], "b-", linewidth=3, alpha=0.7, zorder=2)
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax2.annotate(f"{dist:.0f}", (mx, my), fontsize=7, color="darkblue",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7), zorder=4)

    for s in sinks:
        col = "green" if touched[s.sid] else "red"
        ax2.scatter(s.pos.x, s.pos.y, c=col, s=SINK_MARKER_SIZE+20,
                    marker="s", edgecolors="black", linewidths=1, zorder=5)
        ax2.annotate(f"S{s.sid}", (s.pos.x+12, s.pos.y), fontsize=8, fontweight="bold", zorder=6)

    ax2.scatter(source_pos.x, source_pos.y, c="lime", s=SOURCE_MARKER_SIZE+20,
                marker="*", edgecolors="black", linewidths=2, zorder=5)
    ax2.annotate("SOURCE", (source_pos.x-30, source_pos.y+25),
                 fontsize=10, fontweight="bold", zorder=6)

    for p in pivot_robots:
        ax2.scatter(p.pos.x, p.pos.y, c="magenta", s=PIVOT_MARKER_SIZE,
                    marker="o", edgecolors="black", linewidths=1, zorder=5)
        ax2.annotate(f"P{p.rid}", (p.pos.x+10, p.pos.y-10),
                     fontsize=8, color="magenta", fontweight="bold", zorder=6)

    legend2 = [
        mpatches.Patch(color="lime",    label="Source"),
        mpatches.Patch(color="magenta", label=f"Pivots ({len(pivot_robots)})"),
        mpatches.Patch(color="green",   label=f"Touched ({touched_count})"),
        mpatches.Patch(color="red",     label=f"Untouched ({len(sinks)-touched_count})"),
        mpatches.Patch(color="blue",    label=f"Steiner edges ({len(steiner_edges)})"),
    ]
    ax2.legend(handles=legend2, loc="upper right", fontsize=10)
    ax2.set_title(f"Steiner Skeleton — length={steiner_network_length:.1f}",
                  fontsize=16, fontweight="bold")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y")
    ax2.set_xlim(cx-W/2, cx+W/2)
    ax2.set_ylim(cy+H/2, cy-H/2)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    plt.tight_layout()
    plot2 = os.path.join(folder, f"{base}_steiner.png")
    plt.savefig(plot2, dpi=150, bbox_inches="tight")
    print(f"Steiner plot:  {plot2}")

    plt.show()
