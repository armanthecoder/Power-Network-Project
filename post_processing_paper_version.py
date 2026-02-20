"""
Post-Processing: Simulation Output Writer
==========================================
Writes a structured text file with all raw simulation data.
This file is read by unified_post_processing.py for MST, GeoSteiner,
and power-loss analysis.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from pygame.math import Vector2 as Vec2

from constants_paper_version import (
    Robot, Sink, Role, NodeType,
    R, COMM_RANGE, SPEED_MAX, SINK_TOUCH_DIST,
    K_R, K_THETA, POS_THRESH, PIVOT_THRESHOLD,
    SINK_TOUCH_SETTLEMENT_THRESH, MAX_ROBOTS, SPAWN_INTERVAL,
    GUIDANCE_STICK_FRAMES, NUM_SINKS, SINK_SPACE_WIDTH, SINK_SPACE_HEIGHT,
    SINK_MIN_SEP, SINK_RING_R, FPS, WIDTH, HEIGHT, RANDOM_SEED,
)

from assistive_functions_paper_version import calc_pivot_metric


def generate_report(
    robots: List[Robot],
    sinks: List[Sink],
    touched: List[bool],
    sim_time: float,
    sim_steps: int,
    network_complete_time: Optional[float],
    source_pos: Vec2,
):
    """Write simulation output data to a structured text file.

    The output contains everything the unified post-processing tool needs:
      - Random seed
      - All configuration parameters
      - Source and sink positions
      - Simulation metrics
      - Full robot roster (rid, type, branch, parent, assigned_sinks, position)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Classify robots
    network_robots = [r for r in robots if r.role == Role.NETWORK]
    pivot_robots   = [r for r in robots if r.node_type == NodeType.PIVOT]
    straight_robots = [r for r in network_robots if r.node_type == NodeType.STRAIGHT]
    moving_robots  = [r for r in robots if r.role == Role.MOVING]
    touched_count  = sum(touched)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("     SIMULATION OUTPUT DATA")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── Random Seed ───────────────────────────────────────────
    lines.append("-" * 70)
    lines.append("RANDOM SEED")
    lines.append("-" * 70)
    lines.append(f"  {RANDOM_SEED}")
    lines.append("")

    # ── Configuration ─────────────────────────────────────────
    lines.append("-" * 70)
    lines.append("CONFIGURATION PARAMETERS")
    lines.append("-" * 70)
    lines.append(f"  {'Parameter':<35} {'Value':<20} {'Description'}")
    lines.append(f"  {'-'*33} {'-'*18} {'-'*30}")
    cfg = [
        ("R (Void Radius)",               R,                        "Desired separation distance"),
        ("COMM_RANGE",                     COMM_RANGE,               "Communication radius"),
        ("SPEED_MAX",                      SPEED_MAX,                "Maximum robot speed"),
        ("SINK_TOUCH_DIST",                SINK_TOUCH_DIST,          "Distance to touch sink"),
        ("K_R (Radial gain)",              K_R,                      "Void controller radial gain"),
        ("K_THETA (Angular gain)",         K_THETA,                  "Void controller angular gain"),
        ("POS_THRESH",                     POS_THRESH,               "Settlement threshold for pivot"),
        ("PIVOT_THRESHOLD",                PIVOT_THRESHOLD,          "Pivot eligibility threshold"),
        ("SINK_TOUCH_SETTLEMENT_THRESH",   SINK_TOUCH_SETTLEMENT_THRESH, "Sink touch settlement dist"),
        ("MAX_ROBOTS",                     MAX_ROBOTS,               "Maximum robots allowed"),
        ("SPAWN_INTERVAL",                 SPAWN_INTERVAL,           "Seconds between spawns"),
        ("GUIDANCE_STICK_FRAMES",          GUIDANCE_STICK_FRAMES,    "Guidance lock duration"),
        ("NUM_SINKS",                      NUM_SINKS,                "Number of sinks"),
        ("SINK_SPACE_WIDTH",               SINK_SPACE_WIDTH,         "Sink region width"),
        ("SINK_SPACE_HEIGHT",              SINK_SPACE_HEIGHT,        "Sink region height"),
        ("SINK_MIN_SEP",                   SINK_MIN_SEP,             "Min sink sep (xCOMM_RANGE)"),
        ("SINK_RING_R",                    SINK_RING_R,              "Sink placement radius (legacy)"),
        ("FPS",                            FPS,                      "Frames per second"),
        ("WIDTH",                          WIDTH,                    "Window width"),
        ("HEIGHT",                         HEIGHT,                   "Window height"),
    ]
    for name, val, desc in cfg:
        lines.append(f"  {name:<35} {val:<20} {desc}")
    lines.append("")

    # ── Source & Sinks ────────────────────────────────────────
    lines.append("-" * 70)
    lines.append("SOURCE AND SINK POSITIONS")
    lines.append("-" * 70)
    lines.append(f"  Source Position: ({source_pos.x:.2f}, {source_pos.y:.2f})")
    lines.append("")
    lines.append(f"  {'Sink ID':<10} {'X Position':<15} {'Y Position':<15} {'Status'}")
    lines.append(f"  {'-'*8} {'-'*13} {'-'*13} {'-'*12}")
    for s in sinks:
        status = "TOUCHED" if touched[s.sid] else "UNTOUCHED"
        lines.append(f"  S{s.sid:<8} {s.pos.x:<15.2f} {s.pos.y:<15.2f} {status}")
    lines.append("")

    # ── Metrics ───────────────────────────────────────────────
    lines.append("-" * 70)
    lines.append("OUTPUT METRICS")
    lines.append("-" * 70)
    lines.append(f"  {'Metric':<40} {'Value'}")
    lines.append(f"  {'-'*38} {'-'*25}")
    lines.append(f"  {'Total Simulation Time':<40} {sim_time:.2f} seconds")
    lines.append(f"  {'Total Simulation Steps':<40} {sim_steps}")
    if network_complete_time is not None:
        lines.append(f"  {'Network Completion Time':<40} {network_complete_time:.2f} seconds")
    else:
        lines.append(f"  {'Network Completion Time':<40} NOT COMPLETED")
    lines.append(f"  {'Total Robots Deployed':<40} {len(robots)}")
    lines.append(f"  {'Network Robots (attached)':<40} {len(network_robots)}")
    lines.append(f"  {'Pivot Robots':<40} {len(pivot_robots)}")
    lines.append(f"  {'Straight Robots':<40} {len(straight_robots)}")
    lines.append(f"  {'Moving Robots (free)':<40} {len(moving_robots)}")
    lines.append(f"  {'Sinks Touched':<40} {touched_count}/{len(sinks)}")
    lines.append("")

    # ── Full Robot Roster ─────────────────────────────────────
    lines.append("-" * 70)
    lines.append("FULL ROBOT ROSTER")
    lines.append("-" * 70)
    lines.append(f"  {'RID':<8} {'Type':<10} {'Branch':<12} {'Parent':<8} {'Sinks':<20} {'Position'}")
    lines.append(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*6} {'-'*18} {'-'*20}")

    source_robot = [r for r in robots if r.role == Role.SOURCE]
    for r in source_robot:
        sinks_str = str(r.sink_indices)
        lines.append(
            f"  {r.rid:<8} {'SOURCE':<10} {r.branch_id:<12} {'---':<8} "
            f"{sinks_str:<20} ({r.pos.x:.1f}, {r.pos.y:.1f})"
        )

    for r in sorted(network_robots, key=lambda x: x.rid):
        ntype = "PIVOT" if r.node_type == NodeType.PIVOT else "STRAIGHT"
        parent_str = str(r.parent_id) if r.parent_id is not None else "---"
        sinks_str = str(r.sink_indices)
        lines.append(
            f"  {r.rid:<8} {ntype:<10} {r.branch_id:<12} {parent_str:<8} "
            f"{sinks_str:<20} ({r.pos.x:.1f}, {r.pos.y:.1f})"
        )
    lines.append("")

    # ── Pivot Suitability ──────────────────────────────────────
    lines.append("-" * 70)
    lines.append("PIVOT SUITABILITY (Pivot Robots Only)")
    lines.append("-" * 70)
    lines.append(f"  {'RID':<8} {'Sinks':<20} {'Metric':<12} {'Rating'}")
    lines.append(f"  {'-'*6} {'-'*18} {'-'*10} {'-'*15}")
    pivot_metrics = []
    for r in sorted(pivot_robots, key=lambda x: x.rid):
        metric = calc_pivot_metric(r, robots, sinks)
        sinks_str = str(r.sink_indices)
        if metric is None:
            metric_str = "N/A"
            rating = "(< 2 sinks or no parent)"
        elif metric < PIVOT_THRESHOLD:
            metric_str = f"{metric:.4f}"
            rating = "GOOD (below threshold)"
            pivot_metrics.append(metric)
        else:
            metric_str = f"{metric:.4f}"
            rating = "POOR (above threshold)"
            pivot_metrics.append(metric)
        lines.append(
            f"  {r.rid:<8} {sinks_str:<20} {metric_str:<12} {rating}"
        )
    if pivot_metrics:
        max_metric = max(pivot_metrics)
        lines.append(f"\n  Maximum Pivot Suitability Error: {max_metric:.4f}")
    else:
        lines.append(f"\n  Maximum Pivot Suitability Error: N/A")
    lines.append(f"  Pivot Threshold: {PIVOT_THRESHOLD}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("END OF SIMULATION OUTPUT")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    # ── Save to file ──────────────────────────────────────────
    folder_name = f"{NUM_SINKS}sinks_{R}R_seed{RANDOM_SEED}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    base_filename = f"{NUM_SINKS}sinks_{R}R_seed{RANDOM_SEED}_{timestamp}"
    report_filename = os.path.join(folder_name, f"{base_filename}_simdata.txt")
    with open(report_filename, "w") as f:
        f.write(report_text)
    print(f"\nSimulation data saved to: {report_filename}")

    return report_filename

