"""
Post-Processing and Report Generation for Multi-Sink Network Formation Simulation
Paper Version - Report generation, metrics calculation, and visualization
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

from constants_paper_version import (
    Robot, Sink, Role, NodeType,
    R, COMM_RANGE, SPEED_MAX, SINK_TOUCH_DIST,
    K_R, K_THETA, POS_THRESH, PIVOT_THRESHOLD,
    SINK_TOUCH_SETTLEMENT_THRESH, MAX_ROBOTS, SPAWN_INTERVAL,
    GUIDANCE_STICK_FRAMES, NUM_SINKS, SINK_SPACE_WIDTH, SINK_SPACE_HEIGHT,
    SINK_MIN_SEP, SINK_RING_R, FPS, WIDTH, HEIGHT,
    PIVOT_MARKER_SIZE, ROBOT_MARKER_SIZE, SINK_MARKER_SIZE, SOURCE_MARKER_SIZE
)


def generate_report(
    robots: List[Robot],
    sinks: List[Sink],
    touched: List[bool],
    sim_time: float,
    network_complete_time: Optional[float],
    source_pos: Vec2,
):
    """
    Generate a comprehensive textbook-style report with:
    1. Configuration parameters
    2. Output metrics
    3. Network topology plot
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Count statistics
    network_robots = [r for r in robots if r.role == Role.NETWORK]
    pivot_robots = [r for r in robots if r.node_type == NodeType.PIVOT]
    straight_robots = [r for r in network_robots if r.node_type == NodeType.STRAIGHT]
    moving_robots = [r for r in robots if r.role == Role.MOVING]
    touched_count = sum(touched)
    
    # Calculate branch statistics
    branch_ids = set(r.branch_id for r in network_robots)
    max_branch_depth = max((len(r.branch_id) for r in network_robots), default=1)
    
    # =========================
    # NETWORK LENGTH CALCULATION (Robot-based, for reference)
    # =========================
    total_network_length = 0.0
    
    for r in network_robots:
        if r.parent_id is not None:
            parent = robots[r.parent_id]
            edge_length = (r.pos - parent.pos).length()
            total_network_length += edge_length
    
    # Calculate path lengths: Source -> Pivot(s) -> Sink
    sink_touching_robots = []
    for r in network_robots:
        for s in sinks:
            if (s.pos - r.pos).length() <= SINK_TOUCH_DIST:
                sink_touching_robots.append((r, s))
                break
    
    path_lengths: Dict[int, float] = {}
    for r, s in sink_touching_robots:
        path_length = (s.pos - r.pos).length()
        current = r
        while current.parent_id is not None:
            parent = robots[current.parent_id]
            path_length += (current.pos - parent.pos).length()
            current = parent
        path_lengths[s.sid] = path_length
    
    # =========================
    # STEINER NETWORK LENGTH (Source + Pivots + Sinks only)
    # =========================
    sink_to_branch: Dict[int, str] = {}
    for r, s in sink_touching_robots:
        sink_to_branch[s.sid] = r.branch_id
    
    def find_parent_pivot(robot: Robot) -> Optional[Robot]:
        """Find the nearest ancestor that is a pivot or source"""
        current = robot
        while current.parent_id is not None:
            parent = robots[current.parent_id]
            if parent.node_type == NodeType.PIVOT or parent.role == Role.SOURCE:
                return parent
            current = parent
        return None
    
    steiner_edges: set = set()
    
    for sid, branch_id in sink_to_branch.items():
        sink_pos = sinks[sid].pos
        
        touching_robot = None
        for r, s in sink_touching_robots:
            if s.sid == sid:
                touching_robot = r
                break
        
        if touching_robot is None:
            continue
        
        current = touching_robot
        last_key_pos = sink_pos
        
        while current is not None:
            if current.node_type == NodeType.PIVOT or current.role == Role.SOURCE:
                edge = tuple(sorted([
                    (round(last_key_pos.x, 2), round(last_key_pos.y, 2)),
                    (round(current.pos.x, 2), round(current.pos.y, 2))
                ]))
                steiner_edges.add(edge)
                last_key_pos = current.pos
            
            if current.parent_id is not None:
                current = robots[current.parent_id]
            else:
                break
    
    steiner_network_length = 0.0
    steiner_edge_list = []
    for edge in steiner_edges:
        p1, p2 = edge
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        steiner_network_length += dist
        steiner_edge_list.append((p1, p2, dist))
    
    # =========================
    # MINIMUM SPANNING TREE (MST) CALCULATION
    # =========================
    mst_nodes = [source_pos] + [s.pos for s in sinks]
    mst_n = len(mst_nodes)
    
    mst_dist_matrix = [[0.0] * mst_n for _ in range(mst_n)]
    for i in range(mst_n):
        for j in range(i + 1, mst_n):
            d = (mst_nodes[i] - mst_nodes[j]).length()
            mst_dist_matrix[i][j] = d
            mst_dist_matrix[j][i] = d
    
    in_mst = [False] * mst_n
    min_edge = [float('inf')] * mst_n
    min_edge[0] = 0
    mst_length = 0.0
    mst_edges = []
    parent = [-1] * mst_n
    
    for _ in range(mst_n):
        u = -1
        for v in range(mst_n):
            if not in_mst[v] and (u == -1 or min_edge[v] < min_edge[u]):
                u = v
        
        if u == -1 or min_edge[u] == float('inf'):
            break
        
        in_mst[u] = True
        mst_length += min_edge[u]
        
        if parent[u] != -1:
            p1 = mst_nodes[parent[u]]
            p2 = mst_nodes[u]
            mst_edges.append(((p1.x, p1.y), (p2.x, p2.y), min_edge[u]))
        
        for v in range(mst_n):
            if not in_mst[v] and mst_dist_matrix[u][v] < min_edge[v]:
                min_edge[v] = mst_dist_matrix[u][v]
                parent[v] = u
    
    # =========================
    # TEXT REPORT
    # =========================
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("     NETWORK FORMATION SIMULATION - TEXTBOOK REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Configuration Section
    report_lines.append("-" * 70)
    report_lines.append("CONFIGURATION PARAMETERS")
    report_lines.append("-" * 70)
    report_lines.append(f"  {'Parameter':<35} {'Value':<20} {'Description'}")
    report_lines.append(f"  {'-'*33} {'-'*18} {'-'*30}")
    report_lines.append(f"  {'R (Void Radius)':<35} {R:<20} Desired separation distance")
    report_lines.append(f"  {'COMM_RANGE':<35} {COMM_RANGE:<20} Communication radius")
    report_lines.append(f"  {'SPEED_MAX':<35} {SPEED_MAX:<20} Maximum robot speed")
    report_lines.append(f"  {'SINK_TOUCH_DIST':<35} {SINK_TOUCH_DIST:<20} Distance to touch sink")
    report_lines.append(f"  {'K_R (Radial gain)':<35} {K_R:<20} Void controller radial gain")
    report_lines.append(f"  {'K_THETA (Angular gain)':<35} {K_THETA:<20} Void controller angular gain")
    report_lines.append(f"  {'POS_THRESH':<35} {POS_THRESH:<20} Settlement threshold for pivot")
    report_lines.append(f"  {'PIVOT_THRESHOLD':<35} {PIVOT_THRESHOLD:<20} Pivot eligibility threshold")
    report_lines.append(f"  {'SINK_TOUCH_SETTLEMENT_THRESH':<35} {SINK_TOUCH_SETTLEMENT_THRESH:<20} Sink touch settlement dist")
    report_lines.append(f"  {'MAX_ROBOTS':<35} {MAX_ROBOTS:<20} Maximum robots allowed")
    report_lines.append(f"  {'SPAWN_INTERVAL':<35} {SPAWN_INTERVAL:<20} Seconds between spawns")
    report_lines.append(f"  {'GUIDANCE_STICK_FRAMES':<35} {GUIDANCE_STICK_FRAMES:<20} Guidance lock duration")
    report_lines.append(f"  {'NUM_SINKS':<35} {NUM_SINKS:<20} Number of sinks")
    report_lines.append(f"  {'SINK_SPACE_WIDTH':<35} {SINK_SPACE_WIDTH:<20} Sink region width")
    report_lines.append(f"  {'SINK_SPACE_HEIGHT':<35} {SINK_SPACE_HEIGHT:<20} Sink region height")
    report_lines.append(f"  {'SINK_MIN_SEP':<35} {SINK_MIN_SEP:<20} Min sink sep (xCOMM_RANGE)")
    report_lines.append(f"  {'SINK_RING_R':<35} {SINK_RING_R:<20} Sink placement radius (legacy)")
    report_lines.append(f"  {'FPS':<35} {FPS:<20} Frames per second")
    report_lines.append("")
    
    # Source and Sink Positions Section
    report_lines.append("-" * 70)
    report_lines.append("SOURCE AND SINK POSITIONS")
    report_lines.append("-" * 70)
    report_lines.append(f"  Source Position: ({source_pos.x:.2f}, {source_pos.y:.2f})")
    report_lines.append("")
    report_lines.append(f"  {'Sink ID':<10} {'X Position':<15} {'Y Position':<15} {'Status'}")
    report_lines.append(f"  {'-'*8} {'-'*13} {'-'*13} {'-'*12}")
    for s in sinks:
        status = "TOUCHED" if touched[s.sid] else "UNTOUCHED"
        report_lines.append(f"  S{s.sid:<8} {s.pos.x:<15.2f} {s.pos.y:<15.2f} {status}")
    report_lines.append("")
    
    # Output Metrics Section
    report_lines.append("-" * 70)
    report_lines.append("OUTPUT METRICS")
    report_lines.append("-" * 70)
    report_lines.append(f"  {'Metric':<40} {'Value'}")
    report_lines.append(f"  {'-'*38} {'-'*25}")
    report_lines.append(f"  {'Total Simulation Time':<40} {sim_time:.2f} seconds")
    if network_complete_time is not None:
        report_lines.append(f"  {'Network Completion Time':<40} {network_complete_time:.2f} seconds")
    else:
        report_lines.append(f"  {'Network Completion Time':<40} NOT COMPLETED")
    report_lines.append(f"  {'Total Robots Deployed':<40} {len(robots)}")
    report_lines.append(f"  {'Network Robots (attached)':<40} {len(network_robots)}")
    report_lines.append(f"  {'Pivot Robots':<40} {len(pivot_robots)}")
    report_lines.append(f"  {'Straight Robots':<40} {len(straight_robots)}")
    report_lines.append(f"  {'Moving Robots (free)':<40} {len(moving_robots)}")
    report_lines.append(f"  {'Sinks Touched':<40} {touched_count}/{len(sinks)}")
    report_lines.append(f"  {'Unique Branches':<40} {len(branch_ids)}")
    report_lines.append(f"  {'Max Branch Depth':<40} {max_branch_depth}")
    report_lines.append(f"  {'Total Network Length (tree edges)':<40} {total_network_length:.2f} units")
    report_lines.append("")
    
    # Network Length Details Section
    report_lines.append("-" * 70)
    report_lines.append("NETWORK LENGTH DETAILS")
    report_lines.append("-" * 70)
    report_lines.append(f"  Total Network Length (sum of all edges): {total_network_length:.2f} units")
    report_lines.append(f"  (Each parent-child edge counted exactly once)")
    report_lines.append("")
    report_lines.append(f"  {'Sink ID':<12} {'Path Length (Source->Sink)':<30} {'Status'}")
    report_lines.append(f"  {'-'*10} {'-'*28} {'-'*15}")
    for sid in range(len(sinks)):
        if sid in path_lengths:
            status = "TOUCHED"
            report_lines.append(f"  S{sid:<10} {path_lengths[sid]:<30.2f} {status}")
        else:
            status = "NOT REACHED"
            report_lines.append(f"  S{sid:<10} {'N/A':<30} {status}")
    if path_lengths:
        avg_path = sum(path_lengths.values()) / len(path_lengths)
        max_path = max(path_lengths.values())
        min_path = min(path_lengths.values())
        report_lines.append("")
        report_lines.append(f"  Average Path Length: {avg_path:.2f} units")
        report_lines.append(f"  Max Path Length: {max_path:.2f} units")
        report_lines.append(f"  Min Path Length: {min_path:.2f} units")
    report_lines.append("")
    
    # STEINER NETWORK LENGTH Section
    report_lines.append("-" * 70)
    report_lines.append("STEINER NETWORK LENGTH (Source + Pivots + Sinks only)")
    report_lines.append("-" * 70)
    report_lines.append(f"  This metric counts ONLY straight-line distances between:")
    report_lines.append(f"    - Source")
    report_lines.append(f"    - Pivot nodes ({len(pivot_robots)} pivots)")
    report_lines.append(f"    - Sink nodes ({touched_count} touched)")
    report_lines.append("")
    report_lines.append(f"  *** STEINER NETWORK LENGTH: {steiner_network_length:.2f} units ***")
    report_lines.append(f"  Number of edges: {len(steiner_edges)}")
    report_lines.append("")
    report_lines.append(f"  Edge Details:")
    report_lines.append(f"  {'From':<25} {'To':<25} {'Length'}")
    report_lines.append(f"  {'-'*23} {'-'*23} {'-'*12}")
    for p1, p2, dist in sorted(steiner_edge_list, key=lambda x: -x[2]):
        report_lines.append(f"  ({p1[0]:.1f}, {p1[1]:.1f}){'':<8} ({p2[0]:.1f}, {p2[1]:.1f}){'':<8} {dist:.2f}")
    report_lines.append("")
    
    # MINIMUM SPANNING TREE Section
    report_lines.append("-" * 70)
    report_lines.append("MINIMUM SPANNING TREE (Source + All Sinks)")
    report_lines.append("-" * 70)
    report_lines.append(f"  This is the theoretical minimum network length connecting:")
    report_lines.append(f"    - Source (1 node)")
    report_lines.append(f"    - All Sinks ({len(sinks)} sinks)")
    report_lines.append(f"  Using straight-line distances (Euclidean MST)")
    report_lines.append("")
    report_lines.append(f"  *** MINIMUM SPANNING TREE LENGTH: {mst_length:.2f} units ***")
    report_lines.append(f"  Number of edges: {len(mst_edges)}")
    report_lines.append("")
    
    report_lines.append("")
    report_lines.append(f"  Comparison:")
    report_lines.append(f"    - MST Length (theoretical min):     {mst_length:.2f} units")
    report_lines.append(f"    - Steiner Network Length:           {steiner_network_length:.2f} units")
    report_lines.append(f"    - Actual Network Length (robot):    {total_network_length:.2f} units")
    if mst_length > 0:
        report_lines.append(f"    - Steiner/MST ratio:                {steiner_network_length/mst_length:.2f}")
        report_lines.append(f"    - Robot Network/MST ratio:          {total_network_length/mst_length:.2f}")
    report_lines.append("")
    
    # Efficiency Metrics
    report_lines.append("-" * 70)
    report_lines.append("EFFICIENCY METRICS")
    report_lines.append("-" * 70)
    if network_complete_time is not None and len(network_robots) > 0:
        robots_per_sink = len(network_robots) / len(sinks)
        time_per_sink = network_complete_time / len(sinks)
        report_lines.append(f"  {'Robots per Sink':<40} {robots_per_sink:.2f}")
        report_lines.append(f"  {'Time per Sink':<40} {time_per_sink:.2f} seconds")
        report_lines.append(f"  {'Spawn Rate':<40} {1/SPAWN_INTERVAL:.2f} robots/sec")
        report_lines.append(f"  {'Network Length per Sink (robot)':<40} {total_network_length/len(sinks):.2f} units")
        report_lines.append(f"  {'Steiner Length per Sink':<40} {steiner_network_length/len(sinks):.2f} units")
    report_lines.append("")
    
    # Pivot Details
    report_lines.append("-" * 70)
    report_lines.append("PIVOT ROBOTS DETAILS")
    report_lines.append("-" * 70)
    if pivot_robots:
        report_lines.append(f"  {'Robot ID':<10} {'Position':<25} {'Branch ID':<15} {'Children':<15} {'Sinks'}")
        report_lines.append(f"  {'-'*8} {'-'*23} {'-'*13} {'-'*13} {'-'*15}")
        for p in pivot_robots:
            pos_str = f"({p.pos.x:.1f}, {p.pos.y:.1f})"
            children_str = str(p.children_ids)
            sinks_str = str(p.sink_indices[:5]) + ("..." if len(p.sink_indices) > 5 else "")
            report_lines.append(f"  {p.rid:<10} {pos_str:<25} {p.branch_id:<15} {children_str:<15} {sinks_str}")
    else:
        report_lines.append("  No pivot robots formed.")
    report_lines.append("")
    
    # Branch Summary
    report_lines.append("-" * 70)
    report_lines.append("BRANCH SUMMARY")
    report_lines.append("-" * 70)
    branch_counts: Dict[str, int] = {}
    for r in network_robots:
        branch_counts[r.branch_id] = branch_counts.get(r.branch_id, 0) + 1
    for bid in sorted(branch_counts.keys(), key=lambda x: (len(x), x)):
        report_lines.append(f"  Branch {bid}: {branch_counts[bid]} robots")
    report_lines.append("")
    
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)
    
    report_text = "\n".join(report_lines)
    
    # Print to console
    print(report_text)
    
    # Create output folder
    folder_name = f"{NUM_SINKS}sinks_{R}R_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    
    base_filename = f"{NUM_SINKS}sinks_{R}R_{timestamp}"
    
    # Save text report
    report_filename = os.path.join(folder_name, f"{base_filename}_report.txt")
    with open(report_filename, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_filename}")
    
    # =========================
    # EXPORT EXPERIMENT DATA (JSON)
    # =========================
    experiment_data = {
        "metadata": {
            "timestamp": timestamp,
            "generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "num_sinks": NUM_SINKS,
            "R": R,
        },
        "config": {
            "R": R,
            "COMM_RANGE": COMM_RANGE,
            "SPEED_MAX": SPEED_MAX,
            "SINK_TOUCH_DIST": SINK_TOUCH_DIST,
            "K_R": K_R,
            "K_THETA": K_THETA,
            "POS_THRESH": POS_THRESH,
            "PIVOT_THRESHOLD": PIVOT_THRESHOLD,
            "SINK_TOUCH_SETTLEMENT_THRESH": SINK_TOUCH_SETTLEMENT_THRESH,
            "MAX_ROBOTS": MAX_ROBOTS,
            "SPAWN_INTERVAL": SPAWN_INTERVAL,
            "GUIDANCE_STICK_FRAMES": GUIDANCE_STICK_FRAMES,
            "NUM_SINKS": NUM_SINKS,
            "SINK_SPACE_WIDTH": SINK_SPACE_WIDTH,
            "SINK_SPACE_HEIGHT": SINK_SPACE_HEIGHT,
            "SINK_MIN_SEP": SINK_MIN_SEP,
            "SINK_RING_R": SINK_RING_R,
            "FPS": FPS,
            "WIDTH": WIDTH,
            "HEIGHT": HEIGHT,
        },
        "source": {
            "x": float(source_pos.x),
            "y": float(source_pos.y),
        },
        "sinks": [
            {
                "sid": s.sid,
                "x": float(s.pos.x),
                "y": float(s.pos.y),
                "touched": touched[s.sid],
            }
            for s in sinks
        ],
        "robots": [
            {
                "rid": r.rid,
                "x": float(r.pos.x),
                "y": float(r.pos.y),
                "role": r.role,
                "node_type": r.node_type,
                "parent_id": r.parent_id,
                "children_ids": r.children_ids,
                "branch_id": r.branch_id,
                "sink_indices": r.sink_indices,
                "void_pos_x": float(r.void_pos_vis.x) if r.void_pos_vis else None,
                "void_pos_y": float(r.void_pos_vis.y) if r.void_pos_vis else None,
            }
            for r in robots
        ],
        "metrics": {
            "sim_time": sim_time,
            "network_complete_time": network_complete_time,
            "total_robots": len(robots),
            "network_robots": len(network_robots),
            "pivot_robots": len(pivot_robots),
            "straight_robots": len(straight_robots),
            "moving_robots": len(moving_robots),
            "touched_count": touched_count,
            "unique_branches": len(branch_ids),
            "max_branch_depth": max_branch_depth,
            "total_network_length": total_network_length,
            "steiner_network_length": steiner_network_length,
            "mst_length": mst_length,
        },
        "path_lengths": {str(k): v for k, v in path_lengths.items()},
        "steiner_edges": [
            {"p1": list(e[0]), "p2": list(e[1]), "length": e[2]}
            for e in steiner_edge_list
        ],
        "mst_edges": [
            {"p1": list(e[0]), "p2": list(e[1]), "length": e[2]}
            for e in mst_edges
        ],
    }
    
    json_filename = os.path.join(folder_name, f"{base_filename}_data.json")
    with open(json_filename, "w") as f:
        json.dump(experiment_data, f, indent=2)
    print(f"Experiment data exported to: {json_filename}")
    
    # =========================
    # PLOT GENERATION
    # =========================
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    ax.invert_yaxis()
    
    # Plot sinks
    for s in sinks:
        color = 'green' if touched[s.sid] else 'red'
        ax.scatter(s.pos.x, s.pos.y, c=color, s=SINK_MARKER_SIZE, marker='s', edgecolors='black', linewidths=1, zorder=5)
        ax.annotate(f'S{s.sid}', (s.pos.x + 10, s.pos.y), fontsize=7, zorder=6)
    
    # Plot source
    ax.scatter(source_pos.x, source_pos.y, c='lime', s=SOURCE_MARKER_SIZE, marker='*', edgecolors='black', linewidths=1, zorder=5)
    ax.annotate('SOURCE', (source_pos.x - 25, source_pos.y + 20), fontsize=9, fontweight='bold', zorder=6)
    
    # Plot connections (parent-child edges)
    for r in robots:
        if r.parent_id is not None and r.role == Role.NETWORK:
            parent = robots[r.parent_id]
            ax.plot([parent.pos.x, r.pos.x], [parent.pos.y, r.pos.y], 
                   'b-', linewidth=1.5, alpha=0.6, zorder=2)
    
    # Plot robot-to-sink connections
    for r in robots:
        if r.role == Role.NETWORK:
            for s in sinks:
                dist = (s.pos - r.pos).length()
                if dist <= SINK_TOUCH_DIST:
                    ax.plot([r.pos.x, s.pos.x], [r.pos.y, s.pos.y], 
                           'g--', linewidth=2, alpha=0.8, zorder=3)
    
    # Plot robots
    for r in robots:
        if r.role == Role.SOURCE:
            continue
        elif r.node_type == NodeType.PIVOT:
            ax.scatter(r.pos.x, r.pos.y, c='magenta', s=PIVOT_MARKER_SIZE, marker='o', edgecolors='black', linewidths=2, zorder=4)
            ax.annotate(f'P{r.rid}', (r.pos.x + 10, r.pos.y - 10), fontsize=7, color='magenta', zorder=6)
        elif r.role == Role.NETWORK:
            ax.scatter(r.pos.x, r.pos.y, c='royalblue', s=ROBOT_MARKER_SIZE, marker='o', edgecolors='black', linewidths=1, zorder=3)
            ax.annotate(f'{r.rid}', (r.pos.x + 8, r.pos.y - 5), fontsize=6, color='blue', alpha=0.7, zorder=6)
        else:
            ax.scatter(r.pos.x, r.pos.y, c='cyan', s=(ROBOT_MARKER_SIZE//2)*3, marker='o', alpha=0.5, zorder=2)
    
    # Plot void positions
    for r in robots:
        if r.role == Role.NETWORK and r.void_pos_vis is not None:
            ax.scatter(r.void_pos_vis.x, r.void_pos_vis.y, c='yellow', s=20, marker='x', alpha=0.5, zorder=1)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='lime', label='Source'),
        mpatches.Patch(color='magenta', label=f'Pivots ({len(pivot_robots)})'),
        mpatches.Patch(color='royalblue', label=f'Network Straight ({len(straight_robots)})'),
        mpatches.Patch(color='cyan', label=f'Moving ({len(moving_robots)})'),
        mpatches.Patch(color='green', label=f'Sink Touched ({touched_count})'),
        mpatches.Patch(color='red', label=f'Sink Untouched ({len(sinks)-touched_count})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Title and labels
    title = f"Network Topology - {len(network_robots)} Network Robots, {len(pivot_robots)} Pivots"
    if network_complete_time is not None:
        title += f"\nCompleted in {network_complete_time:.2f}s"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    
    # Calculate axis limits
    all_x = [s.pos.x for s in sinks] + [r.pos.x for r in robots] + [source_pos.x]
    all_y = [s.pos.y for s in sinks] + [r.pos.y for r in robots] + [source_pos.y]
    x_min, x_max = min(all_x) - 400, max(all_x) + 400
    y_min, y_max = min(all_y) - 400, max(all_y) + 400
    plot_width = max(x_max - x_min, 2000)
    plot_height = max(y_max - y_min, 3000)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    ax.set_xlim(x_center - plot_width/2, x_center + plot_width/2)
    ax.set_ylim(y_center + plot_height/2, y_center - plot_height/2)
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add configuration text box
    config_text = (
        f"Configuration:\n"
        f"R={R}, COMM={COMM_RANGE}\n"
        f"Sinks={NUM_SINKS}\n"
        f"Spawn={SPAWN_INTERVAL}s"
    )
    ax.text(0.02, 0.98, config_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(folder_name, f"{base_filename}_network_plot.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    
    # =========================
    # STEINER NETWORK PLOT
    # =========================
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
    ax2.invert_yaxis()
    
    # Plot Steiner edges
    for p1, p2, dist in steiner_edge_list:
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                'b-', linewidth=3, alpha=0.7, zorder=2)
        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax2.annotate(f'{dist:.0f}', (mid_x, mid_y), fontsize=7, color='darkblue', 
                    ha='center', va='center', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7), zorder=4)
    
    # Plot sinks
    for s in sinks:
        color = 'green' if touched[s.sid] else 'red'
        ax2.scatter(s.pos.x, s.pos.y, c=color, s=SINK_MARKER_SIZE + 20, marker='s', edgecolors='black', linewidths=1, zorder=5)
        ax2.annotate(f'S{s.sid}', (s.pos.x + 12, s.pos.y), fontsize=8, fontweight='bold', zorder=6)
    
    # Plot source
    ax2.scatter(source_pos.x, source_pos.y, c='lime', s=SOURCE_MARKER_SIZE + 20, marker='*', edgecolors='black', linewidths=2, zorder=5)
    ax2.annotate('SOURCE', (source_pos.x - 30, source_pos.y + 25), fontsize=10, fontweight='bold', zorder=6)
    
    # Plot pivots
    for p in pivot_robots:
        ax2.scatter(p.pos.x, p.pos.y, c='magenta', s=PIVOT_MARKER_SIZE, marker='o', edgecolors='black', linewidths=1, zorder=5)
        ax2.annotate(f'P{p.rid}', (p.pos.x + 10, p.pos.y - 10), fontsize=8, color='magenta', fontweight='bold', zorder=6)
    
    # Legend
    legend_elements2 = [
        mpatches.Patch(color='lime', label='Source'),
        mpatches.Patch(color='magenta', label=f'Pivots ({len(pivot_robots)})'),
        mpatches.Patch(color='green', label=f'Sink Touched ({touched_count})'),
        mpatches.Patch(color='red', label=f'Sink Untouched ({len(sinks)-touched_count})'),
        mpatches.Patch(color='blue', label=f'Steiner Edges ({len(steiner_edges)})'),
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=10)
    
    # Title
    title2 = f"STEINER NETWORK: Source + Pivots + Sinks\nTotal Length: {steiner_network_length:.2f} units"
    ax2.set_title(title2, fontsize=16, fontweight='bold')
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    
    ax2.set_xlim(x_center - plot_width/2, x_center + plot_width/2)
    ax2.set_ylim(y_center + plot_height/2, y_center - plot_height/2)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Add info text box
    info_text = (
        f"Steiner Network Summary:\n"
        f"Total Length: {steiner_network_length:.2f}\n"
        f"Pivots: {len(pivot_robots)}\n"
        f"Edges: {len(steiner_edges)}\n"
        f"Sinks Touched: {touched_count}/{len(sinks)}"
    )
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save Steiner plot
    steiner_plot_filename = os.path.join(folder_name, f"{base_filename}_steiner_plot.png")
    plt.savefig(steiner_plot_filename, dpi=150, bbox_inches='tight')
    print(f"Steiner plot saved to: {steiner_plot_filename}")
    
    # Show both plots
    plt.show()
