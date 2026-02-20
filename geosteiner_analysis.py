# ==============================================================
#  ROBOT SPAWNING WITH COSINE RULE AT JUNCTIONS
# ==============================================================

def _angle_at_B(A, B, C):
    """Angle at vertex B in the path A→B→C  (0..π)."""
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot  = BA[0]*BC[0] + BA[1]*BC[1]
    cross = BA[0]*BC[1] - BA[1]*BC[0]
    return math.atan2(abs(cross), dot)          # always 0..π


def _cosine_rule_next_d(d_ri_B, angle_at_B, r_desired):
    """Distance from junction B to the next robot R_{i+1} on the outgoing edge.

    Triangle  R_i – B – R_{i+1}:
        d(R_i, R_{i+1}) = r_desired
        d(R_i, B)        = d_ri_B
        angle at B        = angle_at_B

    Cosine rule → r² = d² + x² - 2·d·x·cos(θ)
    Solve for x (positive root closest to 0).
    """
    a = 1.0
    b = -2.0 * d_ri_B * math.cos(angle_at_B)
    c = d_ri_B ** 2 - r_desired ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return r_desired                         # fallback (very acute angle)
    sq = math.sqrt(disc)
    x1 = (-b + sq) / 2.0
    x2 = (-b - sq) / 2.0
    candidates = sorted(x for x in (x1, x2) if x > 1e-9)
    return candidates[0] if candidates else r_desired


def spawn_robots_on_tree(nodes, children, r_spacing=60):
    """Place robots on a rooted tree (root = index 0) using the cosine
    rule at every junction / corner.

    Algorithm
    ---------
    DFS from source.  Walk along each edge placing robots every *r_spacing*.
    When the last robot is closer to the child node than r_spacing, carry
    the residual into the next edge(s):

    * At a **corner** (MST sink that is a pass-through): use the cosine
      rule with the angle at the junction to find where the first robot
      on the next edge goes.
    * At a **pivot / Steiner point** (GeoSteiner): the same residual
      spawns one robot on *each* outgoing edge via the cosine rule,
      then each branch continues independently.

    Parameters
    ----------
    nodes     : list of (x, y) — index 0 is source.
    children  : dict  int → [int]  (rooted-tree adjacency from build_tree).
    r_spacing : desired spacing (R).

    Returns
    -------
    all_positions    : list of (x, y) for every spawned robot
    edge_robot_map   : dict (parent_idx, child_idx) → [(x,y), …]
    edge_robot_count : dict (parent_idx, child_idx) → int
    """
    all_positions    = []
    edge_robot_map   = {}
    edge_robot_count = {}

    def _dist(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _walk_edge(parent_idx, child_idx, last_robot_pos, d_last_to_node):
        """Place robots along edge parent→child.

        last_robot_pos : (x,y) of the most-recently-placed robot
                         (may be on the *previous* edge).
        d_last_to_node : dist(last_robot, parent_node).
        """
        p_pos = nodes[parent_idx]
        c_pos = nodes[child_idx]
        d_edge = _dist(p_pos, c_pos)
        if d_edge < 1e-9:
            return [], last_robot_pos, d_last_to_node

        # Unit direction parent → child
        dx = (c_pos[0] - p_pos[0]) / d_edge
        dy = (c_pos[1] - p_pos[1]) / d_edge

        # Compute offset to first robot on this edge via cosine rule
        if last_robot_pos is not None and d_last_to_node > 1e-9:
            angle = _angle_at_B(last_robot_pos, p_pos, c_pos)
            first_d = _cosine_rule_next_d(d_last_to_node, angle, r_spacing)
        else:
            # Source — first robot is R from source
            first_d = r_spacing

        edge_bots = []
        s = first_d
        while s < d_edge - 1e-9:
            rx = p_pos[0] + s * dx
            ry = p_pos[1] + s * dy
            edge_bots.append((rx, ry))
            s += r_spacing

        # Residual
        if edge_bots:
            d_to_child = _dist(edge_bots[-1], c_pos)
            return edge_bots, edge_bots[-1], d_to_child
        else:
            # No robot placed — the edge is shorter than first_d.
            # Residual: distance from last_robot to child_node
            if last_robot_pos is not None:
                d_to_child = _dist(last_robot_pos, c_pos)
            else:
                d_to_child = d_edge
            return edge_bots, last_robot_pos, d_to_child

    def _dfs(node_idx, last_robot_pos, d_last_to_node):
        for child_idx in children.get(node_idx, []):
            bots, new_last, d_to_child = _walk_edge(
                node_idx, child_idx, last_robot_pos, d_last_to_node
            )
            edge_robot_map[(node_idx, child_idx)]   = bots
            edge_robot_count[(node_idx, child_idx)] = len(bots)
            all_positions.extend(bots)
            # Recurse — each child branch gets its own copy of residual
            _dfs(child_idx, new_last, d_to_child)

    _dfs(0, None, 0)
    return all_positions, edge_robot_map, edge_robot_count


# ==============================================================
#  DRAWING HELPERS (use spawned positions)
# ==============================================================

def _draw_robots_on_edges(ax, edges_as_pos_pairs, r_spacing=60,
                          robot_color="#3a7bd5", robot_size=12):
    """Place and draw robot dots along a list of edges (legacy uniform)."""
    all_robot_pos = []
    for p1, p2 in edges_as_pos_pairs:
        d = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        if d < 1e-6:
            continue
        n = max(1, math.ceil(d / r_spacing))
        for k in range(1, n):
            t = k / n
            all_robot_pos.append((p1[0]+t*(p2[0]-p1[0]), p1[1]+t*(p2[1]-p1[1])))
    if all_robot_pos:
        rxs, rys = zip(*all_robot_pos)
        ax.scatter(rxs, rys, color=robot_color, s=robot_size,
                   zorder=4, alpha=0.8, edgecolors="white", linewidths=0.3)
    return all_robot_pos


def _draw_spawned_robots(ax, robot_positions, robot_color="#3a7bd5",
                         robot_size=12):
    """Draw pre-computed robot positions on an axes."""
    if robot_positions:
        rxs, rys = zip(*robot_positions)
        ax.scatter(rxs, rys, color=robot_color, s=robot_size,
                   zorder=4, alpha=0.8, edgecolors="white", linewidths=0.3)


def _build_rooted_tree(all_nodes, pos_edges):
    """Build a rooted tree (root=index 0) from undirected position-based edges.

    Returns children dict  int -> [int].
    """
    from collections import defaultdict, deque
    TOL = 1e-3

    def _idx(pos):
        for i, nd in enumerate(all_nodes):
            if abs(nd[0] - pos[0]) < TOL and abs(nd[1] - pos[1]) < TOL:
                return i
        return None

    adj = defaultdict(set)
    for p1, p2 in pos_edges:
        i, j = _idx(p1), _idx(p2)
        if i is not None and j is not None:
            adj[i].add(j)
            adj[j].add(i)

    children = defaultdict(list)
    visited = {0}
    q = deque([0])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in visited:
                visited.add(v)
                children[u].append(v)
                q.append(v)
    return dict(children)


def _draw_leaf_to_sink(ax, robots, sinks):
    """Draw dashed lines from leaf robots to their nearest sink.

    Leaf robots are those with assigned_sinks == [] (last robot before a sink).
    For each sink, find the nearest robot and draw a dashed connector.
    """
    # Build set of rids that are parents (i.e. have children)
    parent_rids = {r["parent"] for r in robots.values() if r["parent"] is not None}
    # Leaf robots: not a parent of anyone
    leaf_rids = [rid for rid in robots if rid not in parent_rids]

    for sid, s in sinks.items():
        sp = s["pos"]
        best_rid, best_d = None, float("inf")
        for rid in leaf_rids:
            d = math.sqrt((robots[rid]["pos"][0] - sp[0])**2 +
                          (robots[rid]["pos"][1] - sp[1])**2)
            if d < best_d:
                best_d, best_r = d, rid
                best_rid = rid
        if best_rid is not None:
            rp = robots[best_rid]["pos"]
            ax.plot([rp[0], sp[0]], [rp[1], sp[1]],
                    color="#ff6b6b", linewidth=1.2, linestyle="--", alpha=0.8, zorder=2)


def plot_three_methods(data, terminals, mst_edges, smt_steiner_pts, smt_edges_idx):
    src    = data["source"]
    sinks  = data["sinks"]
    robots = data["robots"]
    pivots = {p["id"]: p for p in data["pivots"]}
    all_nodes = terminals + smt_steiner_pts
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.patch.set_facecolor("#0f0f1a")
    titles = ["Robot Network (Simulation Output)", "Minimum Spanning Tree (MST)", "GeoSteiner SMT (Optimal)"]

    # Pre-compute spawned robot positions for MST and GeoSteiner
    mst_ch = _build_rooted_tree(terminals, mst_edges)
    mst_spawned, _, _ = spawn_robots_on_tree(terminals, mst_ch)

    geo_pos_edges = []
    if smt_edges_idx and all_nodes:
        for (i, j) in smt_edges_idx:
            if i < len(all_nodes) and j < len(all_nodes):
                geo_pos_edges.append((all_nodes[i], all_nodes[j]))
    geo_ch = _build_rooted_tree(all_nodes, geo_pos_edges)
    geo_spawned, _, _ = spawn_robots_on_tree(all_nodes, geo_ch)
    # Panel 1: Robot Network
    ax1 = axes[0]
    ax1.set_facecolor("#0f0f1a")
    ax1.set_title(titles[0], color="white", fontsize=14)
    for rid, robot in robots.items():
        if robot["parent"] is not None and robot["parent"] in robots:
            p1 = robot["pos"]
            p2 = robots[robot["parent"]]["pos"]
            color = "#3a7bd5" if robot["type"] == "STRAIGHT" else "#f9ca24"
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=0.8, alpha=0.6, zorder=1)
    straights = [(r["pos"][0], r["pos"][1]) for r in robots.values() if r["type"] == "STRAIGHT"]
    if straights:
        xs, ys = zip(*straights)
        ax1.scatter(xs, ys, color="#3a7bd5", s=12, zorder=3, label="Straight robots")
    for pid, pv in pivots.items():
        ax1.scatter(*pv["pos"], color="#f9ca24", s=80, zorder=5, marker="D", edgecolors="white", linewidths=0.5)
        ax1.annotate(f"P{pid}", pv["pos"], textcoords="offset points", xytext=(6, 4), color="#f9ca24", fontsize=7)
    for sid, s in sinks.items():
        ax1.scatter(*s["pos"], color="#ff4757", s=120, zorder=6, marker="*", edgecolors="white", linewidths=0.5)
        ax1.annotate(f"S{sid}", s["pos"], textcoords="offset points", xytext=(6, 4), color="#ff4757", fontsize=9, fontweight="bold")
    ax1.scatter(*src, color="#2ed573", s=200, zorder=7, marker="^", edgecolors="white", linewidths=1)
    ax1.annotate("SOURCE", src, textcoords="offset points", xytext=(8, 4), color="#2ed573", fontsize=9, fontweight="bold")
    _draw_leaf_to_sink(ax1, robots, sinks)
    _style_ax(ax1)
    # Panel 2: MST
    ax2 = axes[1]
    ax2.set_facecolor("#0f0f1a")
    ax2.set_title(titles[1], color="white", fontsize=14)
    for (p1, p2) in mst_edges:
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#a29bfe", linewidth=2, linestyle="--", alpha=0.8)
    _draw_spawned_robots(ax2, mst_spawned, robot_color="#a29bfe")
    xs, ys = zip(*terminals)
    ax2.scatter(xs, ys, color="#ff4757", s=120, zorder=6, marker="*")
    ax2.scatter(*terminals[0], color="#2ed573", s=200, zorder=7, marker="^")
    ax2.annotate("SOURCE", terminals[0], textcoords="offset points", xytext=(8, 4), color="#2ed573", fontsize=9, fontweight="bold")
    for i, pos in enumerate(terminals[1:], 1):
        ax2.annotate(f"S{i}", pos, textcoords="offset points", xytext=(6, 4), color="#ff4757", fontsize=9, fontweight="bold")
    _style_ax(ax2)
    # Panel 3: GeoSteiner SMT
    ax3 = axes[2]
    ax3.set_facecolor("#0f0f1a")
    ax3.set_title(titles[2], color="white", fontsize=14)
    for (p1, p2) in geo_pos_edges:
        ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#fd79a8", linewidth=2.0, alpha=0.9)
    _draw_spawned_robots(ax3, geo_spawned, robot_color="#fd79a8")
    if smt_steiner_pts:
        sx, sy = zip(*smt_steiner_pts)
        ax3.scatter(sx, sy, color="#f9ca24", s=80, zorder=5, marker="D", edgecolors="white", linewidths=0.5, label="Steiner pts (pivot)")
        for si, sp in enumerate(smt_steiner_pts):
            ax3.annotate(f"ST{si}", sp, textcoords="offset points", xytext=(6, 4), color="#f9ca24", fontsize=7)
    xs, ys = zip(*terminals)
    ax3.scatter(xs, ys, color="#ff4757", s=120, zorder=6, marker="*")
    ax3.scatter(*terminals[0], color="#2ed573", s=200, zorder=7, marker="^")
    ax3.annotate("SOURCE", terminals[0], textcoords="offset points", xytext=(8, 4), color="#2ed573", fontsize=9, fontweight="bold")
    for i, pos in enumerate(terminals[1:], 1):
        ax3.annotate(f"S{i}", pos, textcoords="offset points", xytext=(6, 4), color="#ff4757", fontsize=9, fontweight="bold")
    _style_ax(ax3)
    plt.tight_layout()
    plt.savefig("three_methods_plot.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Figure saved → three_methods_plot.png")
    plt.show()
def plot_mst(terminals, mst_edges):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#0f0f1a")
    ax.set_title("Minimum Spanning Tree (MST)", color="white", fontsize=14)
    for (p1, p2) in mst_edges:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#a29bfe", linewidth=2, linestyle="--", alpha=0.8)
    mst_ch = _build_rooted_tree(terminals, mst_edges)
    mst_spawned, _, _ = spawn_robots_on_tree(terminals, mst_ch)
    _draw_spawned_robots(ax, mst_spawned, robot_color="#a29bfe")
    xs, ys = zip(*terminals)
    ax.scatter(xs, ys, color="#ff4757", s=120, zorder=6, marker="*")
    ax.scatter(*terminals[0], color="#2ed573", s=200, zorder=7, marker="^")
    ax.annotate("SOURCE", terminals[0], textcoords="offset points", xytext=(8, 4), color="#2ed573", fontsize=9, fontweight="bold")
    for i, pos in enumerate(terminals[1:], 1):
        ax.annotate(f"S{i}", pos, textcoords="offset points", xytext=(6, 4), color="#ff4757", fontsize=9, fontweight="bold")
    _style_ax(ax)
    plt.tight_layout()
    plt.savefig("mst_plot.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Figure saved → mst_plot.png")
    plt.close(fig)

def plot_geosteiner(terminals, smt_steiner_pts, smt_edges_idx):
    all_nodes = terminals + smt_steiner_pts
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#0f0f1a")
    ax.set_title("GeoSteiner SMT (Optimal Steiner Tree)", color="white", fontsize=14)
    geo_pos_edges = []
    if smt_edges_idx and all_nodes:
        for (i, j) in smt_edges_idx:
            if i < len(all_nodes) and j < len(all_nodes):
                p1, p2 = all_nodes[i], all_nodes[j]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#fd79a8", linewidth=2.0, alpha=0.9)
                geo_pos_edges.append((p1, p2))
    geo_ch = _build_rooted_tree(all_nodes, geo_pos_edges)
    geo_spawned, _, _ = spawn_robots_on_tree(all_nodes, geo_ch)
    _draw_spawned_robots(ax, geo_spawned, robot_color="#fd79a8")
    if smt_steiner_pts:
        sx, sy = zip(*smt_steiner_pts)
        ax.scatter(sx, sy, color="#f9ca24", s=80, zorder=5, marker="D", edgecolors="white", linewidths=0.5, label="Steiner pts (pivot)")
        for si, sp in enumerate(smt_steiner_pts):
            ax.annotate(f"ST{si}", sp, textcoords="offset points", xytext=(6, 4), color="#f9ca24", fontsize=7)
    xs, ys = zip(*terminals)
    ax.scatter(xs, ys, color="#ff4757", s=120, zorder=6, marker="*")
    ax.scatter(*terminals[0], color="#2ed573", s=200, zorder=7, marker="^")
    ax.annotate("SOURCE", terminals[0], textcoords="offset points", xytext=(8, 4), color="#2ed573", fontsize=9, fontweight="bold")
    for i, pos in enumerate(terminals[1:], 1):
        ax.annotate(f"S{i}", pos, textcoords="offset points", xytext=(6, 4), color="#ff4757", fontsize=9, fontweight="bold")
    _style_ax(ax)
    plt.tight_layout()
    plt.savefig("geosteiner_plot.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Figure saved → geosteiner_plot.png")
    plt.close(fig)

def plot_robot_network(data):
    src    = data["source"]
    sinks  = data["sinks"]
    robots = data["robots"]
    pivots = {p["id"]: p for p in data["pivots"]}
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#0f0f1a")
    ax.set_title("Robot Network (Simulation Output)", color="white", fontsize=14)
    for rid, robot in robots.items():
        if robot["parent"] is not None and robot["parent"] in robots:
            p1 = robot["pos"]
            p2 = robots[robot["parent"]]["pos"]
            color = "#3a7bd5" if robot["type"] == "STRAIGHT" else "#f9ca24"
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=0.8, alpha=0.6, zorder=1)
    straights = [(r["pos"][0], r["pos"][1]) for r in robots.values() if r["type"] == "STRAIGHT"]
    if straights:
        xs, ys = zip(*straights)
        ax.scatter(xs, ys, color="#3a7bd5", s=12, zorder=3, label="Straight robots")
    for pid, pv in pivots.items():
        ax.scatter(*pv["pos"], color="#f9ca24", s=80, zorder=5, marker="D", edgecolors="white", linewidths=0.5)
        ax.annotate(f"P{pid}", pv["pos"], textcoords="offset points", xytext=(6, 4), color="#f9ca24", fontsize=7)
    for sid, s in sinks.items():
        ax.scatter(*s["pos"], color="#ff4757", s=120, zorder=6, marker="*", edgecolors="white", linewidths=0.5)
        ax.annotate(f"S{sid}", s["pos"], textcoords="offset points", xytext=(6, 4), color="#ff4757", fontsize=9, fontweight="bold")
    ax.scatter(*src, color="#2ed573", s=200, zorder=7, marker="^", edgecolors="white", linewidths=1)
    ax.annotate("SOURCE", src, textcoords="offset points", xytext=(8, 4), color="#2ed573", fontsize=9, fontweight="bold")
    _draw_leaf_to_sink(ax, robots, sinks)
    _style_ax(ax)
    plt.tight_layout()
    plt.savefig("robot_network_plot.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Figure saved → robot_network_plot.png")
    plt.close(fig)
"""
=============================================================
 GeoSteiner Analysis & Visualization Tool
 For: Robot Swarm Network Formation Research (IROS 2026)
=============================================================
 Requirements:
   pip install matplotlib numpy scipy

 GeoSteiner must be installed and 'esmt' binary accessible.
 Update GEOSTEINER_PATH below to point to your binary.
=============================================================
"""

import re
import math
import subprocess
import sys
import os
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────
#  CONFIG  ← adjust these two paths for your machine
# ─────────────────────────────────────────────────────────────
REPORT_FILE     = "3sinks_60R_20260219_120808_report.txt"   # path to your report
GEOSTEINER_PATH = r"C:\Users\rashi\network_problem\geosteiner folder\geosteiner-5.3\esmt.exe"  # path to GeoSteiner binary
# ─────────────────────────────────────────────────────────────


# =============================================================
#  SECTION 1 — PARSE REPORT FILE
# =============================================================

def parse_report(filepath: str) -> dict:
    """
    Reads a simulation report and extracts all useful data:
    source position, sink positions, pivot robots, straight
    robots, and key metrics.
    """
    with open(filepath, "r") as f:
        text = f.read()

    data = {}

    # --- Source position ---
    src_match = re.search(r"Source Position:\s*\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)", text)
    if not src_match:
        raise ValueError("Could not find Source Position in report.")
    data["source"] = (float(src_match.group(1)), float(src_match.group(2)))

    # --- Sink positions ---
    sink_pattern = re.compile(
        r"S(\d+)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+(\w+)"
    )
    data["sinks"] = {}
    for m in sink_pattern.finditer(text):
        sid    = int(m.group(1))
        x, y   = float(m.group(2)), float(m.group(3))
        status = m.group(4)
        data["sinks"][sid] = {"pos": (x, y), "status": status}

    # --- Pivot robots ---
    pivot_pattern = re.compile(
        r"(\d+)\s+\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)\s+(\S+)\s+(\[.*?\])\s+(\[.*?\])",
        re.MULTILINE,
    )
    data["pivots"] = []
    pivot_section = re.search(
        r"PIVOT ROBOTS DETAILS.*?(?=\n-{5,})", text, re.DOTALL
    )
    if pivot_section:
        for m in pivot_pattern.finditer(pivot_section.group()):
            data["pivots"].append({
                "id":       int(m.group(1)),
                "pos":      (float(m.group(2)), float(m.group(3))),
                "branch":   m.group(4),
                "children": m.group(5),
                "sinks":    m.group(6),
            })

    # --- Full robot roster (for drawing edges) ---
    roster_pattern = re.compile(
        r"^\s*(\d+)\s+(SOURCE|STRAIGHT|PIVOT)\s+(\S+)\s+(---|\d+)\s+(\[.*?\])\s+\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)",
        re.MULTILINE,
    )
    data["robots"] = {}
    for m in roster_pattern.finditer(text):
        rid    = int(m.group(1))
        rtype  = m.group(2)
        branch = m.group(3)
        parent = None if m.group(4) == "---" else int(m.group(4))
        sinks_str = m.group(5)  # e.g. "[0, 1]" or "[]"
        assigned_sinks = ast.literal_eval(sinks_str)
        x, y   = float(m.group(6)), float(m.group(7))
        data["robots"][rid] = {
            "type": rtype, "branch": branch,
            "parent": parent, "pos": (x, y),
            "assigned_sinks": assigned_sinks,
        }

    # --- Key scalar metrics ---
    def grab(label):
        m = re.search(rf"{re.escape(label)}\s+([+-]?\d+\.?\d*)", text)
        return float(m.group(1)) if m else None

    data["robot_network_length"] = grab("Total Network Length (tree edges)")
    if data["robot_network_length"] is None:
        data["robot_network_length"] = grab("Total Network Length (incl. sinks)")
    data["steiner_network_length"] = grab(r"\*\*\* STEINER NETWORK LENGTH:")
    if data["steiner_network_length"] is None:
        m = re.search(r"STEINER NETWORK LENGTH:\s*([0-9.]+)", text)
        data["steiner_network_length"] = float(m.group(1)) if m else None
    data["mst_length"] = grab(r"\*\*\* MINIMUM SPANNING TREE LENGTH:")
    if data["mst_length"] is None:
        m = re.search(r"MINIMUM SPANNING TREE LENGTH:\s*([0-9.]+)", text)
        data["mst_length"] = float(m.group(1)) if m else None

    return data


# =============================================================
#  SECTION 2 — EUCLIDEAN DISTANCE HELPERS
# =============================================================

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def compute_mst_length(points):
    """Prim's algorithm for Euclidean MST."""
    n = len(points)
    in_mst   = [False] * n
    min_edge = [float("inf")] * n
    parent   = [-1] * n
    min_edge[0] = 0.0
    total = 0.0
    edges = []

    for _ in range(n):
        u = min((v for v in range(n) if not in_mst[v]), key=lambda v: min_edge[v])
        in_mst[u] = True
        total += min_edge[u]
        if parent[u] != -1:
            edges.append((points[parent[u]], points[u]))
        for v in range(n):
            if not in_mst[v]:
                d = dist(points[u], points[v])
                if d < min_edge[v]:
                    min_edge[v] = d
                    parent[v]   = u

    return total, edges


# =============================================================
#  SECTION 3 — GEOSTEINER INTERFACE
# =============================================================

def run_geosteiner(points, binary_path=GEOSTEINER_PATH):
    """
    Calls the GeoSteiner 'esmt' binary with the given terminal
    points.  Returns (smt_length, steiner_points, edges) or
    (None, [], []) if the binary is not found.

    Input  : list of (x, y) tuples — ALL terminals (source + sinks)
    Output : optimal Steiner Minimum Tree length
    """
    if not os.path.isfile(binary_path):
        print(f"\n[WARNING] GeoSteiner binary not found at '{binary_path}'.")
        print("          Install GeoSteiner and update GEOSTEINER_PATH.")
        print("          Skipping GeoSteiner computation.\n")
        return None, [], []

    n = len(points)
    stdin_str = f"{n}\n" + "\n".join(f"{x} {y}" for x, y in points) + "\n"

    # Build environment with MSYS2 MinGW64 bin in PATH so DLLs are found
    env = os.environ.copy()
    msys2_bin = r"C:\msys64\mingw64\bin"
    if os.path.isdir(msys2_bin):
        env["PATH"] = msys2_bin + ";" + env.get("PATH", "")

    try:
        result = subprocess.run(
            [binary_path],
            input=stdin_str,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=os.path.dirname(binary_path) or None,
        )
    except subprocess.TimeoutExpired:
        print("[ERROR] GeoSteiner timed out.")
        return None, [], []
    except Exception as e:
        print(f"[ERROR] Failed to run GeoSteiner: {e}")
        return None, [], []

    # Debug: print any errors from the binary
    if result.returncode != 0:
        print(f"  [DEBUG] esmt.exe return code: {result.returncode}")
    if result.stderr and result.stderr.strip():
        print(f"  [DEBUG] esmt.exe stderr: {result.stderr.strip()}")

    output = result.stdout
    if not output.strip():
        print("  [WARNING] GeoSteiner produced no output.")
        print(f"  [DEBUG] Return code: {result.returncode}")
        return None, [], []

    smt_length    = None
    steiner_pts   = []
    smt_edges     = []

    for line in output.splitlines():
        # Length line — GeoSteiner prints something like "Length = 1234.56"
        m = re.search(r"[Ll]ength\s*[=:]\s*([0-9.]+)", line)
        if m:
            smt_length = float(m.group(1))

        # Steiner point lines  e.g.  "s  x  y"
        m = re.match(r"^\s*s\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)", line)
        if m:
            steiner_pts.append((float(m.group(1)), float(m.group(2))))

        # Edge lines  e.g.  "e  i  j"  (indices into terminal + Steiner array)
        m = re.match(r"^\s*e\s+(\d+)\s+(\d+)", line)
        if m:
            smt_edges.append((int(m.group(1)), int(m.group(2))))

    if smt_length is not None:
        print(f"  GeoSteiner returned: length={smt_length:.4f}, "
              f"{len(steiner_pts)} Steiner points, {len(smt_edges)} edges")
    else:
        print("  [WARNING] Could not parse GeoSteiner output.")
        print(f"  [DEBUG] stdout: {output[:500]}")

    return smt_length, steiner_pts, smt_edges


# =============================================================
#  SECTION 4 — PRINT RESULTS TO TERMINAL
# =============================================================

def print_results(data, geosteiner_length):
    src   = data["source"]
    sinks = data["sinks"]

    print("\n" + "="*60)
    print("  DISTANCE ANALYSIS RESULTS")
    print("="*60)
    print(f"\n  Source : {src}")
    print(f"\n  Sinks:")
    for sid, s in sinks.items():
        d = dist(src, s["pos"])
        print(f"    S{sid}  {s['pos']}   dist from source = {d:.2f} units")

    all_pts = [src] + [s["pos"] for s in sinks.values()]

    print(f"\n  ── Pairwise Distances (Source ↔ Each Sink) ──")
    for sid, s in sinks.items():
        print(f"    Source → S{sid} : {dist(src, s['pos']):.2f}")

    print(f"\n  ── All Sink-to-Sink Distances ──")
    sink_list = list(sinks.items())
    for i in range(len(sink_list)):
        for j in range(i+1, len(sink_list)):
            sid_a, sa = sink_list[i]
            sid_b, sb = sink_list[j]
            print(f"    S{sid_a} → S{sid_b} : {dist(sa['pos'], sb['pos']):.2f}")

    mst_len, _ = compute_mst_length(all_pts)

    print(f"\n  ── Network Length Comparison ──")
    print(f"    MST  (Euclidean, recomputed)  : {mst_len:.2f} units")
    print(f"    MST  (from report)            : {data['mst_length']:.2f} units")
    print(f"    Steiner Network (robot proxy) : {data['steiner_network_length']:.2f} units")
    print(f"    Robot Network (actual)        : {data['robot_network_length']:.2f} units")
    if geosteiner_length:
        print(f"    GeoSteiner SMT (optimal)      : {geosteiner_length:.2f} units")
        print(f"\n  ── Optimality Ratios ──")
        print(f"    Robot / MST                   : {data['robot_network_length']/mst_len:.4f}")
        print(f"    Robot / GeoSteiner            : {data['robot_network_length']/geosteiner_length:.4f}")
        print(f"    GeoSteiner / MST              : {geosteiner_length/mst_len:.4f}")
    else:
        print(f"\n  ── Optimality Ratios (GeoSteiner not available) ──")
        print(f"    Robot / MST                   : {data['robot_network_length']/mst_len:.4f}")
    print()


# =============================================================
#  SECTION 5 — VISUALIZATION
# =============================================================

def visualize(data, mst_edges, geosteiner_length=None, smt_steiner_pts=[], smt_edges_idx=[]):
    """
    Produces a 2-panel figure:
      Left  — Robot network (all robots + edges)
      Right — MST vs GeoSteiner overlay + bar chart of lengths
    """
    src    = data["source"]
    sinks  = data["sinks"]
    robots = data["robots"]
    pivots = {p["id"]: p for p in data["pivots"]}

    # Build terminal list for GeoSteiner index mapping
    terminals = [src] + [s["pos"] for s in sinks.values()]
    all_nodes  = terminals + smt_steiner_pts   # GeoSteiner indexes into this

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("#0f0f1a")
    plt.suptitle(
        "Robot Swarm Network — GeoSteiner Analysis",
        color="white", fontsize=15, fontweight="bold", y=1.01
    )

    # ── Panel 1: Robot network ────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor("#0f0f1a")
    ax1.set_title("Robot Network (Simulation Output)", color="white", fontsize=12)

    # Draw robot chain edges
    for rid, robot in robots.items():
        if robot["parent"] is not None and robot["parent"] in robots:
            p1 = robot["pos"]
            p2 = robots[robot["parent"]]["pos"]
            color = "#3a7bd5" if robot["type"] == "STRAIGHT" else "#f9ca24"
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]],
                     color=color, linewidth=0.8, alpha=0.6, zorder=1)

    # Draw straight robots
    straights = [(r["pos"][0], r["pos"][1])
                 for r in robots.values() if r["type"] == "STRAIGHT"]
    if straights:
        xs, ys = zip(*straights)
        ax1.scatter(xs, ys, color="#3a7bd5", s=12, zorder=3, label="Straight robots")

    # Draw pivot robots
    for pid, pv in pivots.items():
        ax1.scatter(*pv["pos"], color="#f9ca24", s=80, zorder=5,
                    marker="D", edgecolors="white", linewidths=0.5)
        ax1.annotate(f"P{pid}", pv["pos"],
                     textcoords="offset points", xytext=(6, 4),
                     color="#f9ca24", fontsize=7)

    # Draw sinks
    for sid, s in sinks.items():
        ax1.scatter(*s["pos"], color="#ff4757", s=120, zorder=6,
                    marker="*", edgecolors="white", linewidths=0.5)
        ax1.annotate(f"S{sid}", s["pos"],
                     textcoords="offset points", xytext=(6, 4),
                     color="#ff4757", fontsize=9, fontweight="bold")

    # Draw source
    ax1.scatter(*src, color="#2ed573", s=200, zorder=7,
                marker="^", edgecolors="white", linewidths=1)
    ax1.annotate("SOURCE", src,
                 textcoords="offset points", xytext=(8, 4),
                 color="#2ed573", fontsize=9, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color="#3a7bd5", label="Straight robots"),
        mpatches.Patch(color="#f9ca24", label="Pivot robots"),
        mpatches.Patch(color="#ff4757", label="Sinks"),
        mpatches.Patch(color="#2ed573", label="Source"),
    ]
    ax1.legend(handles=legend_handles, loc="upper right",
               facecolor="#1a1a2e", edgecolor="gray",
               labelcolor="white", fontsize=8)
    _draw_leaf_to_sink(ax1, robots, sinks)
    _style_ax(ax1)

    # ── Panel 2: Network comparison ───────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#0f0f1a")
    ax2.set_title("Network Comparison", color="white", fontsize=12)

    # MST edges
    for (p1, p2) in mst_edges:
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color="#a29bfe", linewidth=1.5, linestyle="--",
                 alpha=0.7, zorder=2)

    # GeoSteiner edges (if available)
    if smt_edges_idx and all_nodes:
        for (i, j) in smt_edges_idx:
            if i < len(all_nodes) and j < len(all_nodes):
                p1, p2 = all_nodes[i], all_nodes[j]
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                         color="#fd79a8", linewidth=2.0,
                         alpha=0.9, zorder=3)

    # GeoSteiner Steiner points
    if smt_steiner_pts:
        sx, sy = zip(*smt_steiner_pts)
        ax2.scatter(sx, sy, color="#fd79a8", s=80, zorder=5,
                    marker="o", edgecolors="white", linewidths=0.5,
                    label="GeoSteiner points")

    # Sinks and source
    for sid, s in sinks.items():
        ax2.scatter(*s["pos"], color="#ff4757", s=120, zorder=6, marker="*")
        ax2.annotate(f"S{sid}", s["pos"],
                     textcoords="offset points", xytext=(6, 4),
                     color="#ff4757", fontsize=9, fontweight="bold")
    ax2.scatter(*src, color="#2ed573", s=200, zorder=7, marker="^")
    ax2.annotate("SOURCE", src,
                 textcoords="offset points", xytext=(8, 4),
                 color="#2ed573", fontsize=9, fontweight="bold")

    # Legend
    legend2 = [
        mpatches.Patch(color="#a29bfe", label=f"MST  ({data['mst_length']:.0f} units)"),
    ]
    if geosteiner_length:
        legend2.append(
            mpatches.Patch(color="#fd79a8",
                           label=f"GeoSteiner SMT ({geosteiner_length:.0f} units)")
        )
    ax2.legend(handles=legend2, loc="upper right",
               facecolor="#1a1a2e", edgecolor="gray",
               labelcolor="white", fontsize=8)
    _style_ax(ax2)

    plt.tight_layout()
    plt.savefig("network_analysis.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Figure saved → network_analysis.png")
    plt.show()

    # ── Bar chart: length comparison ─────────────────────────
    _bar_chart(data, geosteiner_length)


def _bar_chart(data, geosteiner_length):
    labels  = ["MST\n(theoretical)", "Robot Network\n(simulation)"]
    values  = [data["mst_length"], data["robot_network_length"]]
    colors  = ["#a29bfe", "#3a7bd5"]

    if geosteiner_length:
        labels.insert(1, "GeoSteiner SMT\n(optimal)")
        values.insert(1, geosteiner_length)
        colors.insert(1, "#fd79a8")

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    bars = ax.bar(labels, values, color=colors, width=0.5,
                  edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{val:.1f}", ha="center", va="bottom",
                color="white", fontsize=10, fontweight="bold")

    ax.set_title("Network Length Comparison", color="white",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Length (units)", color="white")
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("gray")
    ax.yaxis.label.set_color("white")
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig("length_comparison.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Figure saved → length_comparison.png")
    plt.show()


def _style_ax(ax):
    ax.tick_params(colors="white")
    ax.spines[["top", "right", "left", "bottom"]].set_color("gray")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="gray", alpha=0.15, linewidth=0.5)


# =============================================================
#  MAIN
# =============================================================

if __name__ == "__main__":

    report_path = sys.argv[1] if len(sys.argv) > 1 else REPORT_FILE

    print(f"\n  Loading report: {report_path}")
    data = parse_report(report_path)

    src        = data["source"]
    sinks      = data["sinks"]
    terminals  = [src] + [s["pos"] for s in sinks.values()]

    # Recompute MST for edge drawing
    mst_len, mst_edges = compute_mst_length(terminals)

    # Run GeoSteiner
    print(f"  Running GeoSteiner on {len(terminals)} terminals …")
    geo_len, smt_steiner_pts, smt_edges_idx = run_geosteiner(terminals)

    # Print results
    print_results(data, geo_len)

    # Visualize
    visualize(data, mst_edges, geo_len, smt_steiner_pts, smt_edges_idx)

    # Save three separate plots
    plot_mst(terminals, mst_edges)
    plot_geosteiner(terminals, smt_steiner_pts, smt_edges_idx)
    plot_robot_network(data)

    # Save big plot with three subplots
    plot_three_methods(data, terminals, mst_edges, smt_steiner_pts, smt_edges_idx)