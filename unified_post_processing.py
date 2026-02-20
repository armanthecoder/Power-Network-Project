"""
Unified Post-Processing Analysis Tool
======================================
Reads a simulation output text file and performs complete analysis:
  1. MST computation and cosine-rule robot spawning
  2. GeoSteiner SMT computation and robot spawning
  3. Power loss analysis for all three methods
  4. Network length comparisons (with leaf residual gaps)
  5. Visualization plots and comparison report

Correct behaviours (unified from best of both modules):
  - Robots stop BEFORE child node  (s < d_edge - 1e-9)
  - Source robot is NOT counted in robot totals / power loss
  - Leaf residual gap IS added to network lengths
  - Robust angle method (atan2) for cosine rule at junctions

Usage:
    python unified_post_processing.py                     # auto-discover all _simdata.txt in local folders
    python unified_post_processing.py <path_to_simdata.txt>  # process a single file
"""

import re
import math
import os
import sys
import ast
import json
import glob
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, deque
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
GEOSTEINER_PATH = (
    r"C:\Users\rashi\network_problem\geosteiner folder"
    r"\geosteiner-5.3\esmt.exe"
)


# =============================================================
#  SECTION 1 — EUCLIDEAN HELPERS
# =============================================================

def dist(p1, p2):
    """Euclidean distance between two (x, y) tuples."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# =============================================================
#  SECTION 2 — PARSE SIMULATION OUTPUT
# =============================================================

def parse_simulation_output(filepath: str) -> dict:
    """Parse the simulation output text file produced by
    post_processing_paper_version.py.

    Returns a dict with keys:
        random_seed, source, config, sinks, metrics, robots
    """
    with open(filepath, "r") as f:
        text = f.read()

    data: dict = {}

    # ── Random seed ───────────────────────────────────────────
    m = re.search(r"RANDOM SEED\s*\n-+\s*\n\s*(\d+)", text)
    data["random_seed"] = int(m.group(1)) if m else None

    # ── Source position ───────────────────────────────────────
    m = re.search(
        r"Source Position:\s*\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)", text
    )
    if not m:
        raise ValueError("Could not find Source Position in report.")
    data["source"] = (float(m.group(1)), float(m.group(2)))

    # ── Configuration parameters ──────────────────────────────
    cfg: dict = {}
    for label, key in [
        (r"R \(Void Radius\)", "R"),
        (r"COMM_RANGE", "COMM_RANGE"),
        (r"SPEED_MAX", "SPEED_MAX"),
        (r"SINK_TOUCH_DIST", "SINK_TOUCH_DIST"),
        (r"NUM_SINKS", "NUM_SINKS"),
        (r"SPAWN_INTERVAL", "SPAWN_INTERVAL"),
        (r"FPS", "FPS"),
        (r"WIDTH", "WIDTH"),
        (r"HEIGHT", "HEIGHT"),
    ]:
        p = re.search(rf"{label}\s+([+-]?\d+\.?\d*)", text)
        if p:
            val = float(p.group(1))
            cfg[key] = int(val) if val == int(val) else val
    data["config"] = cfg

    # ── Sink positions ────────────────────────────────────────
    sink_pat = re.compile(
        r"S(\d+)\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+(\w+)"
    )
    data["sinks"] = {}
    for m in sink_pat.finditer(text):
        sid = int(m.group(1))
        data["sinks"][sid] = {
            "pos": (float(m.group(2)), float(m.group(3))),
            "status": m.group(4),
        }

    # ── Metrics ───────────────────────────────────────────────
    metrics: dict = {}
    p = re.search(r"Total Simulation Time\s+([0-9.]+)\s+seconds", text)
    metrics["sim_time"] = float(p.group(1)) if p else None
    p = re.search(r"Total Simulation Steps\s+(\d+)", text)
    metrics["sim_steps"] = int(p.group(1)) if p else None
    p = re.search(r"Network Completion Time\s+([0-9.]+)\s+seconds", text)
    metrics["network_complete_time"] = float(p.group(1)) if p else None
    data["metrics"] = metrics

    # ── Robot roster ──────────────────────────────────────────
    roster_pat = re.compile(
        r"^\s*(\d+)\s+(SOURCE|STRAIGHT|PIVOT)\s+(\S+)\s+(---|\d+)"
        r"\s+(\[.*?\])\s+"
        r"\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)",
        re.MULTILINE,
    )
    data["robots"] = {}
    for m in roster_pat.finditer(text):
        rid = int(m.group(1))
        data["robots"][rid] = {
            "type": m.group(2),
            "branch": m.group(3),
            "parent": None if m.group(4) == "---" else int(m.group(4)),
            "assigned_sinks": ast.literal_eval(m.group(5)),
            "pos": (float(m.group(6)), float(m.group(7))),
        }

    return data


# =============================================================
#  SECTION 3 — MST (Prim's Algorithm)
# =============================================================

def compute_mst(points):
    """Prim's MST on a list of (x, y) points.

    Returns
    -------
    mst_length : float
    mst_edges  : list of ((x1,y1), (x2,y2))
    parent_arr : list[int]  (index-based parent array, -1 for root)
    """
    n = len(points)
    in_mst = [False] * n
    min_edge = [float("inf")] * n
    parent_arr = [-1] * n
    min_edge[0] = 0.0
    total = 0.0
    edges = []

    for _ in range(n):
        u = min(
            (v for v in range(n) if not in_mst[v]),
            key=lambda v: min_edge[v],
        )
        in_mst[u] = True
        total += min_edge[u]
        if parent_arr[u] != -1:
            edges.append((points[parent_arr[u]], points[u]))
        for v in range(n):
            if not in_mst[v]:
                d = dist(points[u], points[v])
                if d < min_edge[v]:
                    min_edge[v] = d
                    parent_arr[v] = u

    return total, edges, parent_arr


# =============================================================
#  SECTION 4 — GEOSTEINER INTERFACE
# =============================================================

def run_geosteiner(points, binary_path=GEOSTEINER_PATH):
    """Call the GeoSteiner *esmt* binary and parse its output.

    Returns (smt_length | None, steiner_pts, edge_index_pairs).
    """
    if not os.path.isfile(binary_path):
        print(f"[WARNING] GeoSteiner binary not found: {binary_path}")
        return None, [], []

    n = len(points)
    stdin_str = f"{n}\n" + "\n".join(f"{x} {y}" for x, y in points) + "\n"

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
    except Exception as exc:
        print(f"[ERROR] GeoSteiner failed: {exc}")
        return None, [], []

    out = result.stdout
    if not out.strip():
        print("[WARNING] GeoSteiner produced no output.")
        return None, [], []

    smt_length, steiner_pts, smt_edges = None, [], []
    for line in out.splitlines():
        m = re.search(r"[Ll]ength\s*[=:]\s*([0-9.]+)", line)
        if m:
            smt_length = float(m.group(1))
        m = re.match(r"^\s*s\s+([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)", line)
        if m:
            steiner_pts.append((float(m.group(1)), float(m.group(2))))
        m = re.match(r"^\s*e\s+(\d+)\s+(\d+)", line)
        if m:
            smt_edges.append((int(m.group(1)), int(m.group(2))))

    if smt_length is not None:
        print(
            f"  GeoSteiner: length={smt_length:.4f}, "
            f"{len(steiner_pts)} Steiner pts, {len(smt_edges)} edges"
        )
    else:
        print("[WARNING] Could not parse GeoSteiner output.")
    return smt_length, steiner_pts, smt_edges


# =============================================================
#  SECTION 5 — ROOTED TREE BUILDER
# =============================================================

def build_rooted_tree(all_nodes, pos_edges):
    """BFS from index-0 root, returns *children* dict int -> [int]."""
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


# =============================================================
#  SECTION 6 — COSINE-RULE ROBOT SPAWNING
# =============================================================
# Uses geosteiner_analysis's robust angle method (atan2).
# Stops BEFORE child node.  Source robot is NOT included in output.

def _angle_at_B(A, B, C):
    """Angle at vertex B in path A->B->C  (0..pi)."""
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot_ = BA[0] * BC[0] + BA[1] * BC[1]
    cross_ = BA[0] * BC[1] - BA[1] * BC[0]
    return math.atan2(abs(cross_), dot_)


def _cosine_rule_next_d(d_ri_B, angle, r_desired):
    """Distance from junction B along outgoing edge to the next robot."""
    a, b, c = 1.0, -2.0 * d_ri_B * math.cos(angle), d_ri_B ** 2 - r_desired ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return r_desired
    sq = math.sqrt(disc)
    x1, x2 = (-b + sq) / 2.0, (-b - sq) / 2.0
    cands = sorted(x for x in (x1, x2) if x > 1e-9)
    return cands[0] if cands else r_desired


def spawn_robots_on_tree(nodes, children, r_spacing=60):
    """Place robots on a rooted tree (root=0) using cosine rule.

    Returns
    -------
    all_positions   : list[(x,y)]   — every spawned robot (excl. source)
    edge_robot_map  : dict (pi,ci) -> [(x,y), ...]
    edge_robot_count: dict (pi,ci) -> int
    leaf_residuals  : dict ci -> gap from last robot to child node (leaf only)
    total_r2r       : float — sum of consecutive robot-to-robot distances
    """
    all_positions: list = []
    edge_robot_map: dict = {}
    edge_robot_count: dict = {}
    leaf_residuals: dict = {}
    total_r2r = 0.0

    def _walk(parent_idx, child_idx, last_robot_pos, d_last_to_node):
        nonlocal total_r2r
        p_pos, c_pos = nodes[parent_idx], nodes[child_idx]
        d_edge = dist(p_pos, c_pos)
        if d_edge < 1e-9:
            return [], last_robot_pos, d_last_to_node

        dx = (c_pos[0] - p_pos[0]) / d_edge
        dy = (c_pos[1] - p_pos[1]) / d_edge

        if last_robot_pos is not None and d_last_to_node > 1e-9:
            angle = _angle_at_B(last_robot_pos, p_pos, c_pos)
            first_d = _cosine_rule_next_d(d_last_to_node, angle, r_spacing)
        else:
            first_d = r_spacing

        edge_bots = []
        prev = last_robot_pos
        s = first_d
        while s < d_edge - 1e-9:                      # stop BEFORE child
            rx, ry = p_pos[0] + s * dx, p_pos[1] + s * dy
            if prev is not None:
                total_r2r += dist(prev, (rx, ry))
            prev = (rx, ry)
            edge_bots.append((rx, ry))
            s += r_spacing

        if edge_bots:
            return edge_bots, edge_bots[-1], dist(edge_bots[-1], c_pos)
        else:
            d2c = dist(last_robot_pos, c_pos) if last_robot_pos else d_edge
            return edge_bots, last_robot_pos, d2c

    def _dfs(node_idx, last_robot_pos, d_last_to_node):
        for ci in children.get(node_idx, []):
            bots, new_last, d2child = _walk(
                node_idx, ci, last_robot_pos, d_last_to_node
            )
            edge_robot_map[(node_idx, ci)] = bots
            edge_robot_count[(node_idx, ci)] = len(bots)
            all_positions.extend(bots)
            if not children.get(ci, []):              # leaf
                leaf_residuals[ci] = d2child
            _dfs(ci, new_last, d2child)

    _dfs(0, None, 0)
    return all_positions, edge_robot_map, edge_robot_count, leaf_residuals, total_r2r


# =============================================================
#  SECTION 7 — DOWNSTREAM SINK COUNTING
# =============================================================

def count_downstream_sinks(children, sink_indices):
    """Post-order DFS → dict  node_idx -> # sinks in subtree."""
    ds = {}
    stack = [(0, False)]
    while stack:
        nd, done = stack.pop()
        if done:
            c = 1 if nd in sink_indices else 0
            for ch in children.get(nd, []):
                c += ds.get(ch, 0)
            ds[nd] = c
        else:
            stack.append((nd, True))
            for ch in children.get(nd, []):
                stack.append((ch, False))
    return ds


# =============================================================
#  SECTION 8 — POWER LOSS
# =============================================================

def tree_power_loss(children, nodes, downstream, total_sinks,
                    edge_robot_count, r_per_robot=1.0):
    """Power loss for MST / GeoSteiner tree topology.

    P_edge = (sinks_downstream / total_sinks)^2  x  n_robots  x  r_per_robot
    """
    edge_data, total = [], 0.0

    def _traverse(ni):
        nonlocal total
        for ci in children.get(ni, []):
            pf, pt = nodes[ni], nodes[ci]
            nr = edge_robot_count.get((ni, ci), 0)
            sd = downstream.get(ci, 0)
            I = sd / total_sinks if total_sinks else 0
            P = I ** 2 * nr * r_per_robot
            edge_data.append(dict(
                from_pos=pf, to_pos=pt,
                length=dist(pf, pt), n_robots=nr,
                sinks_down=sd, I_edge=I, P_edge=P,
            ))
            total += P
            _traverse(ci)

    _traverse(0)
    return edge_data, total


def _source_pos(robots_dict):
    """Return source position from the robots dict."""
    for info in robots_dict.values():
        if info["type"] == "SOURCE":
            return info["pos"]
    return None


def _is_at_source(info, src_pos, tol=1e-3):
    """True if robot sits on the source position (should be excluded)."""
    if src_pos is None:
        return False
    return dist(info["pos"], src_pos) < tol


def robot_network_power_loss(robots_dict, n_sinks, r_per_robot=1.0):
    """Power loss for the proposed robot network.

    Source robot and any robot co-located with source are SKIPPED.
    assigned_sinks == [] -> treat as 1 sink.
    """
    if n_sinks == 0:
        return [], 0.0

    src_pos = _source_pos(robots_dict)
    edge_data, total = [], 0.0
    for rid in sorted(robots_dict):
        info = robots_dict[rid]
        if info["type"] == "SOURCE" or _is_at_source(info, src_pos):
            continue
        assigned = info.get("assigned_sinks", [])
        n_a = len(assigned) if assigned else 1          # [] -> 1
        I = n_a / n_sinks
        P = I ** 2 * r_per_robot
        pid = info["parent"]
        pf = robots_dict[pid]["pos"] if pid is not None and pid in robots_dict else info["pos"]
        pt = info["pos"]
        edge_data.append(dict(
            from_pos=pf, to_pos=pt, length=dist(pf, pt),
            n_robots=1, sinks_down=n_a, I_edge=I, P_edge=P, rid=rid,
        ))
        total += P
    return edge_data, total


# =============================================================
#  SECTION 9 — NETWORK LENGTH
# =============================================================

def robot_network_length(robots_dict, sinks_dict, sink_touch_dist):
    """Total robot network length = parent-child edges + last-robot-to-sink."""
    src_pos = _source_pos(robots_dict)
    r2r_total = 0.0
    for rid, info in robots_dict.items():
        if info["type"] == "SOURCE" or _is_at_source(info, src_pos):
            continue
        pid = info["parent"]
        if pid is not None and pid in robots_dict:
            r2r_total += dist(info["pos"], robots_dict[pid]["pos"])

    r2s_total = 0.0
    for sid, sink in sinks_dict.items():
        sp = sink["pos"]
        best_d = float("inf")
        for rid, info in robots_dict.items():
            if info["type"] == "SOURCE" or _is_at_source(info, src_pos):
                continue
            d = dist(info["pos"], sp)
            if d <= sink_touch_dist and d < best_d:
                best_d = d
        if best_d < float("inf"):
            r2s_total += best_d

    return r2r_total + r2s_total, r2r_total, r2s_total


def tree_network_length(total_r2r, leaf_residuals):
    """MST / GeoSteiner robot network length.

    = robot-to-robot total + leaf residual gaps
    """
    leaf_gap = sum(leaf_residuals.values())
    return total_r2r + leaf_gap, total_r2r, leaf_gap


# =============================================================
#  SECTION 10 — FULL ANALYSIS PIPELINE
# =============================================================

def _summary(method, edge_data, total_loss, n_sinks, total_robots,
             net_length, r2r_len, residual_len):
    return dict(
        method=method, total_loss=total_loss,
        total_robots=total_robots, n_sinks=n_sinks,
        net_length=net_length, r2r_length=r2r_len,
        residual_length=residual_len, edge_data=edge_data,
    )


def run_analysis(data):
    """Run the full MST / GeoSteiner / Robot-Network comparison.

    Parameters
    ----------
    data : dict produced by parse_simulation_output()

    Returns
    -------
    results : dict with keys  mst, geo, rn  (each a summary dict),
              plus raw intermediate data.
    """
    src = data["source"]
    sinks = data["sinks"]
    robots = data["robots"]
    cfg = data["config"]
    r_spacing = int(cfg.get("R", 60))
    sink_touch_dist = float(cfg.get("SINK_TOUCH_DIST", r_spacing))
    n_sinks = len(sinks)
    sink_positions = [sinks[sid]["pos"] for sid in sorted(sinks)]
    terminals = [src] + sink_positions

    # ── MST ───────────────────────────────────────────────────
    mst_euc_len, mst_edges, mst_parent = compute_mst(terminals)
    mst_children = defaultdict(list)
    for ci, pi in enumerate(mst_parent):
        if pi != -1:
            mst_children[pi].append(ci)
    mst_children = dict(mst_children)

    mst_sink_idx = set(range(1, n_sinks + 1))
    mst_ds = count_downstream_sinks(mst_children, mst_sink_idx)
    mst_pos, mst_emap, mst_ecnt, mst_leaf_res, mst_r2r = spawn_robots_on_tree(
        terminals, mst_children, r_spacing
    )
    mst_ed, mst_loss = tree_power_loss(
        mst_children, terminals, mst_ds, n_sinks, mst_ecnt
    )
    mst_net_len, mst_r2r_len, mst_res_len = tree_network_length(mst_r2r, mst_leaf_res)
    mst_n_robots = len(mst_pos)  # source NOT counted
    mst_sum = _summary(
        "MST", mst_ed, mst_loss, n_sinks,
        mst_n_robots, mst_net_len, mst_r2r_len, mst_res_len,
    )
    mst_sum["euc_length"] = mst_euc_len
    mst_sum["edges"] = mst_edges
    mst_sum["robot_positions"] = mst_pos
    mst_sum["children"] = mst_children

    # ── GeoSteiner ────────────────────────────────────────────
    geo_euc_len, steiner_pts, geo_edge_idx = run_geosteiner(terminals)
    geo_sum = None
    if geo_euc_len is not None:
        all_geo_nodes = terminals + steiner_pts
        geo_pos_edges = []
        for i, j in geo_edge_idx:
            if i < len(all_geo_nodes) and j < len(all_geo_nodes):
                geo_pos_edges.append((all_geo_nodes[i], all_geo_nodes[j]))
        geo_children = build_rooted_tree(all_geo_nodes, geo_pos_edges)
        geo_ds = count_downstream_sinks(geo_children, mst_sink_idx)
        geo_pos, geo_emap, geo_ecnt, geo_leaf_res, geo_r2r = spawn_robots_on_tree(
            all_geo_nodes, geo_children, r_spacing
        )
        geo_ed, geo_loss = tree_power_loss(
            geo_children, all_geo_nodes, geo_ds, n_sinks, geo_ecnt
        )
        geo_net_len, geo_r2r_len, geo_res_len = tree_network_length(geo_r2r, geo_leaf_res)
        geo_n_robots = len(geo_pos)
        geo_sum = _summary(
            "GeoSteiner", geo_ed, geo_loss, n_sinks,
            geo_n_robots, geo_net_len, geo_r2r_len, geo_res_len,
        )
        geo_sum["euc_length"] = geo_euc_len
        geo_sum["steiner_pts"] = steiner_pts
        geo_sum["pos_edges"] = geo_pos_edges
        geo_sum["robot_positions"] = geo_pos
        geo_sum["children"] = geo_children
        geo_sum["all_nodes"] = all_geo_nodes

    # ── Robot Network (proposed) ──────────────────────────────
    rn_ed, rn_loss = robot_network_power_loss(robots, n_sinks)
    rn_net_len, rn_r2r_len, rn_r2s_len = robot_network_length(
        robots, sinks, sink_touch_dist
    )
    _src_pos = _source_pos(robots)
    rn_n_robots = sum(
        1 for r in robots.values()
        if r["type"] != "SOURCE" and not _is_at_source(r, _src_pos)
    )
    rn_sum = _summary(
        "Robot Network", rn_ed, rn_loss, n_sinks,
        rn_n_robots, rn_net_len, rn_r2r_len, rn_r2s_len,
    )

    return dict(
        mst=mst_sum, geo=geo_sum, rn=rn_sum,
        terminals=terminals, data=data,
    )


# =============================================================
#  SECTION 11 — TEXT REPORT
# =============================================================

def print_report(results, out_file=None):
    """Print (and optionally save) the analysis comparison report."""
    mst = results["mst"]
    geo = results["geo"]
    rn = results["rn"]
    data = results["data"]
    src = data["source"]
    sinks = data["sinks"]
    n_sinks = len(sinks)
    cfg = data["config"]
    r_spacing = int(cfg.get("R", 60))

    lines = []
    lines.append("=" * 70)
    lines.append("     UNIFIED POST-PROCESSING ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Random Seed: {data.get('random_seed', 'N/A')}")
    lines.append(f"R (spacing): {r_spacing}")
    lines.append(f"Source: {src}")
    lines.append(f"Sinks: {n_sinks}")
    for sid in sorted(sinks):
        s = sinks[sid]
        lines.append(f"  S{sid}  {s['pos']}  ({s['status']})")
    lines.append("")

    # ── Network Length Comparison ─────────────────────────────
    lines.append("-" * 70)
    lines.append("NETWORK LENGTH COMPARISON")
    lines.append("-" * 70)
    hdr = f"  {'Method':<20} {'Euclidean':<14} {'Robot R2R':<14} {'Leaf/Sink Gap':<14} {'Total':<14}"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    mst_euc = f"{mst['euc_length']:.2f}"
    lines.append(
        f"  {'MST':<20} {mst_euc:<14} {mst['r2r_length']:<14.2f} "
        f"{mst['residual_length']:<14.2f} {mst['net_length']:<14.2f}"
    )
    if geo:
        geo_euc = f"{geo['euc_length']:.2f}"
        lines.append(
            f"  {'GeoSteiner':<20} {geo_euc:<14} {geo['r2r_length']:<14.2f} "
            f"{geo['residual_length']:<14.2f} {geo['net_length']:<14.2f}"
        )
    lines.append(
        f"  {'Robot Network':<20} {'N/A':<14} {rn['r2r_length']:<14.2f} "
        f"{rn['residual_length']:<14.2f} {rn['net_length']:<14.2f}"
    )
    lines.append("")

    # ── Robot Count Comparison ───────────────────────────────
    lines.append("-" * 70)
    lines.append("ROBOT COUNT COMPARISON")
    lines.append("-" * 70)
    lines.append(f"  {'MST (cosine-rule)':<30} {mst['total_robots']}")
    if geo:
        lines.append(f"  {'GeoSteiner (cosine-rule)':<30} {geo['total_robots']}")
    lines.append(f"  {'Robot Network (simulation)':<30} {rn['total_robots']}")
    lines.append("")

    # ── Power Loss Comparison ────────────────────────────────
    lines.append("-" * 70)
    lines.append("POWER LOSS COMPARISON")
    lines.append("-" * 70)
    lines.append(f"  Power model: P_edge = (k_e / N)^2 x n_robots x r")
    lines.append(f"  where k_e = downstream sinks, N = {n_sinks}, r = 1.0/robot")
    lines.append("")
    hdr2 = f"  {'Method':<20} {'Total Loss':>12} {'Robots':>8} {'Net Length':>12}"
    lines.append(hdr2)
    lines.append("  " + "-" * (len(hdr2) - 2))
    for s in [mst, geo, rn]:
        if s is None:
            continue
        lines.append(
            f"  {s['method']:<20} {s['total_loss']:>12.6f} "
            f"{s['total_robots']:>8d} {s['net_length']:>12.2f}"
        )
    lines.append("")

    # ── Ratios ────────────────────────────────────────────────
    lines.append("-" * 70)
    lines.append("RATIOS")
    lines.append("-" * 70)
    if geo and geo["total_loss"] > 0:
        for s in [mst, geo, rn]:
            if s is None:
                continue
            rl = s["total_loss"] / geo["total_loss"]
            rr = s["total_robots"] / geo["total_robots"] if geo["total_robots"] else 0
            rx = s["net_length"] / geo["net_length"] if geo["net_length"] else 0
            lines.append(
                f"  {s['method']:<20} Loss x{rl:.4f}  "
                f"Robots x{rr:.4f}  Length x{rx:.4f}"
            )
    elif mst["net_length"] > 0:
        for s in [mst, rn]:
            if s is None:
                continue
            rx = s["net_length"] / mst["net_length"] if mst["net_length"] else 0
            lines.append(f"  {s['method']:<20} Length x{rx:.4f}")
    lines.append("")

    # ── Per-edge detail (MST) ─────────────────────────────────
    lines.append("-" * 70)
    lines.append("MST EDGE DETAILS (cosine-rule)")
    lines.append("-" * 70)
    lines.append(
        f"  {'From':<22} {'To':<22} {'Len':>8} {'Bots':>6} "
        f"{'k_down':>6} {'I':>8} {'P':>10}"
    )
    lines.append("  " + "-" * 82)
    for e in mst["edge_data"]:
        pf = f"({e['from_pos'][0]:.1f},{e['from_pos'][1]:.1f})"
        pt = f"({e['to_pos'][0]:.1f},{e['to_pos'][1]:.1f})"
        lines.append(
            f"  {pf:<22} {pt:<22} {e['length']:>8.2f} {e['n_robots']:>6} "
            f"{e['sinks_down']:>6} {e['I_edge']:>8.4f} {e['P_edge']:>10.6f}"
        )
    lines.append("")

    # ── Per-edge detail (GeoSteiner) if available ─────────────
    if geo:
        lines.append("-" * 70)
        lines.append("GEOSTEINER EDGE DETAILS (cosine-rule)")
        lines.append("-" * 70)
        lines.append(
            f"  {'From':<22} {'To':<22} {'Len':>8} {'Bots':>6} "
            f"{'k_down':>6} {'I':>8} {'P':>10}"
        )
        lines.append("  " + "-" * 82)
        for e in geo["edge_data"]:
            pf = f"({e['from_pos'][0]:.1f},{e['from_pos'][1]:.1f})"
            pt = f"({e['to_pos'][0]:.1f},{e['to_pos'][1]:.1f})"
            lines.append(
                f"  {pf:<22} {pt:<22} {e['length']:>8.2f} {e['n_robots']:>6} "
                f"{e['sinks_down']:>6} {e['I_edge']:>8.4f} {e['P_edge']:>10.6f}"
            )
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF ANALYSIS REPORT")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    if out_file:
        with open(out_file, "w") as f:
            f.write(report)
        print(f"\nAnalysis report saved to: {out_file}")


# =============================================================
#  SECTION 12 — VISUALIZATION
# =============================================================

def _style_ax(ax, dark=False):
    if dark:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("gray")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.15 if dark else 0.3, color="gray" if dark else "lightgray")


def _axis_limits(all_positions):
    """Compute padded axis limits for a list of (x,y) tuples."""
    xs = [p[0] for p in all_positions]
    ys = [p[1] for p in all_positions]
    xmin, xmax = min(xs) - 200, max(xs) + 200
    ymin, ymax = min(ys) - 200, max(ys) + 200
    w = max(xmax - xmin, 800)
    h = max(ymax - ymin, 800)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2


def plot_three_methods(results, out_dir=""):
    """3-panel plot: Robot Network | MST | GeoSteiner."""
    mst = results["mst"]
    geo = results["geo"]
    rn = results["rn"]
    data = results["data"]
    src = data["source"]
    sinks = data["sinks"]
    robots = data["robots"]
    terminals = results["terminals"]

    n_panels = 3 if geo else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 8))
    fig.patch.set_facecolor("#0f0f1a")
    if n_panels == 1:
        axes = [axes]

    all_pts = list(terminals)
    for r in robots.values():
        all_pts.append(r["pos"])
    xlo, xhi, ylo, yhi = _axis_limits(all_pts)

    titles = ["Robot Network (Proposed)", "MST (Prim's)", "GeoSteiner SMT"]

    # Panel 0 — Robot Network
    ax0 = axes[0]
    ax0.set_title(titles[0], color="white", fontsize=13, fontweight="bold")
    _style_ax(ax0, dark=True)
    ax0.set_xlim(xlo, xhi); ax0.set_ylim(yhi, ylo)  # inverted y
    for rid, info in robots.items():
        pid = info["parent"]
        if pid is not None and pid in robots:
            ax0.plot(
                [info["pos"][0], robots[pid]["pos"][0]],
                [info["pos"][1], robots[pid]["pos"][1]],
                color="#3a7bd5" if info["type"] == "STRAIGHT" else "#f9ca24",
                linewidth=0.8, alpha=0.6,
            )
    # Draw leaf-to-sink dashed
    parent_rids = {r["parent"] for r in robots.values() if r["parent"] is not None}
    for sid in sinks:
        sp = sinks[sid]["pos"]
        best_d, best_r = float("inf"), None
        for rid, info in robots.items():
            d = dist(info["pos"], sp)
            if d < best_d:
                best_d, best_r = d, rid
        if best_r is not None:
            rp = robots[best_r]["pos"]
            ax0.plot([rp[0], sp[0]], [rp[1], sp[1]],
                     color="#ff6b6b", lw=1.2, ls="--", alpha=0.8)
    # Draw robot dots
    for rid, info in robots.items():
        if info["type"] == "SOURCE":
            continue
        c = "#f9ca24" if info["type"] == "PIVOT" else "#3a7bd5"
        sz = 40 if info["type"] == "PIVOT" else 10
        mk = "D" if info["type"] == "PIVOT" else "o"
        ax0.scatter(*info["pos"], color=c, s=sz, zorder=4, marker=mk,
                    edgecolors="white", linewidths=0.3)
    # Sinks + Source
    for sid in sinks:
        ax0.scatter(*sinks[sid]["pos"], color="#ff4757", s=120, zorder=6,
                    marker="*", edgecolors="white", linewidths=0.5)
        ax0.annotate(f"S{sid}", sinks[sid]["pos"], textcoords="offset points",
                     xytext=(6, 4), color="#ff4757", fontsize=9, fontweight="bold")
    ax0.scatter(*src, color="#2ed573", s=200, zorder=7, marker="^",
                edgecolors="white", linewidths=1)
    ax0.annotate("SOURCE", src, textcoords="offset points", xytext=(8, 4),
                 color="#2ed573", fontsize=9, fontweight="bold")
    ax0.text(
        0.02, 0.98,
        f"Robots: {rn['total_robots']}\nLength: {rn['net_length']:.2f}\n"
        f"Power: {rn['total_loss']:.6f}",
        transform=ax0.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", fc="#1a1a2e", ec="gray", alpha=0.8),
        color="white",
    )

    # Panel 1 — MST
    ax1 = axes[1]
    ax1.set_title(titles[1], color="white", fontsize=13, fontweight="bold")
    _style_ax(ax1, dark=True)
    ax1.set_xlim(xlo, xhi); ax1.set_ylim(yhi, ylo)
    for p1, p2 in mst["edges"]:
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color="#a29bfe", linewidth=2, ls="--", alpha=0.8)
    if mst["robot_positions"]:
        rxs, rys = zip(*mst["robot_positions"])
        ax1.scatter(rxs, rys, color="#a29bfe", s=12, zorder=4, alpha=0.8,
                    edgecolors="white", linewidths=0.3)
    sink_ids = sorted(sinks)
    for i, pos in enumerate(terminals):
        if i == 0:
            ax1.scatter(*pos, color="#2ed573", s=200, zorder=7, marker="^",
                        edgecolors="white", linewidths=1)
            ax1.annotate("SOURCE", pos, textcoords="offset points", xytext=(8, 4),
                         color="#2ed573", fontsize=9, fontweight="bold")
        else:
            ax1.scatter(*pos, color="#ff4757", s=120, zorder=6, marker="*",
                        edgecolors="white", linewidths=0.5)
            sid = sink_ids[i-1]
            ax1.annotate(f"S{sid}", pos, textcoords="offset points", xytext=(6, 4),
                         color="#ff4757", fontsize=9, fontweight="bold")
    ax1.text(
        0.02, 0.98,
        f"Euc: {mst['euc_length']:.2f}\nRobots: {mst['total_robots']}\n"
        f"Length: {mst['net_length']:.2f}\nPower: {mst['total_loss']:.6f}",
        transform=ax1.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", fc="#1a1a2e", ec="gray", alpha=0.8),
        color="white",
    )

    # Panel 2 — GeoSteiner
    if geo:
        ax2 = axes[2]
        ax2.set_title(titles[2], color="white", fontsize=13, fontweight="bold")
        _style_ax(ax2, dark=True)
        ax2.set_xlim(xlo, xhi); ax2.set_ylim(yhi, ylo)
        for p1, p2 in geo["pos_edges"]:
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                     color="#fd79a8", linewidth=2, alpha=0.9)
        if geo["robot_positions"]:
            rxs, rys = zip(*geo["robot_positions"])
            ax2.scatter(rxs, rys, color="#fd79a8", s=12, zorder=4, alpha=0.8,
                        edgecolors="white", linewidths=0.3)
        if geo["steiner_pts"]:
            sx, sy = zip(*geo["steiner_pts"])
            ax2.scatter(sx, sy, color="#f9ca24", s=80, zorder=5, marker="D",
                        edgecolors="white", linewidths=0.5)
            for si, sp in enumerate(geo["steiner_pts"]):
                ax2.annotate(f"ST{si}", sp, textcoords="offset points",
                             xytext=(6, 4), color="#f9ca24", fontsize=7)
        sink_ids = sorted(sinks)
        for i, pos in enumerate(terminals):
            if i == 0:
                ax2.scatter(*pos, color="#2ed573", s=200, zorder=7, marker="^",
                            edgecolors="white", linewidths=1)
                ax2.annotate("SOURCE", pos, textcoords="offset points",
                             xytext=(8, 4), color="#2ed573", fontsize=9, fontweight="bold")
            else:
                ax2.scatter(*pos, color="#ff4757", s=120, zorder=6, marker="*",
                            edgecolors="white", linewidths=0.5)
                sid = sink_ids[i-1]
                ax2.annotate(f"S{sid}", pos, textcoords="offset points",
                             xytext=(6, 4), color="#ff4757", fontsize=9, fontweight="bold")
        ax2.text(
            0.02, 0.98,
            f"Euc: {geo['euc_length']:.2f}\nRobots: {geo['total_robots']}\n"
            f"Length: {geo['net_length']:.2f}\nPower: {geo['total_loss']:.6f}",
            transform=ax2.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", fc="#1a1a2e", ec="gray", alpha=0.8),
            color="white",
        )

    plt.tight_layout()
    fname = os.path.join(out_dir, "three_methods_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)


def plot_power_loss_bars(results, out_dir=""):
    """Bar chart comparing power loss and robot counts."""
    summaries = [results["mst"]]
    if results["geo"]:
        summaries.append(results["geo"])
    summaries.append(results["rn"])

    cmap = {"MST": "#a29bfe", "GeoSteiner": "#fd79a8", "Robot Network": "#3a7bd5"}
    methods = [s["method"] for s in summaries]
    losses = [s["total_loss"] for s in summaries]
    robots = [s["total_robots"] for s in summaries]
    colors = [cmap.get(m, "#ccc") for m in methods]
    x = np.arange(len(methods))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in (a1, a2):
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("white")
        ax.title.set_color("white")
        ax.yaxis.label.set_color("white")

    a1.bar(x, losses, color=colors, edgecolor="white", linewidth=0.5)
    a1.set_xticks(x); a1.set_xticklabels(methods, color="white")
    a1.set_ylabel("Power Loss"); a1.set_title("Total Power Loss per Method")
    for i, v in enumerate(losses):
        a1.text(i, v + max(losses) * 0.02, f"{v:.6f}", ha="center",
                color="white", fontsize=9)

    a2.bar(x, robots, color=colors, edgecolor="white", linewidth=0.5)
    a2.set_xticks(x); a2.set_xticklabels(methods, color="white")
    a2.set_ylabel("Robot Count"); a2.set_title("Total Robots per Method")
    for i, v in enumerate(robots):
        a2.text(i, v + max(robots) * 0.02, str(v), ha="center",
                color="white", fontsize=9)

    plt.tight_layout()
    fname = os.path.join(out_dir, "power_loss_comparison.png")
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)


def _aggregate_rn_edges(robots_dict, n_sinks):
    """Collapse per-robot edges into branch segments for the heatmap.

    Merges straight chains between branch-points (PIVOT/SOURCE) into
    one visual edge with total robot count and shared I.
    """
    if n_sinks == 0:
        return []
    # Build children map, identify source and co-located robots
    source_rid = None
    src_pos = _source_pos(robots_dict)
    skip_rids = set()
    children_map = defaultdict(list)
    for rid, info in robots_dict.items():
        if info["type"] == "SOURCE":
            source_rid = rid
            skip_rids.add(rid)
        elif _is_at_source(info, src_pos):
            skip_rids.add(rid)
        pid = info.get("parent")
        if pid is not None and pid in robots_dict:
            children_map[pid].append(rid)
    if source_rid is None:
        return []

    # Re-parent: if a skipped robot's children exist, attach them to its parent
    for skip_rid in list(skip_rids):
        if skip_rid == source_rid:
            continue
        parent_rid = robots_dict[skip_rid].get("parent")
        for child_rid in children_map.get(skip_rid, []):
            if parent_rid is not None:
                children_map[parent_rid].append(child_rid)
        children_map.pop(skip_rid, None)
        # Remove from parent's children list
        if parent_rid is not None and parent_rid in children_map:
            children_map[parent_rid] = [
                c for c in children_map[parent_rid] if c != skip_rid
            ]

    aggregated = []

    def _walk(start_rid):
        for kid in sorted(children_map.get(start_rid, [])):
            chain = [kid]
            cur = kid
            while len(children_map.get(cur, [])) == 1:
                cur = children_map[cur][0]
                chain.append(cur)
            # segment from start_rid to chain[-1]
            assigned = robots_dict[chain[0]].get("assigned_sinks", [])
            k_e = len(assigned) if assigned else 1
            I = k_e / n_sinks
            n_robots = len(chain)
            P = I ** 2 * n_robots
            aggregated.append(dict(
                from_pos=robots_dict[start_rid]["pos"],
                to_pos=robots_dict[chain[-1]]["pos"],
                length=dist(robots_dict[start_rid]["pos"],
                            robots_dict[chain[-1]]["pos"]),
                n_robots=n_robots, sinks_down=k_e,
                I_edge=I, P_edge=P,
            ))
            if children_map.get(chain[-1]):
                _walk(chain[-1])

    _walk(source_rid)
    return aggregated


def plot_edge_heatmap(results, out_dir=""):
    """Per-method edge power heatmap."""
    summaries = [results["mst"]]
    labels = ["MST"]
    if results["geo"]:
        summaries.append(results["geo"])
        labels.append("GeoSteiner")
    summaries.append(results["rn"])
    labels.append("Robot Network")

    data = results["data"]
    src = data["source"]
    sinks = data["sinks"]

    nm = len(summaries)
    fig, axes = plt.subplots(1, nm, figsize=(7 * nm, 6))
    fig.patch.set_facecolor("#0f0f1a")
    if nm == 1:
        axes = [axes]
    cm = plt.cm.plasma

    for idx, (s, label) in enumerate(zip(summaries, labels)):
        ax = axes[idx]
        ax.set_facecolor("#0f0f1a")
        ax.set_title(label, color="white", fontsize=13)
        ax.tick_params(colors="white")
        ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_color("white")

        # For Robot Network: aggregate per-robot edges into branch segments
        if label == "Robot Network":
            edges = _aggregate_rn_edges(data["robots"], len(data["sinks"]))
        else:
            edges = s["edge_data"]

        if not edges:
            ax.text(0.5, 0.5, "No edges", transform=ax.transAxes,
                    ha="center", color="white")
            continue
        pv = [e["P_edge"] for e in edges]
        pmin, pmax = min(pv), max(pv)
        if pmax == pmin:
            pmax += 1e-9
        norm = plt.Normalize(pmin, pmax)
        for e in edges:
            xs = [e["from_pos"][0], e["to_pos"][0]]
            ys = [e["from_pos"][1], e["to_pos"][1]]
            frac = (e["P_edge"] - pmin) / (pmax - pmin)
            lw = 1 + frac * 5
            ax.plot(xs, ys, color=cm(norm(e["P_edge"])), linewidth=lw,
                    solid_capstyle="round")
            mx, my = (xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2
            ax.annotate(
                f"n={e['n_robots']} I={e['I_edge']:.2f}", (mx, my),
                color="white", fontsize=7, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="#0f0f1a",
                          ec="none", alpha=0.7),
            )
        ax.plot(*src, marker="^", color="limegreen", markersize=12, zorder=5,
                markeredgecolor="white", linewidth=0)
        for sid in sinks:
            ax.plot(*sinks[sid]["pos"], marker="*", color="red", markersize=12,
                    zorder=5, markeredgecolor="white", linewidth=0)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("P_edge", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

    plt.tight_layout()
    fname = os.path.join(out_dir, "edge_power_heatmap.png")
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor())
    print(f"  Saved: {fname}")
    plt.close(fig)


# =============================================================
#  SECTION 13 — JSON EXPORT
# =============================================================

def export_json(results, out_file):
    """Export key results to a JSON file."""
    def _ser(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    out = {}
    for key in ("mst", "geo", "rn"):
        s = results[key]
        if s is None:
            out[key] = None
            continue
        out[key] = {
            "method": s["method"],
            "total_loss": s["total_loss"],
            "total_robots": s["total_robots"],
            "net_length": s["net_length"],
            "r2r_length": s["r2r_length"],
            "residual_length": s["residual_length"],
            "euc_length": s.get("euc_length"),
        }
    out["source"] = results["data"]["source"]
    out["sinks"] = {
        str(k): v for k, v in results["data"]["sinks"].items()
    }
    out["config"] = results["data"]["config"]
    out["random_seed"] = results["data"].get("random_seed")

    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, default=_ser)
    print(f"  JSON exported to: {out_file}")


# =============================================================
#  MAIN
# =============================================================

def discover_simdata_files(root_dir="."):
    """Find all *_simdata.txt files inside immediate subdirectories."""
    found = []
    for entry in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, entry)
        if os.path.isdir(full):
            for f in os.listdir(full):
                if f.endswith("_simdata.txt"):
                    found.append(os.path.join(full, f))
    return found


def process_one(report_path):
    """Run the full analysis pipeline on a single simdata.txt file.

    Creates an _analysis subfolder next to the input with:
      - analysis report (.txt)
      - analysis data (.json)
      - three_methods_comparison.png
      - power_loss_comparison.png
      - edge_power_heatmap.png
    """
    print(f"\n{'=' * 70}")
    print(f"  Processing: {report_path}")
    print(f"{'=' * 70}")

    data = parse_simulation_output(report_path)

    src = data["source"]
    sinks = data["sinks"]
    print(f"  Source: {src}")
    print(f"  Sinks:  {len(sinks)}")
    for sid in sorted(sinks):
        print(f"    S{sid}  {sinks[sid]['pos']}  ({sinks[sid]['status']})")
    print(f"  Robots: {len(data['robots'])}")
    print(f"  R:      {data['config'].get('R', '?')}")

    # ── Run analysis ──────────────────────────────────────────
    print("\n  Running analysis ...")
    results = run_analysis(data)

    # ── Output directory — save into the SAME folder as the simdata file ──
    in_dir = os.path.dirname(os.path.abspath(report_path))
    base = os.path.splitext(os.path.basename(report_path))[0]

    # Put analysis outputs directly alongside the simdata file
    out_dir = in_dir

    # ── Print & save report ───────────────────────────────────
    report_file = os.path.join(out_dir, f"{base}_analysis_report.txt")
    print_report(results, out_file=report_file)

    # ── Export JSON ───────────────────────────────────────────
    json_file = os.path.join(out_dir, f"{base}_analysis_data.json")
    export_json(results, json_file)

    # ── Plots ─────────────────────────────────────────────────
    print("\n  Generating plots ...")
    plot_three_methods(results, out_dir=out_dir)
    plot_power_loss_bars(results, out_dir=out_dir)
    plot_edge_heatmap(results, out_dir=out_dir)

    print(f"\n  Done: {os.path.basename(in_dir)}")
    return results


def main():
    if len(sys.argv) >= 2:
        # ── Explicit path(s) provided ─────────────────────────
        paths = sys.argv[1:]
    else:
        # ── Auto-discover all _simdata.txt in local folders ───
        paths = discover_simdata_files(".")
        if not paths:
            print("No *_simdata.txt files found in subdirectories.")
            print("Usage:")
            print("  python unified_post_processing.py                       "
                  "# auto-discover")
            print("  python unified_post_processing.py <simdata.txt> [...]   "
                  "# explicit files")
            sys.exit(1)
        print(f"\n  Auto-discovered {len(paths)} simulation output(s):")
        for p in paths:
            print(f"    {p}")

    for path in paths:
        if not os.path.isfile(path):
            print(f"\n  [SKIP] File not found: {path}")
            continue
        process_one(path)

    print(f"\n{'=' * 70}")
    print(f"  All done — processed {len(paths)} file(s).")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
