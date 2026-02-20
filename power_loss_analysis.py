"""
power_loss_analysis.py
======================
Power-loss estimation for robot-swarm network topologies (IROS 2026).

Simply reads the simulation report via geosteiner_analysis.parse_report(),
uses compute_mst_length() and run_geosteiner() from the same module,
then computes power loss for all three methods: MST, GeoSteiner, Robot Network.

Power model per edge:
    P_edge = I_edge² × n_robots × r_robot

where
    I_edge    = sinks_downstream / total_sinks
    n_robots  = ceil(edge_length / R)
    r_robot   = 1.0
"""

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict

# Import everything we need from the existing analysis module
from geosteiner_analysis import (
    parse_report,
    compute_mst_length,
    run_geosteiner,
    dist,
    REPORT_FILE,
    spawn_robots_on_tree,
)

# ── Spacing config ────────────────────────────────────────────
R_SPACING = 60


# ===========================================================
#  Core helpers
# ===========================================================

def robots_on_edge(p1, p2, r_spacing=R_SPACING):
    """ceil(edge_length / r_spacing), minimum 1."""
    d = dist(p1, p2)
    return max(1, math.ceil(d / r_spacing)) if d > 0 else 1


def build_tree(source, all_nodes, edges):
    """BFS from all_nodes[0] to produce a rooted tree.

    Parameters
    ----------
    source : tuple – (x,y) of root (must match all_nodes[0]).
    all_nodes : list of (x,y) tuples.
    edges : list of ((x1,y1),(x2,y2)) undirected edges.

    Returns
    -------
    children : dict  node_idx → [child indices]
    parent   : dict  node_idx → parent idx (root→None)
    nodes    : same as all_nodes
    """
    TOL = 1e-3

    def _idx(pos):
        for i, nd in enumerate(all_nodes):
            if abs(nd[0] - pos[0]) < TOL and abs(nd[1] - pos[1]) < TOL:
                return i
        return None

    adj = defaultdict(set)
    for p1, p2 in edges:
        i, j = _idx(p1), _idx(p2)
        if i is None or j is None:
            continue
        adj[i].add(j)
        adj[j].add(i)

    children = defaultdict(list)
    parent = {0: None}
    visited = {0}
    q = deque([0])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[u].append(v)
                q.append(v)

    return dict(children), parent, all_nodes


def count_downstream_sinks(children, parent, nodes, sink_indices):
    """Post-order DFS → dict node_idx → # sinks in subtree."""
    ds = {}
    stack = [(0, False)]
    while stack:
        nd, done = stack.pop()
        if done:
            c = 1 if nd in sink_indices else 0
            for ch in children.get(nd, []):
                c += ds[ch]
            ds[nd] = c
        else:
            stack.append((nd, True))
            for ch in children.get(nd, []):
                stack.append((ch, False))
    return ds


def compute_power_loss(children, parent, nodes, downstream, total_sinks,
                       edge_robot_count, r_per_robot=1.0):
    """Return (edge_data_list, total_loss) for every tree edge.

    edge_robot_count : dict (parent_idx, child_idx) → int
        Actual robot count per edge from spawn_robots_on_tree.
    """
    edges, total = [], 0.0
    for ni, pi in parent.items():
        if pi is None:
            continue
        pf, pt = nodes[pi], nodes[ni]
        length = dist(pf, pt)
        nr = edge_robot_count.get((pi, ni), 0)
        sd = downstream.get(ni, 0)
        I = sd / total_sinks if total_sinks else 0
        P = I ** 2 * nr * r_per_robot
        edges.append(dict(from_pos=pf, to_pos=pt, length=length,
                          n_robots=nr, r_edge=nr * r_per_robot,
                          sinks_down=sd, I_edge=I, P_edge=P))
        total += P
    return edges, total


def _summary(method, edge_data, total_loss, n_sinks):
    return dict(method=method, total_loss=total_loss,
                total_robots=sum(e["n_robots"] for e in edge_data),
                total_length=sum(e["length"] for e in edge_data),
                n_sinks=n_sinks, edge_data=edge_data)


# ===========================================================
#  Per-method wrappers
# ===========================================================

def mst_power_loss(source, sink_positions, mst_edges,
                   r_spacing=R_SPACING, r_per_robot=1.0):
    """Power loss on MST topology (cosine-rule robot spawning)."""
    nodes = [source] + list(sink_positions)
    sinks_idx = set(range(1, len(sink_positions) + 1))
    ch, par, nd = build_tree(source, nodes, mst_edges)
    ds = count_downstream_sinks(ch, par, nd, sinks_idx)
    # Spawn robots with cosine rule at corners
    all_pos, edge_map, edge_cnt = spawn_robots_on_tree(nd, ch, r_spacing)
    ed, loss = compute_power_loss(ch, par, nd, ds, len(sink_positions),
                                  edge_cnt, r_per_robot)
    summary = _summary("MST", ed, loss, len(sink_positions))
    summary["robot_positions"] = all_pos
    summary["edge_robot_map"]  = edge_map
    return ed, loss, summary


def geosteiner_power_loss(source, sink_positions, steiner_pts, edges_idx,
                          r_spacing=R_SPACING, r_per_robot=1.0):
    """Power loss on GeoSteiner SMT topology (cosine-rule robot spawning).

    edges_idx indexes into [source]+sink_positions+steiner_pts.
    Steiner points carry current but do NOT consume it.
    """
    nodes = [source] + list(sink_positions) + list(steiner_pts)
    n_sinks = len(sink_positions)
    sinks_idx = set(range(1, n_sinks + 1))

    pos_edges = []
    for i, j in edges_idx:
        if 0 <= i < len(nodes) and 0 <= j < len(nodes):
            pos_edges.append((nodes[i], nodes[j]))

    ch, par, nd = build_tree(source, nodes, pos_edges)
    ds = count_downstream_sinks(ch, par, nd, sinks_idx)
    # Spawn robots with cosine rule at Steiner pivots
    all_pos, edge_map, edge_cnt = spawn_robots_on_tree(nd, ch, r_spacing)
    ed, loss = compute_power_loss(ch, par, nd, ds, n_sinks,
                                  edge_cnt, r_per_robot)
    summary = _summary("GeoSteiner", ed, loss, n_sinks)
    summary["robot_positions"] = all_pos
    summary["edge_robot_map"]  = edge_map
    return ed, loss, summary


def robot_network_power_loss(robots_dict, sinks_dict, r_per_robot=1.0):
    """Power loss on the actual robot network from the simulation.

    Simple model:
      - Robot 0 (SOURCE) is skipped — it does not consume power.
      - Each other robot has an 'assigned_sinks' list from the report.
      - I_robot = len(assigned_sinks) / total_sinks
        (special case: if assigned_sinks is [], treat as 1 sink)
      - P_robot = I_robot² × r_per_robot
      - Total loss = sum of P_robot for all robots except source.
    """
    n_sinks = len(sinks_dict)
    if n_sinks == 0:
        return [], 0.0, _summary("Robot Network", [], 0.0, 0)

    edge_data, total = [], 0.0
    for rid, info in sorted(robots_dict.items()):
        # Skip source — it doesn't consume power
        if info["type"] == "SOURCE":
            continue

        assigned = info.get("assigned_sinks", [])
        n_assigned = len(assigned) if len(assigned) > 0 else 1  # [] → 1

        I = n_assigned / n_sinks
        P = I ** 2 * r_per_robot

        # Build edge info (parent → this robot)
        parent_id = info["parent"]
        pf = robots_dict[parent_id]["pos"] if parent_id is not None else info["pos"]
        pt = info["pos"]
        length = dist(pf, pt)

        edge_data.append(dict(from_pos=pf, to_pos=pt, length=length,
                              n_robots=1, r_edge=r_per_robot,
                              sinks_down=n_assigned, I_edge=I, P_edge=P))
        total += P

    return edge_data, total, _summary("Robot Network", edge_data, total, n_sinks)


# ===========================================================
#  Output helpers
# ===========================================================

def print_power_loss_comparison(summaries):
    """Formatted table + ratios relative to GeoSteiner."""
    hdr = f"{'Method':<20} {'Total Loss':>12} {'Robots':>8} {'Length':>12}"
    sep = "-" * len(hdr)
    print("\n" + sep)
    print("POWER LOSS COMPARISON")
    print(sep)
    print(hdr)
    print(sep)
    for s in summaries:
        print(f"{s['method']:<20} {s['total_loss']:>12.4f} "
              f"{s['total_robots']:>8d} {s['total_length']:>12.2f}")
    print(sep)

    geo = next((s for s in summaries if s["method"] == "GeoSteiner"), None)
    if geo and geo["total_loss"] > 0:
        print("\nRatios relative to GeoSteiner:")
        for s in summaries:
            rl = s["total_loss"] / geo["total_loss"]
            rr = s["total_robots"] / geo["total_robots"] if geo["total_robots"] else 0
            rx = s["total_length"] / geo["total_length"] if geo["total_length"] else 0
            print(f"  {s['method']:<20} Loss×{rl:.4f}  "
                  f"Robots×{rr:.4f}  Length×{rx:.4f}")
    print()


def plot_power_loss_comparison(summaries):
    """Two-panel bar chart → power_loss_comparison.png."""
    cmap = {"MST": "#a29bfe", "GeoSteiner": "#fd79a8", "Robot Network": "#3a7bd5"}
    methods = [s["method"] for s in summaries]
    losses  = [s["total_loss"] for s in summaries]
    robots  = [s["total_robots"] for s in summaries]
    colors  = [cmap.get(m, "#ccc") for m in methods]
    x = np.arange(len(methods))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in (a1, a2):
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")
        ax.title.set_color("white")
        ax.yaxis.label.set_color("white")

    a1.bar(x, losses, color=colors, edgecolor="white", linewidth=.5)
    a1.set_xticks(x); a1.set_xticklabels(methods, color="white")
    a1.set_ylabel("Power Loss"); a1.set_title("Total Power Loss per Method")
    for i, v in enumerate(losses):
        a1.text(i, v + max(losses)*.02, f"{v:.4f}", ha="center", color="white", fontsize=9)

    a2.bar(x, robots, color=colors, edgecolor="white", linewidth=.5)
    a2.set_xticks(x); a2.set_xticklabels(methods, color="white")
    a2.set_ylabel("Robot Count"); a2.set_title("Total Robots per Method")
    for i, v in enumerate(robots):
        a2.text(i, v + max(robots)*.02, str(v), ha="center", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig("power_loss_comparison.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("[INFO] Saved power_loss_comparison.png")


def plot_edge_power_heatmap(edge_data_list, labels, source, sinks_dict,
                           robots_dict=None):
    """One subplot per method, edges colored by P_edge → edge_power_heatmap.png."""
    nm = len(edge_data_list)
    fig, axes = plt.subplots(1, nm, figsize=(7 * nm, 6))
    fig.patch.set_facecolor("#0f0f1a")
    if nm == 1:
        axes = [axes]
    cm = plt.cm.plasma

    for idx, (edges, label) in enumerate(zip(edge_data_list, labels)):
        ax = axes[idx]
        ax.set_facecolor("#0f0f1a"); ax.set_title(label, color="white", fontsize=13)
        ax.tick_params(colors="white"); ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_color("white")
        if not edges:
            ax.text(.5, .5, "No edges", transform=ax.transAxes, ha="center", color="white")
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
            ax.plot(xs, ys, color=cm(norm(e["P_edge"])), linewidth=lw, solid_capstyle="round")
            mx, my = (xs[0]+xs[1])/2, (ys[0]+ys[1])/2
            ax.annotate(f"n={e['n_robots']} I={e['I_edge']:.2f}", (mx, my),
                        color="white", fontsize=7, ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#0f0f1a", ec="none", alpha=.7))

        ax.plot(*source, marker="^", color="limegreen", markersize=12, zorder=5,
                markeredgecolor="white", linewidth=0)
        for si in sinks_dict.values():
            ax.plot(*si["pos"], marker="*", color="red", markersize=12, zorder=5,
                    markeredgecolor="white", linewidth=0)

        # Draw leaf-robot-to-sink dashed lines for Robot Network panel
        if label == "Robot Network" and robots_dict:
            parent_rids = {r["parent"] for r in robots_dict.values() if r["parent"] is not None}
            leaf_rids = [rid for rid in robots_dict if rid not in parent_rids]
            for sid, si in sinks_dict.items():
                sp = si["pos"]
                best_rid, best_d = None, float("inf")
                for rid in leaf_rids:
                    d = dist(robots_dict[rid]["pos"], sp)
                    if d < best_d:
                        best_d = d
                        best_rid = rid
                if best_rid is not None:
                    rp = robots_dict[best_rid]["pos"]
                    ax.plot([rp[0], sp[0]], [rp[1], sp[1]],
                            color="#ff6b6b", linewidth=1.2, linestyle="--", alpha=0.8, zorder=2)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=.046, pad=.04)
        cb.set_label("P_edge", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

    plt.tight_layout()
    plt.savefig("edge_power_heatmap.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("[INFO] Saved edge_power_heatmap.png")


# ===========================================================
#  MAIN
# ===========================================================

if __name__ == "__main__":

    report_path = sys.argv[1] if len(sys.argv) > 1 else REPORT_FILE

    # ── 1. Read report ────────────────────────────────────────
    print(f"\n  Loading report: {report_path}")
    data = parse_report(report_path)

    src   = data["source"]
    sinks = data["sinks"]
    robots = data["robots"]
    sink_positions = [s["pos"] for s in sinks.values()]
    terminals = [src] + sink_positions
    n_sinks = len(sink_positions)

    print(f"  Source : {src}")
    print(f"  Sinks : {n_sinks}")
    for sid, s in sinks.items():
        print(f"    S{sid}  {s['pos']}  ({s['status']})")

    # ── 2. Compute MST edges ─────────────────────────────────
    mst_len, mst_edges = compute_mst_length(terminals)
    print(f"\n  MST length (Prim):  {mst_len:.2f}")

    # ── 3. Run GeoSteiner ────────────────────────────────────
    print(f"  Running GeoSteiner on {len(terminals)} terminals …")
    geo_len, smt_steiner_pts, smt_edges_idx = run_geosteiner(terminals)

    # ── 4. Power loss — MST ──────────────────────────────────
    mst_ed, mst_loss, mst_sum = mst_power_loss(src, sink_positions, mst_edges)

    # ── 5. Power loss — GeoSteiner ───────────────────────────
    geo_ed, geo_loss, geo_sum = geosteiner_power_loss(
        src, sink_positions, smt_steiner_pts, smt_edges_idx)

    # ── 6. Power loss — Robot Network ────────────────────────
    rn_ed, rn_loss, rn_sum = robot_network_power_loss(robots, sinks)

    # ── 7. Print results ─────────────────────────────────────
    for label, ed in [("MST", mst_ed), ("GeoSteiner", geo_ed)]:
        print(f"\n  {label} edges:")
        for e in ed:
            print(f"    {e['from_pos']} → {e['to_pos']}  "
                  f"L={e['length']:.1f}  n={e['n_robots']}  "
                  f"sinks↓={e['sinks_down']}  I={e['I_edge']:.4f}  P={e['P_edge']:.4f}")

    print(f"\n  Robot Network: {len(rn_ed)} edges, loss = {rn_loss:.4f}")

    summaries = [mst_sum, geo_sum, rn_sum]
    print_power_loss_comparison(summaries)

    # ── 8. Plots ──────────────────────────────────────────────
    plot_power_loss_comparison(summaries)
    plot_edge_power_heatmap(
        [mst_ed, geo_ed, rn_ed],
        ["MST", "GeoSteiner", "Robot Network"],
        src, sinks, robots_dict=robots,
    )
    print("  Done.")
