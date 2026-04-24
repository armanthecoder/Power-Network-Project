"""
Main Simulation for Multi-Sink Network Formation
Paper Version - Main pygame simulation loop
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pygame
from pygame.math import Vector2 as Vec2

import math

from constants_paper_version import (
    Robot, Sink, Role, NodeType,
    WIDTH, HEIGHT, FPS, R, SPEED_MAX,
    MAX_ROBOTS, SPAWN_INTERVAL, PIVOT_THRESHOLD,
    SINK_TOUCH_DIST, SINK_TOUCH_SETTLEMENT_THRESH, RECRUIT_SETTLE_THRESH,
    VISUALIZE_ON, FLOCK_SIZE, FLOCK_RADIUS,
)

from assistive_functions_paper_version import (
    robot_step, deliver_messages,
    compute_touched_sinks, remove_sink_from_subtree, restore_sink_to_subtree,
    calc_pivot_metric, is_local_pivot_candidate,
    enforce_pivot_split, make_demo_sinks,
    guidance_dir_from_robot, safe_normalize,
    compute_hop_counts,
)

from post_processing_paper_version import generate_report


def main():
    """Main simulation entry point."""
    # Only initialize pygame display if visualization is on
    if VISUALIZE_ON:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Multi-sink network growth (recursive pivot splitting)")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 22)
    else:
        pygame.init()  # Still need pygame for Vec2
        clock = pygame.time.Clock()
        screen = None
        font = None

    SOURCE_POS = Vec2(WIDTH * 0.45, HEIGHT * 0.70)
    sinks = make_demo_sinks(SOURCE_POS)

    robots: List[Robot] = []

    # Source robot
    r0 = Robot(
        rid=0,
        pos=SOURCE_POS,
        role=Role.SOURCE,
        node_type=NodeType.STRAIGHT,
        max_children=1,
        branch_id="1",
        sink_indices=list(range(len(sinks))),
    )
    robots.append(r0)

    # Initial flock in circle formation near the source
    flock_center = Vec2(0, 800)
    for i in range(FLOCK_SIZE):
        angle = 2 * math.pi * i / FLOCK_SIZE
        offset = Vec2(math.cos(angle), math.sin(angle)) * (FLOCK_RADIUS * R)
        robots.append(Robot(
            rid=len(robots),
            pos=flock_center + offset,
            role=Role.MOVING,
            node_type=NodeType.STRAIGHT,
            max_children=1,
            branch_id="1",
            sink_indices=list(range(len(sinks))),
        ))

    sim_time = 0.0
    next_spawn_time = 1.0
    network_complete_time: Optional[float] = None
    touched = [False] * len(sinks)  # Initialize touched list
    MAX_SIM_TIME = 1200000.0  # DEBUG: stop after 60 seconds sim time

    running = True
    try:
        while running:

            # Event handling (only when visualization is on)
            if VISUALIZE_ON:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            
            # In headless mode, stop when network is complete or max time reached
            if not VISUALIZE_ON and (network_complete_time is not None or sim_time > MAX_SIM_TIME):
                running = False
                continue

            # Spawn a new flock in circle formation
            if len(robots) < MAX_ROBOTS and sim_time >= next_spawn_time:
                flock_center = Vec2(0, 800)
                for i in range(FLOCK_SIZE):
                    if len(robots) >= MAX_ROBOTS:
                        break
                    angle = 2 * math.pi * i / FLOCK_SIZE
                    offset = Vec2(math.cos(angle), math.sin(angle)) * (FLOCK_RADIUS * R)
                    robots.append(Robot(
                        rid=len(robots),
                        pos=flock_center + offset,
                        role=Role.MOVING,
                        node_type=NodeType.STRAIGHT,
                        max_children=1,
                        branch_id="1",
                        sink_indices=list(range(len(sinks))),
                    ))
                next_spawn_time += SPAWN_INTERVAL

            # Compute touched sinks
            touched = compute_touched_sinks(robots, sinks)

            # Check if all sinks are touched
            if network_complete_time is None and all(touched):
                network_complete_time = sim_time
                print(f"\n{'='*50}")
                print(f"NETWORK COMPLETE! All {len(sinks)} sinks touched.")
                print(f"Total time: {network_complete_time:.2f} seconds")
                print(f"Total robots used: {len(robots)}")
                print(f"{'='*50}\n")

            # Time step: use clock in visual mode, fixed dt in headless mode
            if VISUALIZE_ON:
                dt = clock.tick(FPS) / 1000.0
                dt = min(dt, 0.1)  # Cap dt to prevent huge jumps
            else:
                dt =( 1.0 / FPS )*1# Fixed timestep for headless
            sim_time += dt

            # Sink removal for settled robots
            for r in robots:
                if r.role != Role.NETWORK:
                    continue
                
                if r.desired_dir_d is None or r.desired_R is None or r.parent_id is None:
                    continue
                parent = robots[r.parent_id]
                my_void_pos = parent.pos + r.desired_dir_d * r.desired_R
                settle_thresh = RECRUIT_SETTLE_THRESH if (r.is_additive_leg or r.is_al_child) else SINK_TOUCH_SETTLEMENT_THRESH
                if (r.pos - my_void_pos).length() > settle_thresh:
                    continue
                
                for s in sinks:
                    if s.sid not in r.sink_indices:
                        continue
                    if (s.pos - r.pos).length() > SINK_TOUCH_DIST:
                        continue
                    # For leg robots remove from their network parent's subtree,
                    # so N1's normal children stop pursuing the now-covered sink.
                    subtree_root = r.parent_id if (r.is_additive_leg or r.is_al_child) else r.rid
                    remove_sink_from_subtree(robots, subtree_root, s.sid)

            # Organic touch: if a normal branch already touched an AL's target sink, release the AL branch
            for r in robots:
                if not r.is_additive_leg or not r.sink_indices:
                    continue
                sid = r.sink_indices[0]
                if not touched[sid]:
                    continue
                # Sink already covered organically — release AL and its ALC if any
                n1_id = r.parent_id
                print(f"[AL RELEASE] Sink {sid} touched organically, releasing AL {r.rid}")
                # Release ALC first
                if r.al_child_id is not None:
                    alc = robots[r.al_child_id]
                    r.al_child_id = None
                    r.max_children -= 1
                    if alc.rid in r.children_ids:
                        r.children_ids.remove(alc.rid)
                    alc.is_al_child = False
                    alc.al_child_parent_id = None
                    alc.parent_id = None
                    alc.role = Role.MOVING
                    alc.desired_R = None
                    alc.desired_dir_d = None
                    alc.sink_indices = []
                    alc.max_children = 1
                # Release AL from N1
                if n1_id is not None:
                    n1 = robots[n1_id]
                    n1.additive_leg_child_id = None
                    n1.max_children -= 1
                    if r.rid in n1.children_ids:
                        n1.children_ids.remove(r.rid)
                r.is_additive_leg = False
                r.parent_id = None
                r.role = Role.MOVING
                r.desired_R = None
                r.desired_dir_d = None
                r.sink_indices = []
                r.max_children = 1

            # ALC failure check: settled at void but sink out of reach → release and restore
            for r in robots:
                if not r.is_al_child or not r.sink_indices:
                    continue
                if r.desired_dir_d is None or r.desired_R is None or r.parent_id is None:
                    continue
                al_parent = robots[r.parent_id]
                void_pos = al_parent.pos + r.desired_dir_d * r.desired_R
                if (r.pos - void_pos).length() > RECRUIT_SETTLE_THRESH:
                    continue  # not settled yet
                sid = r.sink_indices[0]
                if (sinks[sid].pos - r.pos).length() <= SINK_TOUCH_DIST:
                    continue  # can reach it, no problem
                # Settled but sink still out of reach — release ALC and AL, restore sink
                n1_id = al_parent.parent_id  # AL's parent is N1
                print(f"[ALC FAIL] Robot {r.rid} settled but cannot reach sink {sid}, restoring to subtree")
                # Release ALC
                al_parent.al_child_id = None
                al_parent.max_children -= 1
                if r.rid in al_parent.children_ids:
                    al_parent.children_ids.remove(r.rid)
                r.is_al_child = False
                r.al_child_parent_id = None
                r.parent_id = None
                r.role = Role.MOVING
                r.desired_R = None
                r.desired_dir_d = None
                r.sink_indices = []
                r.max_children = 1
                # Release AL
                if n1_id is not None:
                    n1 = robots[n1_id]
                    n1.additive_leg_child_id = None
                    n1.max_children -= 1
                    if al_parent.rid in n1.children_ids:
                        n1.children_ids.remove(al_parent.rid)
                    # Restore sink to N1's normal subtree
                    for cid in n1.children_ids:
                        restore_sink_to_subtree(robots, cid, sid)
                    if sid not in n1.sink_indices:
                        n1.sink_indices.append(sid)
                al_parent.is_additive_leg = False
                al_parent.parent_id = None
                al_parent.role = Role.MOVING
                al_parent.desired_R = None
                al_parent.desired_dir_d = None
                al_parent.sink_indices = []
                al_parent.max_children = 1

            # Compute pivot errors
            errors: Dict[int, Optional[float]] = {}
            for r in robots:
                if r.role != Role.NETWORK or len(r.sink_indices) < 2:
                    errors[r.rid] = None
                else:
                    errors[r.rid] = calc_pivot_metric(r, robots, sinks)

            # Pick all local pivot candidates that pass the threshold
            pivot_candidates = []
            for r in robots:
                if r.node_type == NodeType.PIVOT:
                    pivot_candidates.append(r.rid)
                    continue
                
                if not is_local_pivot_candidate(r.rid, robots, errors):
                    continue
                E = errors[r.rid]
                if E is not None and E < PIVOT_THRESHOLD:
                    pivot_candidates.append(r.rid)

            # Enforce pivot split for each candidate
            for pivot_id in pivot_candidates:
                p = robots[pivot_id]
                # If this robot had an AL while still straight, release it now
                if p.additive_leg_child_id is not None:
                    al = robots[p.additive_leg_child_id]
                    sid = al.sink_indices[0] if al.sink_indices else None
                    # Release ALC if present
                    if al.al_child_id is not None:
                        alc = robots[al.al_child_id]
                        al.al_child_id = None
                        al.max_children -= 1
                        if alc.rid in al.children_ids:
                            al.children_ids.remove(alc.rid)
                        alc.is_al_child = False
                        alc.al_child_parent_id = None
                        alc.parent_id = None
                        alc.role = Role.MOVING
                        alc.desired_R = None
                        alc.desired_dir_d = None
                        alc.sink_indices = []
                        alc.max_children = 1
                    # Release AL from pivot
                    p.additive_leg_child_id = None
                    p.max_children -= 1
                    if al.rid in p.children_ids:
                        p.children_ids.remove(al.rid)
                    al.is_additive_leg = False
                    al.parent_id = None
                    al.role = Role.MOVING
                    al.desired_R = None
                    al.desired_dir_d = None
                    al.sink_indices = []
                    al.max_children = 1
                    # Restore the sink to the pivot and all its normal children
                    if sid is not None:
                        for cid in p.children_ids:
                            restore_sink_to_subtree(robots, cid, sid)
                        if sid not in p.sink_indices:
                            p.sink_indices.append(sid)
                        print(f"[PIVOT PROMOTE] Released AL from pivot {pivot_id}, restored sink {sid}")
                enforce_pivot_split(pivot_id, robots, sinks)

            # Deliver messages (communicate before act)
            deliver_messages(robots)

            # Clear outboxes after delivery
            for r in robots:
                r.outbox.clear()

            # Step robots
            for r in robots:
                robot_step(r, dt, robots, sinks, touched)

            # Compute hop counts for moving robots
            compute_hop_counts(robots)

            # =========================
            # DRAW (only if visualization is enabled)
            # =========================
            if VISUALIZE_ON:
                screen.fill((30, 30, 30))

                # Draw sinks
                for s in sinks:
                    col = (220, 60, 60) if not touched[s.sid] else (60, 220, 60)
                    pygame.draw.circle(screen, col, s.pos, 9)
                    screen.blit(font.render(f"S{s.sid}", True, (240, 240, 240)),
                                (s.pos.x + 10, s.pos.y - 8))

                # Draw source
                pygame.draw.circle(screen, (0, 200, 0), SOURCE_POS, 12)
                screen.blit(font.render("SOURCE", True, (255, 255, 255)),
                            (SOURCE_POS.x - 35, SOURCE_POS.y + 14))

                # Draw robots
                for r in robots:
                    if r.role == Role.SOURCE:
                        color, rad = (0, 200, 0), 10
                    elif r.is_additive_leg:
                        color, rad = (255, 140, 0), 8   # orange
                    elif r.is_al_child:
                        color, rad = (255, 210, 0), 7   # yellow
                    elif r.node_type == NodeType.PIVOT:
                        color, rad = (255, 0, 255), 8
                    elif r.role == Role.NETWORK:
                        color, rad = (200, 200, 255), 7
                    else:
                        color, rad = (0, 160, 255), 7

                    pygame.draw.circle(screen, color, r.pos, rad)

                    # Label
                    screen.blit(font.render(f"R{r.rid}", True, (255, 255, 255)),
                                (r.pos.x + 6, r.pos.y - 10))

                    # Hop count (moving robots only)
                    if r.role == Role.MOVING and r.hop_count is not None:
                        screen.blit(font.render(f"h{r.hop_count}", True, (255, 180, 0)),
                                    (r.pos.x + 6, r.pos.y + 8))

                    # Additive leg / AL child labels
                    if r.is_additive_leg:
                        screen.blit(font.render("AL", True, (255, 140, 0)),
                                    (r.pos.x + 6, r.pos.y + 8))
                    elif r.is_al_child:
                        screen.blit(font.render("ALC", True, (255, 210, 0)),
                                    (r.pos.x + 6, r.pos.y + 8))

                    # Pivot metric text
                    E = errors.get(r.rid)
                    if E is not None:
                        screen.blit(font.render(f"{E:.1f}", True, (255, 255, 0)),
                                    (r.pos.x + 6, r.pos.y + 8))

                    # Void marker
                    if r.parent_id is not None and r.desired_dir_d is not None and r.desired_R is not None:
                        void_pos = robots[r.parent_id].pos + r.desired_dir_d * r.desired_R
                        pygame.draw.circle(screen, (255, 255, 0), void_pos, 4, 1)
                    elif r.void_pos_vis is not None:
                        pygame.draw.circle(screen, (255, 255, 0), r.void_pos_vis, 4, 1)

                    # Show assigned sinks count
                    if r.role == Role.NETWORK and r.parent_id is not None:
                        screen.blit(font.render(f"|S|={len(r.sink_indices)}", True, (160, 160, 160)),
                                    (r.pos.x - 18, r.pos.y + 14))
                    
                    # Show branch ID
                    if r.role == Role.NETWORK or r.role == Role.SOURCE:
                        screen.blit(font.render(f"B:{r.branch_id}", True, (100, 200, 255)),
                                    (r.pos.x - 20, r.pos.y - 25))

                    # --- GUIDANCE VECTOR arrow ---
                    if r.role == Role.NETWORK:
                        g = guidance_dir_from_robot(r, robots, touched)
                        if g is not None:
                            gu = safe_normalize(g)
                            arrow_len = 30
                            tip = r.pos + gu * arrow_len
                            pygame.draw.line(screen, (0, 255, 128), r.pos, tip, 2)
                            # small arrowhead
                            perp = Vec2(-gu.y, gu.x)
                            pygame.draw.line(screen, (0, 255, 128), tip, tip - gu * 6 + perp * 4, 1)
                            pygame.draw.line(screen, (0, 255, 128), tip, tip - gu * 6 - perp * 4, 1)

                    # --- CAN_RECRUIT label (1 or 0) ---
                    if r.role == Role.NETWORK and r.node_type != NodeType.PIVOT:
                        cr_done = all(touched[sid] for sid in r.sink_indices) if r.sink_indices else False
                        cr_leaf_zero = len(r.sink_indices) == 0 and not r.children_ids
                        if cr_done:
                            cr = r.has_free_slot() and not cr_leaf_zero
                        else:
                            cr_has_distant = any(
                                (sinks[sid].pos - r.pos).length() > SINK_TOUCH_DIST
                                for sid in r.sink_indices
                            )
                            cr_all_near = not cr_has_distant and len(r.sink_indices) > 0
                            cr = r.has_free_slot() and not cr_all_near and not cr_leaf_zero
                        cr_col = (0, 255, 0) if cr else (255, 80, 80)
                        screen.blit(font.render(f"CR:{int(cr)}", True, cr_col),
                                    (r.pos.x - 20, r.pos.y + 26))
                    elif r.node_type == NodeType.PIVOT:
                        cr_done = all(touched[sid] for sid in r.sink_indices) if r.sink_indices else False
                        cr_leaf_zero = len(r.sink_indices) == 0 and not r.children_ids
                        cr = (not cr_done) and r.has_free_slot() and not cr_leaf_zero
                        cr_col = (0, 255, 0) if cr else (255, 80, 80)
                        screen.blit(font.render(f"CR:{int(cr)}", True, cr_col),
                                    (r.pos.x - 20, r.pos.y + 26))

                pygame.display.flip()

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user (Ctrl+C)")

    # Generate report when simulation ends (either complete or Ctrl+C)
    print("\nGenerating final report...")
    generate_report(robots, sinks, touched, sim_time, network_complete_time, SOURCE_POS)
    
    if VISUALIZE_ON:
        pygame.quit()


if __name__ == "__main__":
    main()
