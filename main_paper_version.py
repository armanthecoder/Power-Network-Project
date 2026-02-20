"""
Main Simulation for Multi-Sink Network Formation
Paper Version - Main pygame simulation loop
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pygame
from pygame.math import Vector2 as Vec2

from constants_paper_version import (
    Robot, Sink, Role, NodeType,
    WIDTH, HEIGHT, FPS, R, SPEED_MAX,
    MAX_ROBOTS, SPAWN_INTERVAL, PIVOT_THRESHOLD,
    SINK_TOUCH_DIST, SINK_TOUCH_SETTLEMENT_THRESH,
    VISUALIZE_ON
)

from assistive_functions_paper_version import (
    robot_step, deliver_messages,
    compute_touched_sinks, remove_sink_from_subtree,
    calc_pivot_metric, is_local_pivot_candidate,
    enforce_pivot_split, make_demo_sinks,
    guidance_dir_from_robot, safe_normalize
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
        max_children=3,
        branch_id="1",
        sink_indices=list(range(len(sinks))),
    )
    robots.append(r0)

    # Initial moving robot
    robots.append(Robot(
        rid=1,
        pos=SOURCE_POS ,
        role=Role.MOVING,
        node_type=NodeType.STRAIGHT,
        max_children=1,
        branch_id="1",
        sink_indices=list(range(len(sinks))),
    ))

    sim_time = 0.0
    sim_steps = 0
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

            # Spawn new robots
            if len(robots) < MAX_ROBOTS and sim_time >= next_spawn_time:
                rid = len(robots)
                spawn_pos = Vec2(0, 800)
                robots.append(Robot(
                    rid=rid,
                    pos=spawn_pos,
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
            sim_steps += 1

            # Sink removal for settled robots
            for r in robots:
                if r.role != Role.NETWORK:
                    continue
                
                if r.desired_dir_d is None or r.desired_R is None or r.parent_id is None:
                    continue
                parent = robots[r.parent_id]
                my_void_pos = parent.pos + r.desired_dir_d * r.desired_R
                if (r.pos - my_void_pos).length() > SINK_TOUCH_SETTLEMENT_THRESH:
                    continue
                
                for s in sinks:
                    if s.sid not in r.sink_indices:
                        continue
                    if (s.pos - r.pos).length() > SINK_TOUCH_DIST:
                        continue
                    remove_sink_from_subtree(robots, r.rid, s.sid)

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
                enforce_pivot_split(pivot_id, robots, sinks)

            # Deliver messages (communicate before act)
            deliver_messages(robots)

            # Clear outboxes after delivery
            for r in robots:
                r.outbox.clear()

            # Step robots
            for r in robots:
                robot_step(r, dt, robots, sinks, touched)

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
                        g = guidance_dir_from_robot(r, robots, sinks, touched)
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
    generate_report(robots, sinks, touched, sim_time, sim_steps, network_complete_time, SOURCE_POS)
    
    if VISUALIZE_ON:
        pygame.quit()


if __name__ == "__main__":
    main()
