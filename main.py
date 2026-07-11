"""
Main Simulation for Multi-Sink Network Formation
Core Version - No additive leg.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import pygame
from pygame.math import Vector2 as Vec2

from constants import (
    Robot, Sink, Obstacle, Role, NodeType,
    WIDTH, HEIGHT, FPS, R, COMM_RANGE,
    MAX_ROBOTS, SPAWN_INTERVAL, PIVOT_SCORE_THRESHOLD,
    SINK_TOUCH_SETTLEMENT_THRESH, RECRUIT_SETTLE_THRESH,
    VISUALIZE_ON, FLOCK_SIZE, FLOCK_RADIUS, PIVOT_SCORE_MODE,
)
from assistive_functions import (
    robot_step, deliver_messages,
    compute_touched_sinks, remove_sink_from_subtree,
    calc_pivot_metric, is_local_pivot_candidate,
    enforce_pivot_split, make_demo_sinks,
    compute_hop_counts,
    tick_pivot_cooldowns,
)
from post_processing import generate_report


def main():
    # ---- pygame / display setup ----
    if VISUALIZE_ON:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Multi-sink network growth (core version)")
        clock = pygame.time.Clock()
        font  = pygame.font.SysFont(None, 22)
    else:
        pygame.init()
        clock  = pygame.time.Clock()
        screen = None
        font   = None

    SOURCE_POS = Vec2(WIDTH * 0.45, HEIGHT * 0.70)
    sinks      = make_demo_sinks(SOURCE_POS)
    obstacles  = [Obstacle(
        pos    = Vec2(random.uniform(200, WIDTH - 200), random.uniform(150, SOURCE_POS.y - 150)),
        radius = COMM_RANGE/10000 ,   # diameter = COMM_RANGE
    )]
    robots: List[Robot] = []

    # ---- Source robot ----
    robots.append(Robot(
        rid=0,
        pos=SOURCE_POS,
        role=Role.SOURCE,
        node_type=NodeType.STRAIGHT,
        max_children=1,
        branch_id="1",
        sink_indices=list(range(len(sinks))),
    ))

    # ---- Initial flock ----
    def spawn_flock():
        flock_center = Vec2(0, 800)
        for i in range(FLOCK_SIZE):
            if len(robots) >= MAX_ROBOTS:
                break
            angle  = 2 * math.pi * i / FLOCK_SIZE
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

    spawn_flock()

    sim_time             = 0.0
    next_spawn_time      = 1.0
    network_complete_time: Optional[float] = None
    touched              = [False] * len(sinks)
    MAX_SIM_TIME         = 1_200_000.0
    step_count           = 0

    running = True
    try:
        while running:

            # ---- Event handling ----
            if VISUALIZE_ON:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            else:
                if network_complete_time is not None or sim_time > MAX_SIM_TIME:
                    running = False
                    continue

            # ---- Spawn new flock ----
            if len(robots) < MAX_ROBOTS and sim_time >= next_spawn_time:
                spawn_flock()
                next_spawn_time += SPAWN_INTERVAL

            # ---- Compute touched sinks ----
            touched = compute_touched_sinks(robots, sinks)

            # ---- Check completion ----
            if network_complete_time is None and all(touched):
                network_complete_time = sim_time
                print(f"\n{'='*50}")
                print(f"NETWORK COMPLETE! All {len(sinks)} sinks touched.")
                print(f"Time: {network_complete_time:.2f}s   Robots: {len(robots)}")
                print(f"{'='*50}\n")

            # ---- Time step ----
            if VISUALIZE_ON:
                dt = min(clock.tick(FPS) / 1000.0, 0.1)
            else:
                dt = 1.0 / FPS
            sim_time += dt
            step_count += 1

            # ---- Update sink charging levels every step ----
            for s in sinks:
                if touched[s.sid]:
                    s.charging_level = min(100.0, s.charging_level + 0.01)
                else:
                    s.charging_level = max(0.0, s.charging_level - 0.01)

            # ---- Pivot cooldown tick (every step) ----
            tick_pivot_cooldowns(robots)

            # ---- Sink removal for settled robots ----
            for r in robots:
                if r.role != Role.NETWORK:
                    continue
                if r.desired_dir_d is None or r.desired_R is None or r.parent_id is None:
                    continue
                void_pos = robots[r.parent_id].pos + r.desired_dir_d * r.desired_R
                if (r.pos - void_pos).length() > SINK_TOUCH_SETTLEMENT_THRESH:
                    continue
                for s in sinks:
                    if s.sid not in r.sink_indices:
                        continue
                    if (s.pos - r.pos).length() <= SINK_TOUCH_SETTLEMENT_THRESH :
                        remove_sink_from_subtree(robots, r.rid, s.sid)

            # ---- Compute pivot errors ----
            errors: Dict[int, Optional[float]] = {}
            for r in robots:
                if r.role != Role.NETWORK or len(r.sink_indices) < 2:
                    errors[r.rid] = None
                else:
                    errors[r.rid] = calc_pivot_metric(r, robots, sinks)

            scores: Dict[int, Optional[float]] = {}
            for r in robots:
                E = errors.get(r.rid)
                if E is None:
                    scores[r.rid] = None
                else:
                    sink_bonus = 0.0
                    if PIVOT_SCORE_MODE == 1:
                        for s in sinks:
                            d = (s.pos - r.pos).length()
                            if d <= R:
                                sink_bonus = 0.3
                                break
                            elif d <= 2 * R:
                                sink_bonus = 0.2
                    scores[r.rid] = max(0.0, 1.0 - E / 120.0) + sink_bonus

            # ---- Identify and enforce pivot candidates ----
            pivot_candidates: List[int] = []
            for r in robots:
                if r.node_type == NodeType.PIVOT:
                    pivot_candidates.append(r.rid)
                    continue
                if not is_local_pivot_candidate(r.rid, robots, scores):
                    continue
                E = errors.get(r.rid)
                if E is None or max(0.0, 1.0 - E / 120.0) < PIVOT_SCORE_THRESHOLD:
                    continue
                pivot_candidates.append(r.rid)

            for pivot_id in pivot_candidates:
                enforce_pivot_split(pivot_id, robots, sinks)

            # ---- Deliver messages ----
            deliver_messages(robots)
            for r in robots:
                r.outbox.clear()

            # ---- Step all robots ----
            for r in robots:
                robot_step(r, dt, robots, sinks, touched, obstacles)

            # ---- Hop counts ----
            compute_hop_counts(robots)

            # ====================================================
            # DRAW
            # ====================================================
            if not VISUALIZE_ON:
                continue

            screen.fill((30, 30, 30))

            # Obstacles
            for obs in obstacles:
                pygame.draw.circle(screen, (180, 80, 0), obs.pos, int(obs.radius), 3)

            # Sinks
            for s in sinks:
                col = (60, 220, 60) if touched[s.sid] else (220, 60, 60)
                pygame.draw.circle(screen, col, s.pos, 9)
                screen.blit(font.render(f"S{s.sid}", True, (240, 240, 240)),
                            (s.pos.x + 10, s.pos.y - 8))
                charge_col = (int(255 * (1 - s.charging_level / 100)),
                              int(255 * (s.charging_level / 100)), 0)
                screen.blit(font.render(f"{s.charging_level:.0f}%", True, charge_col),
                            (s.pos.x + 10, s.pos.y + 6))

            # Source
            pygame.draw.circle(screen, (0, 200, 0), SOURCE_POS, 12)
            screen.blit(font.render("SRC", True, (255, 255, 255)),
                        (SOURCE_POS.x - 18, SOURCE_POS.y + 14))

            # Robots
            for r in robots:
                if r.role == Role.SOURCE:
                    continue
                if r.node_type == NodeType.PIVOT:
                    color, rad = (255, 0, 255), 8
                elif r.role == Role.NETWORK:
                    color, rad = (200, 200, 255), 7
                else:
                    color, rad = (0, 160, 255), 7

                pygame.draw.circle(screen, color, r.pos, rad)
                screen.blit(font.render(f"R{r.rid}", True, (255, 255, 255)),
                            (r.pos.x + 6, r.pos.y - 10))

                if r.role == Role.NETWORK:
                    E = errors.get(r.rid)
                    pivot_suit = max(0.0, 1.0 - E / 120.0) if E is not None else 0.0
                    sink_bonus = 0.0
                    if PIVOT_SCORE_MODE == 1:
                        for s in sinks:
                            d = (s.pos - r.pos).length()
                            if d <= R:
                                sink_bonus = 0.3
                                break
                            elif d <= 1.5 * R:
                                sink_bonus = 0.2
                    total = pivot_suit + sink_bonus
                    screen.blit(font.render(f"P:{pivot_suit:.2f} B:{sink_bonus:.2f}", True, (255, 255, 0)),
                                (r.pos.x + 6, r.pos.y + 8))
                    screen.blit(font.render(f"T:{total:.2f}", True, (0, 255, 128)),
                                (r.pos.x + 6, r.pos.y + 20))

            pygame.display.flip()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user (Ctrl+C)")

    # ---- Report ----
    print("\nGenerating final report...")
    generate_report(robots, sinks, touched, sim_time, network_complete_time, SOURCE_POS)

    if VISUALIZE_ON:
        pygame.quit()


def run_simulation(pivot_threshold: float, max_sim_time: float = 80.0) -> tuple:
    """
    Run one headless simulation with the given pivot_threshold.
    Returns (network_complete_time, total_robots).
    network_complete_time is None if the network did not complete within max_sim_time.
    """
    pivot_score_threshold = max(0.0, 1.0 - pivot_threshold / 120.0)

    pygame.init()

    SOURCE_POS = Vec2(WIDTH * 0.45, HEIGHT * 0.70)
    sinks      = make_demo_sinks(SOURCE_POS)
    robots: List[Robot] = []

    robots.append(Robot(
        rid=0,
        pos=SOURCE_POS,
        role=Role.SOURCE,
        node_type=NodeType.STRAIGHT,
        max_children=1,
        branch_id="1",
        sink_indices=list(range(len(sinks))),
    ))

    def spawn_flock():
        flock_center = Vec2(0, 800)
        for i in range(FLOCK_SIZE):
            if len(robots) >= MAX_ROBOTS:
                break
            angle  = 2 * math.pi * i / FLOCK_SIZE
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

    spawn_flock()

    sim_time              = 0.0
    next_spawn_time       = 1.0
    network_complete_time: Optional[float] = None
    touched               = [False] * len(sinks)
    dt                    = 1.0 / FPS

    while True:
        if network_complete_time is not None or sim_time > max_sim_time:
            break

        if len(robots) < MAX_ROBOTS and sim_time >= next_spawn_time:
            spawn_flock()
            next_spawn_time += SPAWN_INTERVAL

        touched = compute_touched_sinks(robots, sinks)

        if network_complete_time is None and all(touched):
            network_complete_time = sim_time

        sim_time += dt

        for r in robots:
            if r.role != Role.NETWORK:
                continue
            if r.desired_dir_d is None or r.desired_R is None or r.parent_id is None:
                continue
            void_pos = robots[r.parent_id].pos + r.desired_dir_d * r.desired_R
            if (r.pos - void_pos).length() > SINK_TOUCH_SETTLEMENT_THRESH:
                continue
            for s in sinks:
                if s.sid not in r.sink_indices:
                    continue
                if (s.pos - r.pos).length() <= SINK_TOUCH_SETTLEMENT_THRESH:
                    remove_sink_from_subtree(robots, r.rid, s.sid)

        errors: Dict[int, Optional[float]] = {}
        for r in robots:
            if r.role != Role.NETWORK or len(r.sink_indices) < 2:
                errors[r.rid] = None
            else:
                errors[r.rid] = calc_pivot_metric(r, robots, sinks)

        scores: Dict[int, Optional[float]] = {}
        for r in robots:
            E = errors.get(r.rid)
            if E is None:
                scores[r.rid] = None
            else:
                sink_bonus = 0.0
                if PIVOT_SCORE_MODE == 1:
                    for s in sinks:
                        d = (s.pos - r.pos).length()
                        if d <= R:
                            sink_bonus = 0.3
                            break
                        elif d <= 1.5 * R:
                            sink_bonus = 0.2
                scores[r.rid] = max(0.0, 1.0 - E / 120.0) + sink_bonus

        pivot_candidates: List[int] = []
        for r in robots:
            if r.node_type == NodeType.PIVOT:
                pivot_candidates.append(r.rid)
                continue
            if not is_local_pivot_candidate(r.rid, robots, scores):
                continue
            E = errors.get(r.rid)
            if E is None or max(0.0, 1.0 - E / 120.0) < pivot_score_threshold:
                continue
            pivot_candidates.append(r.rid)

        for pivot_id in pivot_candidates:
            enforce_pivot_split(pivot_id, robots, sinks)

        deliver_messages(robots)
        for r in robots:
            r.outbox.clear()

        for r in robots:
            robot_step(r, dt, robots, sinks, touched, [])

        compute_hop_counts(robots)

    pygame.quit()
    return network_complete_time, len(robots)


if __name__ == "__main__":
    main()
