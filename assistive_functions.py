"""
Assistive Functions for Multi-Sink Network Formation Simulation
Core Version - Geometry, void controller, boids, pivot logic, guidance, messaging.
No additive-leg (AL) code.
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

from pygame.math import Vector2 as Vec2

from constants import (
    Robot, Sink, Role, NodeType,
    R, COMM_RANGE, SPEED_MAX, SINK_TOUCH_DIST,
    K_R, K_THETA, POS_THRESH,
    SINK_TOUCH_SETTLEMENT_THRESH, GUIDANCE_STICK_FRAMES,
    NUM_SINKS, SINK_SPACE_WIDTH, SINK_SPACE_HEIGHT, SINK_MIN_SEP,
    BOID_NEIGHBOR_RADIUS, BOID_SEP_RADIUS,
    BOID_SEP_WEIGHT, BOID_COH_WEIGHT, BOID_ALIGN_WEIGHT,
    RECRUIT_SETTLE_THRESH,
)


# =============================================================================
# Geometry Helpers
# =============================================================================

def safe_normalize(v: Vec2) -> Vec2:
    if v.length_squared() < 1e-12:
        return Vec2(0, 0)
    return v.normalize()


def angle_between_deg(a: Vec2, b: Vec2) -> float:
    if a.length_squared() < 1e-12 or b.length_squared() < 1e-12:
        return 0.0
    dot = max(-1.0, min(1.0, a.normalize().dot(b.normalize())))
    return math.degrees(math.acos(dot))


def cross2(a: Vec2, b: Vec2) -> float:
    return a.x * b.y - a.y * b.x


def split_sinks_star_port(
    node_pos:  Vec2,
    parent_pos: Vec2,
    sink_ids:  List[int],
    sinks:     List[Sink],
) -> Tuple[List[int], List[int]]:
    """
    Split sink_ids into STARBOARD / PORT relative to the incoming direction
    parent -> node.
      STARBOARD = cross(incoming, node->sink) > 0
      PORT      = cross(incoming, node->sink) < 0
    """
    incoming   = node_pos - parent_pos
    incoming_u = safe_normalize(incoming)
    if incoming_u.length_squared() < 1e-12:
        return sink_ids[:], []

    star: List[int] = []
    port: List[int] = []
    for sid in sink_ids:
        v  = sinks[sid].pos - node_pos
        vu = safe_normalize(v)
        s  = cross2(incoming_u, vu)
        if s > 0:
            star.append(sid)
        elif s < 0:
            port.append(sid)
        else:
            # exactly on the line: assign to the smaller side
            (star if len(star) <= len(port) else port).append(sid)
    return star, port


def avg_dir_to_sinks(node_pos: Vec2, sink_ids: List[int], sinks: List[Sink]) -> Vec2:
    """Average unit direction from node_pos toward each sink in sink_ids."""
    s = Vec2(0, 0)
    for sid in sink_ids:
        d = sinks[sid].pos - node_pos
        if d.length_squared() > 1e-12:
            s += d.normalize()
    if s.length_squared() < 1e-12:
        return Vec2(1, 0)
    return s.normalize()


def bisection_direction(
    node_pos:   Vec2,
    parent_pos: Vec2,
    sink_ids:   List[int],
    sinks:      List[Sink],
) -> Vec2:
    """
    Bisection heading for a network node.
    Split sinks into star / port, compute the average unit vector to each side,
    then return their bisector.  Falls back to the average of all sinks when
    one side is empty.
    """
    star, port = split_sinks_star_port(node_pos, parent_pos, sink_ids, sinks)
    if not star or not port:
        return avg_dir_to_sinks(node_pos, sink_ids, sinks)

    v_star = avg_dir_to_sinks(node_pos, star, sinks)
    v_port = avg_dir_to_sinks(node_pos, port, sinks)

    b = v_star + v_port
    if b.length_squared() < 1e-12:
        return avg_dir_to_sinks(node_pos, sink_ids, sinks)
    return b.normalize()


# =============================================================================
# Main step dispatcher
# =============================================================================

def robot_step(
    robot:     Robot,
    dt:        float,
    robots:    List[Robot],
    sinks:     List[Sink],
    touched:   List[bool],
    obstacles: list,
):
    """Dispatch to the correct behaviour, then update the velocity estimate."""
    robot.prev_pos = Vec2(robot.pos)

    if robot.role == Role.SOURCE:
        _try_recruit(robot, robots, sinks, touched)
    elif robot.parent_id is None:
        _free_behavior(robot, dt, robots, sinks, touched)
    else:
        _attached_behavior(robot, dt, robots, sinks, touched, obstacles)

    if dt > 0:
        robot.vel = (robot.pos - robot.prev_pos) / dt
    else:
        robot.vel = Vec2(0, 0)


# =============================================================================
# FREE (MOVING) behaviour
# =============================================================================

def _free_behavior(
    robot:   Robot,
    dt:      float,
    robots:  List[Robot],
    sinks:   List[Sink],
    touched: List[bool],
):
    """
    Free mover: responds to recruitment broadcasts and follows guidance.

    Recruitment overrides any guidance lock.
    Guidance picks the highest binary-valued branch_id visible within COMM_RANGE.
    Falls back to drifting toward the source (robots[0]) when no guidance exists.
    """
    # ---- Step 0: Accept a network attachment offer (must come first) ----
    for msg in robot.inbox:
        if msg.msg_type == "ACCEPT_CHILD" and msg.payload.get("child_id") == robot.rid:
            parent_id    = msg.payload["parent_id"]
            parent_robot = robots[parent_id]

            robot.parent_id     = parent_id
            robot.role          = Role.NETWORK
            robot.desired_R     = float(msg.payload["R"])
            d                   = Vec2(msg.payload["d_x"], msg.payload["d_y"])
            robot.desired_dir_d = safe_normalize(d)
            robot.sink_indices  = list(msg.payload.get("sink_indices", robot.sink_indices))
            robot.branch_id     = msg.payload.get("branch_id", "1")

            parent_type = parent_robot.node_type
            gp_type     = (robots[parent_robot.parent_id].node_type
                           if parent_robot.parent_id is not None else "NONE")
            print(f"[NETWORK] Robot {robot.rid} attached to parent {parent_id} "
                  f"(branch={robot.branch_id}, parent_type={parent_type}, "
                  f"gp_type={gp_type}, sinks={robot.sink_indices[:8]})")
            return  # now a NETWORK robot; no further free-behaviour this tick

    # ---- Step 1: Check for recruitment messages ----
    valid_askers: List[int] = []
    for msg in robot.inbox:
        if msg.msg_type != "ASK_PARENT":
            continue

        asker_id = msg.sender_id
        asker    = robots[asker_id]
        payload  = msg.payload

        if "need_star" in payload and "need_port" in payload:
            # Pivot recruiting: only respond if we are on the needed side
            incoming_u    = Vec2(payload["incoming_x"], payload["incoming_y"])
            v_pc_u        = safe_normalize(robot.pos - asker.pos)
            side_is_star  = cross2(incoming_u, v_pc_u) > 0

            if (side_is_star and payload["need_star"]) or \
               (not side_is_star and payload["need_port"]):
                valid_askers.append(asker_id)
        else:
            valid_askers.append(asker_id)

    if valid_askers:
        # Recruitment overrides guidance lock
        robot.guidance_lock_robot_id = None
        robot.guidance_lock_frames   = 0

        best_id = min(valid_askers,
                      key=lambda aid: (robots[aid].pos - robot.pos).length())
        robot.broadcast("PARENT_STATUS", has_parent=False, responding_to=best_id)
        _simple_move_towards(robot, robots[best_id].pos, dt)
        return  # skip guidance this tick

    # ---- Step 2: Follow guidance ----
    locked_guidance:    Optional[Vec2]  = None
    locked_guide_robot: Optional[Robot] = None

    # Try to reuse the existing lock
    if robot.guidance_lock_robot_id is not None and robot.guidance_lock_frames > 0:
        gid = robot.guidance_lock_robot_id
        if 0 <= gid < len(robots):
            guide = robots[gid]
            if guide.role == Role.NETWORK:
                g = guidance_dir_from_robot(guide, robots, touched)
                if g is None:
                    g = guide.pos - robot.pos
                if g.length_squared() > 1e-12:
                    locked_guidance    = g
                    locked_guide_robot = guide

    # If lock expired or invalid, find the best visible guide
    if locked_guidance is None:
        robot.guidance_lock_robot_id = None
        robot.guidance_lock_frames   = 0

        best_guidance:    Optional[Vec2]  = None
        best_score:       int             = -1
        best_dist:        float           = float("inf")
        best_guide_robot: Optional[Robot] = None

        for other in robots:
            if other.rid == robot.rid or other.role != Role.NETWORK:
                continue
            dist = (other.pos - robot.pos).length()
            if dist > COMM_RANGE:
                continue

            g = guidance_dir_from_robot(other, robots, touched)
            if g is None:
                continue

            branch_val = int(other.branch_id, 2) if other.branch_id else 0

            if branch_val > best_score or (branch_val == best_score and dist < best_dist):
                best_score       = branch_val
                best_dist        = dist
                best_guidance    = g
                best_guide_robot = other

        if best_guidance is not None:
            robot.guidance_lock_robot_id = best_guide_robot.rid
            robot.guidance_lock_frames   = GUIDANCE_STICK_FRAMES
            locked_guidance              = best_guidance
            locked_guide_robot           = best_guide_robot

    # ---- Step 3: Move ----
    if locked_guidance is not None:
        guidance_dir = safe_normalize(locked_guidance)
        if locked_guide_robot is not None:
            guide_R      = locked_guide_robot.desired_R if locked_guide_robot.desired_R is not None else R
            goal_target  = locked_guide_robot.pos + guidance_dir * guide_R
            speed        = max(locked_guide_robot.vel.length(), SPEED_MAX)
        else:
            goal_target = robot.pos + guidance_dir * R
            speed       = SPEED_MAX

        if robot.guidance_lock_frames > 0:
            robot.guidance_lock_frames -= 1
    else:
        # No guidance: drift back toward the source
        goal_target = robots[0].pos
        speed       = SPEED_MAX

    move_dir = _boid_move_direction(robot, robots, goal_target)
    _move_towards_with_speed(robot, robot.pos + move_dir * R, speed, dt)


# =============================================================================
# ATTACHED (NETWORK) behaviour
# =============================================================================

def _attached_behavior(
    robot:     Robot,
    dt:        float,
    robots:    List[Robot],
    sinks:     List[Sink],
    touched:   List[bool],
    obstacles: list,
):
    """
    Network robot:
      1. Void controller: maintain desired position relative to parent.
         Void position is repelled from nearby obstacles.
      2. Seeing-sink check: if within void + sink-touch range, set sees_sink=True,
         capacity → 0, and notify ancestors up to the nearest pivot.
      3. Recruit children when settled and capacity allows.
    """
    # ---- 1. Void controller ----
    if (robot.desired_R is not None
            and robot.desired_dir_d is not None
            and robot.parent_id is not None):

        parent   = robots[robot.parent_id]
        void_pos = parent.pos + robot.desired_dir_d * robot.desired_R

        r_vec = parent.pos - robot.pos
        dist  = r_vec.length()

        if dist > 1e-9:
            u      = r_vec / dist
            u_perp = Vec2(-u.y, u.x)
            u_star = -robot.desired_dir_d

            e_r     = dist - robot.desired_R
            e_theta = cross2(u, u_star)

            v = (K_R * e_r) * u + (-K_THETA * e_theta) * u_perp

            # ---- Obstacle repulsion: added directly to velocity ----
            # Linear: max at obstacle center (d=0), zero at obstacle edge (d=obs.radius).
            for obs in obstacles:
                to_robot = robot.pos - obs.pos
                d        = to_robot.length()
                if 1e-9 < d < obs.radius:
                    strength = 1.0 - d / obs.radius
                    v += safe_normalize(to_robot) * strength * SPEED_MAX

            if v.length() > SPEED_MAX:
                v = v.normalize() * SPEED_MAX

            robot.pos        += v * dt
            robot.void_pos_vis = void_pos

    # ---- 2. Seeing-sink check ----
    _update_sees_sink(robot, robots, sinks)

    # ---- 2.5 Broadcast battery level upstream when touching a sink ----
    if robot.sees_sink and robot.sink_indices:
        sid = robot.sink_indices[0]
        robot.broadcast(
            "BATTERY_REPORT",
            source_rid=robot.rid,
            sid=sid,
            level=sinks[sid].charging_level,
        )

    # ---- 3. Recruitment ----
    _try_recruit(robot, robots, sinks, touched)

    # ---- 4. Battery report relay / pivot pruning ----
    _handle_battery_report(robot, robots, sinks)


# =============================================================================
# Seeing-sink mechanism
# =============================================================================

def _update_sees_sink(robot: Robot, robots: List[Robot], sinks: List[Sink]):
    """
    When a robot is within SINK_TOUCH_DIST of an assigned sink:
      - Remove that sink from each child's subtree so they no longer target it.

    If the robot's ONLY assigned sink is the one it directly sees:
      - sees_sink = True → max_children = 0 (no adoption, no guidance).

    When that condition clears:
      - sees_sink = False, max_children restored to node-type default.
    """
    if not robot.sink_indices:
        return

    seen_sinks = [
        sid for sid in robot.sink_indices
        if (sinks[sid].pos - robot.pos).length() <= SINK_TOUCH_DIST
    ]

    # Remove every seen sink from each child's entire subtree
    for sid in seen_sinks:
        for cid in robot.children_ids:
            remove_sink_from_subtree(robots, cid, sid)

    only_sink_seen = (len(robot.sink_indices) == 1 and len(seen_sinks) == 1)

    was_seeing = robot.sees_sink

    if only_sink_seen and not was_seeing:
        robot.sees_sink    = True
        robot.max_children = 0
        print(f"[SEES_SINK] Robot {robot.rid} sees its only sink {seen_sinks[0]} → capacity=0")

    elif was_seeing and not only_sink_seen:
        robot.sees_sink    = False
        default_cap        = 2 if robot.node_type == NodeType.PIVOT else 1
        robot.max_children = default_cap
        print(f"[SEES_SINK] Robot {robot.rid} no longer sees only sink → capacity={default_cap}")


# =============================================================================
# Battery-report relay and pivot pruning
# =============================================================================

def _handle_battery_report(robot: Robot, robots: List[Robot], sinks: List[Sink]):
    """
    Process BATTERY_REPORT messages received from direct children only.

    STRAIGHT node  → relay the message one hop upstream (add to outbox).
    PIVOT node     → accumulate reports from STAR and PORT sides;
                     once both arrive, release the higher-charge branch
                     and start a 1000-step recruitment cooldown.
    """
    for msg in robot.inbox:
        if msg.msg_type != "BATTERY_REPORT":
            continue
        if msg.sender_id not in robot.children_ids:
            continue  # ignore: not from a direct child

        if robot.node_type == NodeType.PIVOT:
            if robot.pivot_cooldown > 0:
                continue  # waiting out cooldown

            parent_pos = (robots[robot.parent_id].pos if robot.parent_id is not None
                          else robot.pos - Vec2(1, 0))
            incoming_u = safe_normalize(robot.pos - parent_pos)
            sgn        = cross2(incoming_u,
                                safe_normalize(robots[msg.sender_id].pos - robot.pos))
            side = "STAR" if sgn > 0 else "PORT"

            robot.battery_reports[side] = msg.payload["level"]

            # Wait until reports from both sides have arrived
            if "STAR" not in robot.battery_reports or "PORT" not in robot.battery_reports:
                continue

            star_charge = robot.battery_reports["STAR"]
            port_charge = robot.battery_reports["PORT"]

            # Map each current child to its side
            side_to_cid: Dict[str, int] = {}
            for cid in robot.children_ids:
                s = cross2(incoming_u, safe_normalize(robots[cid].pos - robot.pos))
                side_to_cid["STAR" if s > 0 else "PORT"] = cid

            if len(side_to_cid) < 2:
                robot.battery_reports.clear()
                continue

            # Keep the branch with LOWER charge (needs charging); remove the higher one
            if star_charge <= port_charge:
                keep_side, remove_side = "STAR", "PORT"
            else:
                keep_side, remove_side = "PORT", "STAR"

            keep_charge   = robot.battery_reports[keep_side]
            remove_charge = robot.battery_reports[remove_side]

            # Only prune if the candidate branch is above 50% — below that it still needs charging
            if remove_charge <= 50.0:
                robot.battery_reports.clear()
                continue

            remove_cid = side_to_cid[remove_side]

            release_branch_to_moving(robots, sinks, remove_cid)
            robot.children_ids.remove(remove_cid)
            robot.max_children   = 1       # block new child until cooldown expires
            robot.pivot_cooldown = 1000
            robot.battery_reports.clear()

            print(f"[PRUNE] Pivot R{robot.rid}: keep {keep_side} "
                  f"(R{side_to_cid[keep_side]} {keep_charge:.1f}%) "
                  f"remove {remove_side} (R{remove_cid} {remove_charge:.1f}%) "
                  f"cooldown=1000")

        else:
            # STRAIGHT node: relay one hop upstream
            if robot.parent_id is not None:
                robot.broadcast(
                    "BATTERY_REPORT",
                    source_rid=msg.payload["source_rid"],
                    sid=msg.payload["sid"],
                    level=msg.payload["level"],
                )


# =============================================================================
# Recruitment helper
# =============================================================================

def _try_recruit(
    robot:   Robot,
    robots:  List[Robot],
    sinks:   List[Sink],
    touched: List[bool],
):
    """
    Decide whether to broadcast ASK_PARENT and accept incoming PARENT_STATUS replies.

    Conditions for STRAIGHT node:
      - settled at void position
      - has a free slot
      - not all assigned sinks are within immediate touch distance (still needs a child)

    Conditions for PIVOT node:
      - settled
      - has a free slot
      - not fully done

    A robot that sees_sink has max_children=0 so has_free_slot() returns False
    naturally — no special guard needed here.
    """
    is_settled = False
    if robot.parent_id is None:
        is_settled = True  # SOURCE has no parent; it is always at its fixed position
    elif (robot.desired_dir_d is not None and robot.desired_R is not None):
        void_pos   = robots[robot.parent_id].pos + robot.desired_dir_d * robot.desired_R
        is_settled = (robot.pos - void_pos).length() <= RECRUIT_SETTLE_THRESH

    done = (all(touched[sid] for sid in robot.sink_indices)
            if robot.sink_indices else False)

    leaf_zero = len(robot.sink_indices) == 0 and not robot.children_ids

    if robot.node_type == NodeType.PIVOT:
        can_recruit = is_settled and (not done) and robot.has_free_slot() and (not leaf_zero)

        if can_recruit:
            parent_ref  = (robots[robot.parent_id].pos if robot.parent_id is not None
                           else robot.pos - Vec2(1, 0))
            incoming    = robot.pos - parent_ref
            incoming_u  = safe_normalize(incoming)

            has_star = has_port = False
            for cid in robot.children_ids:
                sgn = cross2(incoming_u, safe_normalize(robots[cid].pos - robot.pos))
                if sgn > 0:
                    has_star = True
                else:
                    has_port = True

            robot.broadcast(
                "ASK_PARENT",
                need_star=not has_star,
                need_port=not has_port,
                incoming_x=float(incoming_u.x),
                incoming_y=float(incoming_u.y),
            )
    else:
        has_distant = any(
            (sinks[sid].pos - robot.pos).length() > SINK_TOUCH_DIST
            for sid in robot.sink_indices
        )
        all_near   = (not has_distant) and len(robot.sink_indices) > 0
        can_recruit = is_settled and robot.has_free_slot() and (not all_near) and (not leaf_zero)

        if can_recruit:
            robot.broadcast("ASK_PARENT")

    if not can_recruit:
        return

    # ---- Accept a reply ----
    for msg in robot.inbox:
        if msg.msg_type != "PARENT_STATUS":
            continue
        if msg.payload.get("has_parent", True):
            continue
        if msg.payload.get("responding_to") != robot.rid:
            continue
        if not robot.has_free_slot():
            break

        child_id = msg.sender_id

        if robot.parent_id is None:
            ref_parent_pos = robot.pos - Vec2(1, 0)
        else:
            ref_parent_pos = robots[robot.parent_id].pos

        if robot.node_type == NodeType.PIVOT:
            parent_ref  = (robots[robot.parent_id].pos if robot.parent_id is not None
                           else robot.pos - Vec2(1, 0))
            incoming_u  = safe_normalize(robot.pos - parent_ref)
            v_pc_u      = safe_normalize(robots[child_id].pos - robot.pos)
            side_sgn    = cross2(incoming_u, v_pc_u)

            # Skip if that side already has a child
            existing_sides = {"STAR": False, "PORT": False}
            for eid in robot.children_ids:
                es = cross2(incoming_u, safe_normalize(robots[eid].pos - robot.pos))
                existing_sides["STAR" if es > 0 else "PORT"] = True

            child_side = "STAR" if side_sgn > 0 else "PORT"
            if existing_sides[child_side]:
                continue

            star_sinks, port_sinks = split_sinks_star_port(
                robot.pos, parent_ref, robot.sink_indices, sinks
            )
            child_sinks    = star_sinks if side_sgn > 0 else port_sinks
            child_branch_id = robot.branch_id + ("1" if side_sgn > 0 else "0")
            d               = bisection_direction(robot.pos, ref_parent_pos, child_sinks, sinks)
        else:
            child_sinks    = robot.sink_indices[:]
            child_branch_id = robot.branch_id
            d               = bisection_direction(robot.pos, ref_parent_pos, robot.sink_indices, sinks)

        robot.children_ids.append(child_id)
        robot.broadcast(
            "ACCEPT_CHILD",
            child_id=child_id,
            parent_id=robot.rid,
            R=R,
            d_x=float(d.x),
            d_y=float(d.y),
            sink_indices=child_sinks,
            branch_id=child_branch_id,
        )


# =============================================================================
# Movement helpers
# =============================================================================

def _simple_move_towards(robot: Robot, target: Vec2, dt: float):
    _move_towards_with_speed(robot, target, SPEED_MAX, dt)


def _move_towards_with_speed(robot: Robot, target: Vec2, speed: float, dt: float):
    d = target - robot.pos
    if d.length_squared() > 1e-12 and speed > 1e-9:
        robot.vel  = d.normalize() * speed
        robot.pos += robot.vel * dt
    else:
        robot.vel = Vec2(0, 0)


def _boid_move_direction(robot: Robot, robots: List[Robot], goal_target: Vec2) -> Vec2:
    """
    Reynolds boid: separation + cohesion + goal-heading (replaces alignment).
    Only other free MOVING robots count as flock-mates.
    """
    sep    = Vec2(0, 0)
    center = Vec2(0, 0)
    count  = 0

    for other in robots:
        if other.rid == robot.rid:
            continue
        if other.role != Role.MOVING or other.parent_id is not None:
            continue
        dist = (other.pos - robot.pos).length()
        if dist > BOID_NEIGHBOR_RADIUS:
            continue

        count  += 1
        center += other.pos
        if 1e-9 < dist < BOID_SEP_RADIUS:
            sep += safe_normalize(robot.pos - other.pos) * (BOID_SEP_RADIUS / dist)

    goal_dir = safe_normalize(goal_target - robot.pos)

    if count == 0:
        return goal_dir

    coh_dir = safe_normalize(center / count - robot.pos)
    sep_dir = safe_normalize(sep)

    combined = (BOID_SEP_WEIGHT  * sep_dir
                + BOID_COH_WEIGHT * coh_dir
                + BOID_ALIGN_WEIGHT * goal_dir)

    if combined.length_squared() < 1e-12:
        return goal_dir
    return combined.normalize()


# =============================================================================
# Guidance (multi-sink, both node types)
# =============================================================================

def guidance_dir_from_robot(
    r:       Robot,
    robots:  List[Robot],
    touched: Optional[List[bool]],
) -> Optional[Vec2]:
    """
    Guidance direction emitted by a NETWORK robot toward movers.

    STRAIGHT: guide toward child(ren) if subtree is not done.
    PIVOT:    guide toward the unfinished branch(es).
              - both branches unfinished → star direction
              - only one unfinished       → that direction
              - both done                → None
    """
    if r.role != Role.NETWORK:
        return None
    if r.sees_sink:
        return None
    if not r.sink_indices:
        return None

    def subtree_done(node: Robot) -> bool:
        if not node.sink_indices:
            return True
        if touched is None:
            return False
        return all(touched[sid] for sid in node.sink_indices)

    my_done = subtree_done(r)

    # ---- STRAIGHT ----
    if r.node_type == NodeType.STRAIGHT:
        if my_done:
            return None
        if not r.children_ids:
            # leaf not yet done: guide forward (away from parent)
            if r.parent_id is not None:
                return safe_normalize(r.pos - robots[r.parent_id].pos)
            return None
        v = Vec2(0, 0)
        for cid in r.children_ids:
            v += robots[cid].pos - r.pos
        return safe_normalize(v) if v.length_squared() > 1e-12 else None

    # ---- PIVOT ----
    if my_done:
        return None
    if r.parent_id is None:
        return None

    parent     = robots[r.parent_id]
    incoming   = r.pos - parent.pos
    incoming_u = safe_normalize(incoming)
    if incoming_u.length_squared() < 1e-12:
        return None

    star_dir = Vec2(-incoming_u.y,  incoming_u.x)
    port_dir = Vec2( incoming_u.y, -incoming_u.x)

    has_star = has_port         = False
    star_done = port_done       = True

    for cid in r.children_ids:
        c      = robots[cid]
        sgn    = cross2(incoming_u, safe_normalize(c.pos - r.pos))
        if sgn > 0:
            has_star  = True
            if not subtree_done(c):
                star_done = False
        else:
            has_port  = True
            if not subtree_done(c):
                port_done = False

    star_needs = not has_star or not star_done
    port_needs = not has_port or not port_done

    if star_needs and port_needs:
        return star_dir
    if star_needs:
        return star_dir
    if port_needs:
        return port_dir
    return None


# =============================================================================
# Hop count (BFS from NETWORK → MOVING robots)
# =============================================================================




# =============================================================================
# Messaging
# =============================================================================

def deliver_messages(robots: List[Robot]):
    """Deliver outbox messages to all robots within COMM_RANGE."""
    senders = [(r.rid, r.pos, r.outbox) for r in robots if r.outbox]
    for r in robots:
        r.inbox.clear()

    for sender_id, spos, outbox in senders:
        for msg in outbox:
            for r in robots:
                if r.rid == sender_id:
                    continue
                if (r.pos - spos).length() <= COMM_RANGE:
                    r.inbox.append(msg)


# =============================================================================
# Hop count (BFS from NETWORK → MOVING robots)
# =============================================================================

def compute_hop_counts(robots: List[Robot]):
    """BFS from every NETWORK robot through COMM_RANGE neighbours."""
    for r in robots:
        r.hop_count = None

    queue: deque = deque()
    for r in robots:
        if r.role == Role.NETWORK:
            r.hop_count = 0
            queue.append(r.rid)

    while queue:
        rid     = queue.popleft()
        current = robots[rid]
        for other in robots:
            if other.hop_count is not None:
                continue
            if other.role != Role.MOVING:
                continue
            if (other.pos - current.pos).length() <= 2 * R:
                other.hop_count = current.hop_count + 1
                queue.append(other.rid)


# =============================================================================
# Sink state helpers
# =============================================================================

def node_sees_any_sink(pos: Vec2, sinks: List[Sink]) -> bool:
    return any((s.pos - pos).length() <= SINK_TOUCH_DIST for s in sinks)


def remove_sink_from_subtree(robots: List[Robot], root_id: int, sink_id: int):
    stack = [root_id]
    while stack:
        rid = stack.pop()
        r   = robots[rid]
        if sink_id in r.sink_indices:
            r.sink_indices.remove(sink_id)
        stack.extend(r.children_ids)


def release_branch_to_moving(robots: List[Robot], sinks: List[Sink], root_id: int):
    """
    Walk the entire subtree rooted at root_id and reset every robot to MOVING state.
    Collects all IDs first (before clearing children lists) to avoid skipping nodes.
    """
    to_release: List[int] = []
    stack = [root_id]
    while stack:
        rid = stack.pop()
        to_release.append(rid)
        stack.extend(robots[rid].children_ids)

    for rid in to_release:
        r               = robots[rid]
        r.role          = Role.MOVING
        r.node_type     = NodeType.STRAIGHT
        r.parent_id     = None
        r.children_ids  = []
        r.max_children  = 1
        r.sink_indices  = list(range(len(sinks)))
        r.desired_R     = None
        r.desired_dir_d = None
        r.void_pos_vis  = None
        r.sees_sink     = False
        r.pivot_cooldown = 0

    print(f"[RELEASE] Released subtree rooted at R{root_id}: {to_release}")


def tick_pivot_cooldowns(robots: List[Robot]):
    """
    Decrement cooldown on all pivots every step.
    When it reaches 0, restore max_children=2 so the pivot can recruit the freed side.
    """
    for r in robots:
        if r.node_type != NodeType.PIVOT or r.pivot_cooldown <= 0:
            continue
        r.pivot_cooldown -= 1
        if r.pivot_cooldown == 0:
            r.max_children = 2
            print(f"[COOLDOWN] Pivot R{r.rid} cooldown expired → max_children restored to 2")


def restore_sink_to_subtree(robots: List[Robot], root_id: int, sink_id: int):
    stack = [root_id]
    while stack:
        rid = stack.pop()
        r   = robots[rid]
        if sink_id not in r.sink_indices:
            r.sink_indices.append(sink_id)
        stack.extend(r.children_ids)


def compute_touched_sinks(robots: List[Robot], sinks: List[Sink]) -> List[bool]:
    """A sink counts as touched when a settled network robot is within SINK_TOUCH_DIST."""
    touched = [False] * len(sinks)
    for r in robots:
        if r.role != Role.NETWORK:
            continue
        if r.desired_dir_d is None or r.desired_R is None or r.parent_id is None:
            continue
        void_pos = robots[r.parent_id].pos + r.desired_dir_d * r.desired_R
        if (r.pos - void_pos).length() > SINK_TOUCH_SETTLEMENT_THRESH:
            continue
        for s in sinks:
            if (s.pos - r.pos).length() <= SINK_TOUCH_DIST:
                touched[s.sid] = True
    return touched


# =============================================================================
# Retargeting (used after pivot splits)
# =============================================================================

def intersect_list(a: List[int], b: List[int]) -> List[int]:
    sb = set(b)
    return [x for x in a if x in sb]


def retarget_subtree_intersection(
    robots:       List[Robot],
    sinks:        List[Sink],
    root_id:      int,
    allowed_sinks: List[int],
):
    """
    Restrict sink_indices of every node in the subtree to allowed_sinks.
    Also recomputes desired_dir_d and void_pos_vis for each node.
    """
    stack       = [root_id]
    allowed_set = set(allowed_sinks)

    while stack:
        rid = stack.pop()
        r   = robots[rid]

        if r.sink_indices:
            r.sink_indices = [s for s in r.sink_indices if s in allowed_set]

        if r.parent_id is not None and r.desired_R is not None:
            parent = robots[r.parent_id]
            ref    = (robots[parent.parent_id].pos
                      if parent.parent_id is not None
                      else parent.pos - Vec2(1, 0))
            d              = bisection_direction(parent.pos, ref, r.sink_indices, sinks)
            r.desired_dir_d = d
            r.void_pos_vis  = parent.pos + d * r.desired_R

        stack.extend(r.children_ids)


def assign_branch_id_subtree(robots: List[Robot], root_id: int, new_branch_id: str):
    """Apply branch_id to root and all descendants."""
    stack = [root_id]
    while stack:
        rid = stack.pop()
        r   = robots[rid]
        r.branch_id = new_branch_id
        stack.extend(r.children_ids)


# =============================================================================
# Pivot enforcement
# =============================================================================

def enforce_pivot_split(
    pivot_id: int,
    robots:   List[Robot],
    sinks:    List[Sink],
):
    """
    Promote robot pivot_id to PIVOT node type, split its sink list into
    star / port sides, assign existing children to their respective sides,
    release duplicates, and orient each child correctly.
    """
    p = robots[pivot_id]
    if p.node_type == NodeType.PIVOT:
        return

    p.node_type    = NodeType.PIVOT
    p.max_children = 2

    if not p.sink_indices or p.parent_id is None or len(p.sink_indices) < 2:
        return

    parent = robots[p.parent_id]
    star_sinks, port_sinks = split_sinks_star_port(p.pos, parent.pos, p.sink_indices, sinks)

    if not star_sinks or not port_sinks:
        return

    incoming   = p.pos - parent.pos
    incoming_u = safe_normalize(incoming)
    if incoming_u.length_squared() < 1e-12:
        return

    star_set = set(star_sinks)
    port_set = set(port_sinks)

    def touched_side(child: Robot) -> Optional[str]:
        for s in sinks:
            if (s.pos - child.pos).length() <= SINK_TOUCH_DIST:
                if s.sid in star_set:
                    return "STAR"
                if s.sid in port_set:
                    return "PORT"
        return None

    side_to_children: Dict[str, List[int]] = {"STAR": [], "PORT": []}
    for cid in p.children_ids:
        c    = robots[cid]
        side = touched_side(c)
        if side is None:
            sgn  = cross2(incoming_u, safe_normalize(c.pos - p.pos))
            side = "STAR" if sgn > 0 else "PORT"

        if side == "STAR":
            side_to_children["STAR"].append(cid)
            retarget_subtree_intersection(robots, sinks, cid, star_sinks)
            assign_branch_id_subtree(robots, cid, p.branch_id + "1")
        else:
            side_to_children["PORT"].append(cid)
            retarget_subtree_intersection(robots, sinks, cid, port_sinks)
            assign_branch_id_subtree(robots, cid, p.branch_id + "0")

    def release_extras(child_list: List[int]) -> List[int]:
        if len(child_list) <= 1:
            return child_list
        child_list.sort()
        keep = child_list[0]
        for cid in child_list[1:]:
            c               = robots[cid]
            c.parent_id     = None
            c.role          = Role.MOVING
            c.node_type     = NodeType.STRAIGHT
            c.desired_R     = None
            c.desired_dir_d = None
            c.void_pos_vis  = None
            c.children_ids.clear()
            c.max_children  = 1
            c.sink_indices  = list(range(len(sinks)))
            c.sees_sink     = False
            if cid in p.children_ids:
                p.children_ids.remove(cid)
            print(f"[PIVOT-RELEASE] pivot={pivot_id} released child={cid}, kept={keep}")
        return [keep]

    side_to_children["STAR"] = release_extras(side_to_children["STAR"])
    side_to_children["PORT"] = release_extras(side_to_children["PORT"])

    def orient_child(cid: int, side_sinks: List[int]):
        c               = robots[cid]
        c.parent_id     = pivot_id
        c.role          = Role.NETWORK
        c.node_type     = NodeType.STRAIGHT
        c.desired_R     = R
        dir_vec         = bisection_direction(p.pos, parent.pos, side_sinks, sinks)
        c.desired_dir_d = safe_normalize(dir_vec)
        c.void_pos_vis  = p.pos + c.desired_dir_d * c.desired_R
        c.max_children  = 1
        # Re-evaluate sees_sink: void position just changed
        c.sees_sink     = False

    for cid in side_to_children["STAR"]:
        orient_child(cid, star_sinks)
    for cid in side_to_children["PORT"]:
        orient_child(cid, port_sinks)


# =============================================================================
# Pivot metric & local candidacy
# =============================================================================

def calc_pivot_metric(r: Robot, robots: List[Robot], sinks: List[Sink]) -> Optional[float]:
    """
    120° criterion: the ideal pivot has each of its two branches pointing
    120° away from the incoming direction.  Returns the sum of absolute
    deviations from 120° for star and port sides.
    """
    if r.parent_id is None or len(r.sink_indices) < 2:
        return None

    parent = robots[r.parent_id]
    star, port = split_sinks_star_port(r.pos, parent.pos, r.sink_indices, sinks)
    if not star or not port:
        return None

    incoming_u = safe_normalize(parent.pos - r.pos)   # robot → parent
    if incoming_u.length_squared() < 1e-12:
        return None

    theta_star = angle_between_deg(incoming_u, avg_dir_to_sinks(r.pos, star, sinks))
    theta_port = angle_between_deg(incoming_u, avg_dir_to_sinks(r.pos, port, sinks))

    return abs(theta_star - 120.0) + abs(theta_port - 120.0)


def is_local_pivot_candidate(
    rid:        int,
    robots:     List[Robot],
    scores:     Dict[int, Optional[float]],
    pos_thresh: float = POS_THRESH,
) -> bool:
    """
    A robot is a valid local pivot candidate if:
      - NETWORK role, has a parent, and at least one child
      - settled at its own void position
      - at least one child is settled at its void position
      - score > parent score  (if parent is STRAIGHT)
      - score > all children and grandchildren scores
    """
    r = robots[rid]
    if r.role != Role.NETWORK or r.parent_id is None or not r.children_ids:
        return False

    if robots[r.parent_id].node_type == NodeType.PIVOT:
        return False

    my_score = scores.get(rid)
    if my_score is None:
        return False

    if r.desired_dir_d is None or r.desired_R is None:
        return False

    parent    = robots[r.parent_id]
    my_target = parent.pos + r.desired_dir_d * r.desired_R
    if (r.pos - my_target).length() > pos_thresh:
        return False

    any_child_settled = False
    for cid in r.children_ids:
        c = robots[cid]
        if c.desired_dir_d is None or c.desired_R is None:
            continue
        if (c.pos - (r.pos + c.desired_dir_d * c.desired_R)).length() <= pos_thresh:
            any_child_settled = True
            break
    if not any_child_settled:
        return False

    p_score = scores.get(r.parent_id)
    if robots[r.parent_id].node_type == NodeType.STRAIGHT:
        if p_score is not None and my_score <= p_score:
            return False

    if r.children_ids:
        c_score = scores.get(r.children_ids[0])
        if c_score is not None and my_score <= c_score:
            return False

    return True


# =============================================================================
# Sink placement
# =============================================================================

def make_demo_sinks(source_pos: Vec2) -> List[Sink]:
    """Place NUM_SINKS sinks randomly in a rectangle above the source."""
    sinks: List[Sink] = []
    min_sep      = SINK_MIN_SEP * COMM_RANGE
    max_attempts = 1000

    x_min = source_pos.x - SINK_SPACE_WIDTH  / 2
    x_max = source_pos.x + SINK_SPACE_WIDTH  / 2
    y_min = source_pos.y - SINK_SPACE_HEIGHT
    y_max = source_pos.y - 150
    min_from_src = 150

    for i in range(NUM_SINKS):
        best_candidate: Optional[Vec2] = None
        best_min_dist  = -1.0
        placed         = False

        for _ in range(max_attempts):
            pos = Vec2(random.uniform(x_min, x_max), random.uniform(y_min, y_max))
            if (pos - source_pos).length() < min_from_src:
                continue

            candidate_min = (min((pos - s.pos).length() for s in sinks)
                             if sinks else float("inf"))

            if candidate_min >= min_sep:
                sinks.append(Sink(i, pos))
                placed = True
                break

            if candidate_min > best_min_dist:
                best_min_dist = candidate_min
                best_candidate = pos

        if not placed:
            if best_candidate is None:
                best_candidate = Vec2(random.uniform(x_min, x_max),
                                      random.uniform(y_min, y_max))
            sinks.append(Sink(i, best_candidate))
            print(f"[SINK] Warning: sink {i} placed with min_dist={best_min_dist:.1f} "
                  f"(< {min_sep:.1f})")

    return sinks
