"""
Assistive Functions for Multi-Sink Network Formation Simulation
Core Version - Geometry, void controller, boids, pivot logic, guidance, messaging.
No additive-leg (AL) code.
"""
from __future__ import annotations

import math
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
    RECRUIT_SETTLE_THRESH, MAX_TURN_DEG,
    W_BATTERY, W_PIVOT, W_HYST, TRAITOR_GRACE_TICKS, SPLICE_COOLDOWN_TICKS,
    CANDIDATE_SWITCH_COOLDOWN_TICKS, OLD_SINK_REVISIT_COOLDOWN_TICKS,
    OLD_PARENT_REVISIT_COOLDOWN_TICKS, TRAITOR_ELIGIBLE_CHARGE_THRESHOLD,
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


def pivot_single_sink(y: float) -> float:
    """
    Ideal branching angle (adb, degrees) at the pivot given y = half the
    angle (degrees) between source→sink_A and source→sink_B.

    Geometry: O=source, A=pivot, B and C are the two sinks.
    The 120° equilateral-link assumption gives the chain lengths via law of sines.
    """
    OA = math.sin(math.radians(120)) / math.sin(math.radians(60 - y))
    AD = math.sqrt(OA**2 + 1 - 2 * OA * math.cos(math.radians(2 * y)))
    cos_val = max(-1.0, min(1.0, (AD**2 + 1 - OA**2) / (2 * AD)))
    ADO = math.degrees(math.acos(cos_val))
    return 180 - ADO


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
    elif robot.role == Role.TRAITOR:
        _traitor_behavior(robot, dt, robots, sinks, touched)
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

def _try_accept_network_offer(robot: Robot, robots: List[Robot]) -> bool:
    """
    Accept a network attachment offer (ACCEPT_CHILD addressed to this robot).
    Shared by free movers and traitors. Returns True if it consumed the tick.
    """
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
            robot.old_parent_id      = None
            robot.old_parent_cooldown = 0
            robot.old_sink_id        = None
            robot.old_sink_cooldown  = 0
            robot.traitor_grace      = 0
            robot.candidate_cooldown = 0

            parent_type = parent_robot.node_type
            gp_type     = (robots[parent_robot.parent_id].node_type
                           if parent_robot.parent_id is not None else "NONE")
            print(f"[NETWORK] Robot {robot.rid} attached to parent {parent_id} "
                  f"(branch={robot.branch_id}, parent_type={parent_type}, "
                  f"gp_type={gp_type}, sinks={robot.sink_indices[:8]})")
            return True
    return False


def _respond_to_recruitment(robot: Robot, dt: float, robots: List[Robot]) -> bool:
    """
    Check for recruitment broadcasts (ASK_PARENT) and move toward the nearest
    valid asker. Shared by free movers and traitors. Returns True if it
    consumed the tick.
    """
    valid_askers: List[int] = []
    for msg in robot.inbox:
        if msg.msg_type != "ASK_PARENT":
            continue

        asker_id = msg.sender_id
        if asker_id == robot.old_parent_id:
            # Never immediately re-accept the slot we just defected from.
            continue
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

    if not valid_askers:
        return False

    # Recruitment overrides guidance lock
    robot.guidance_lock_robot_id = None
    robot.guidance_lock_frames   = 0

    best_id = min(valid_askers,
                  key=lambda aid: (robots[aid].pos - robot.pos).length())
    robot.broadcast("PARENT_STATUS", has_parent=False, responding_to=best_id)
    _simple_move_towards(robot, robots[best_id].pos, dt)
    return True


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
    if _try_accept_network_offer(robot, robots):
        return

    # ---- Step 1: Check for recruitment messages ----
    if _respond_to_recruitment(robot, dt, robots):
        return

    # ---- Step 2: Follow guidance ----
    locked_guidance:    Optional[Vec2]  = None
    locked_guide_robot: Optional[Robot] = None

    # Try to reuse the existing lock
    if robot.guidance_lock_robot_id is not None and robot.guidance_lock_frames > 0:
        gid = robot.guidance_lock_robot_id
        if 0 <= gid < len(robots):
            guide = robots[gid]
            if guide.role == Role.NETWORK:
                g = guidance_dir_from_robot(guide, robots, touched, sinks)
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

            g = guidance_dir_from_robot(other, robots, touched, sinks)
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
# TRAITOR behaviour
# =============================================================================

def _traitor_behavior(
    robot:   Robot,
    dt:      float,
    robots:  List[Robot],
    sinks:   List[Sink],
    touched: List[bool],
):
    """
    Traitor: a former NETWORK robot in transit toward a new candidate sink
    (held in sink_indices[0]). Acts exactly like a free mover — same
    reattachment, same recruitment-response, same guide selection — with
    exactly one difference: if the chosen guide's sink_indices doesn't
    contain the candidate, follow the reverse direction instead of forward
    (so it doesn't retrace into the sink it already left). Also re-picks the
    candidate sink if the current one becomes touched by someone else first.
    """
    # ---- Reattachment / recruitment-response: identical to a free mover,
    #      except during the post-defection grace window (see constants.py) ----
    if robot.traitor_grace > 0:
        robot.traitor_grace -= 1
    else:
        if _try_accept_network_offer(robot, robots):
            return
        if _respond_to_recruitment(robot, dt, robots):
            return

    # ---- (a) Re-pick candidate sink if the current one is no longer valid ----
    if robot.candidate_cooldown > 0:
        robot.candidate_cooldown -= 1

    if robot.old_sink_cooldown > 0:
        robot.old_sink_cooldown -= 1
        if robot.old_sink_cooldown == 0:
            robot.old_sink_id = None

    if robot.old_parent_cooldown > 0:
        robot.old_parent_cooldown -= 1
        if robot.old_parent_cooldown == 0:
            robot.old_parent_id = None

    target_sid = robot.sink_indices[0] if robot.sink_indices else None
    if target_sid is None or (touched[target_sid] and robot.candidate_cooldown <= 0):
        guide_for_pivot: Optional[Robot] = None
        gid = robot.guidance_lock_robot_id
        if gid is not None and 0 <= gid < len(robots):
            g = robots[gid]
            if g.role == Role.NETWORK:
                guide_for_pivot = g

        best_sid:   Optional[int] = None
        best_score: float         = float("-inf")
        for s in sinks:
            if touched[s.sid] or s.sid == robot.old_sink_id:
                continue
            score = W_BATTERY * (100.0 - s.charging_level) / 100.0
            if guide_for_pivot is not None:
                pd = pivot_distance(guide_for_pivot, robots, s.sid)
                score += W_PIVOT / (1.0 + pd)
            score -= W_HYST
            if score > best_score:
                best_score, best_sid = score, s.sid

        if best_sid is not None:
            robot.sink_indices      = [best_sid]
            target_sid              = best_sid
            robot.candidate_cooldown = CANDIDATE_SWITCH_COOLDOWN_TICKS
        # else: no untouched sink remains anywhere (network complete) — keep
        # the stale target; harmless, there's nothing left to chase.

    # ---- (b)+(c) Guidance-following: identical to _free_behavior, with one
    #      twist — reverse direction if the guide doesn't have our candidate ----
    locked_guidance:    Optional[Vec2]  = None
    locked_guide_robot: Optional[Robot] = None

    if robot.guidance_lock_robot_id is not None and robot.guidance_lock_frames > 0:
        gid = robot.guidance_lock_robot_id
        if 0 <= gid < len(robots):
            guide = robots[gid]
            if guide.role == Role.NETWORK:
                g = guidance_dir_from_robot(guide, robots, touched, sinks, target_sid)
                if g is None:
                    g = guide.pos - robot.pos
                if g.length_squared() > 1e-12:
                    locked_guidance    = g
                    locked_guide_robot = guide

    if locked_guidance is None:
        robot.guidance_lock_robot_id = None
        robot.guidance_lock_frames   = 0

        best_guidance:    Optional[Vec2]  = None
        best_relevant:    bool            = False
        best_score_g:     int             = -1
        best_dist:        float           = float("inf")
        best_guide_robot: Optional[Robot] = None

        for other in robots:
            if other.rid == robot.rid or other.role != Role.NETWORK:
                continue
            dist = (other.pos - robot.pos).length()
            if dist > COMM_RANGE:
                continue

            g = guidance_dir_from_robot(other, robots, touched, sinks, target_sid)
            if g is None:
                continue

            # Among several guides in range, one whose sink_indices actually
            # contains our candidate always outranks one that doesn't —
            # checked before the generic branch-depth/distance tie-break.
            relevant   = target_sid is not None and target_sid in other.sink_indices
            branch_val = int(other.branch_id, 2) if other.branch_id else 0

            if (relevant, branch_val, -dist) > (best_relevant, best_score_g, -best_dist):
                best_relevant    = relevant
                best_score_g     = branch_val
                best_dist        = dist
                best_guidance    = g
                best_guide_robot = other

        if best_guidance is not None:
            robot.guidance_lock_robot_id = best_guide_robot.rid
            robot.guidance_lock_frames   = GUIDANCE_STICK_FRAMES
            locked_guidance              = best_guidance
            locked_guide_robot           = best_guide_robot

    if locked_guidance is not None:
        guidance_dir = safe_normalize(locked_guidance)
        guide        = locked_guide_robot
        irrelevant   = target_sid is not None and target_sid not in guide.sink_indices
        guide_done   = all(touched[sid] for sid in guide.sink_indices)

        if irrelevant and not guide_done:
            # Guide doesn't have our candidate AND still has its own
            # unfinished work — retreat continuously, relative to our OWN
            # position. Anchoring at the guide (like the forward case does)
            # would just create a fixed nearby point to hover around forever
            # instead of actually moving away. A guide whose own branch is
            # already fully done poses no such risk, so it's followed
            # forward like normal even without the candidate.
            goal_target = robot.pos - guidance_dir * R
        else:
            guide_R     = guide.desired_R if guide.desired_R is not None else R
            goal_target = guide.pos + guidance_dir * guide_R
        speed = max(guide.vel.length(), SPEED_MAX)

        if robot.guidance_lock_frames > 0:
            robot.guidance_lock_frames -= 1
    else:
        # No guidance: drift back toward the source, exactly like a mover.
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

    # ---- 3. Recruitment ----
    _try_recruit(robot, robots, sinks, touched)

    # ---- 4. Gradually steer toward current target sink ----
    _steer_toward_sink(robot, robots, sinks)



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

    if only_sink_seen:
        # Cascade tail-prune: if parent already sees the same sink, this robot
        # has overshot — release it so the chain shortens one robot per step.
        if robot.parent_id is not None and robots[robot.parent_id].sees_sink:
            parent = robots[robot.parent_id]
            if robot.rid in parent.children_ids:
                parent.children_ids.remove(robot.rid)
            release_branch_to_moving(robots, sinks, robot.rid)
            print(f"[CASCADE] R{robot.rid} pruned — parent R{parent.rid} also touches sink")
            return

        if not was_seeing:
            robot.sees_sink             = True
            robot.ticks_since_sink_touch = 0
            robot.max_children = 0
            tail = list(robot.children_ids)
            for cid in tail:
                release_branch_to_moving(robots, sinks, cid)
            robot.children_ids.clear()
            print(f"[SEES_SINK] Robot {robot.rid} sees sink {seen_sinks[0]} "
                  f"→ capacity=0, released {len(tail)} tail robot(s)")
        else:
            robot.ticks_since_sink_touch += 1

    elif was_seeing and not only_sink_seen:
        robot.sees_sink = False
        if robot.pivot_cooldown <= 0:
            default_cap        = 2 if robot.node_type == NodeType.PIVOT else 1
            robot.max_children = default_cap
            print(f"[SEES_SINK] Robot {robot.rid} no longer sees only sink → capacity={default_cap}")
        # else: still cooling down from a recent splice (see convert_to_traitor)
        # — stay at capacity=0 until tick_pivot_cooldowns restores it, so the
        # just-shortened chain doesn't instantly re-recruit a replacement.



def _steer_toward_sink(robot: Robot, robots: List[Robot], sinks: List[Sink]):
    """
    Every step: rotate desired_dir_d by at most MAX_TURN_DEG toward the
    direction of the current target sink. The void controller then moves
    the robot smoothly without snapping.
    """
    if (robot.desired_dir_d is None or robot.parent_id is None
            or not robot.sink_indices):
        return

    parent     = robots[robot.parent_id]
    target_dir = safe_normalize(
        avg_dir_to_sinks(parent.pos, robot.sink_indices, sinks)
    )
    if target_dir.length_squared() < 1e-12:
        return

    current = robot.desired_dir_d
    angle   = angle_between_deg(current, target_dir)
    if angle < 0.01:
        return

    t               = min(1.0, MAX_TURN_DEG / angle)
    robot.desired_dir_d = safe_normalize(current * (1.0 - t) + target_dir * t)
    if robot.desired_R is not None:
        robot.void_pos_vis = parent.pos + robot.desired_dir_d * robot.desired_R


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
            child_sinks     = robot.sink_indices[:]
            child_branch_id = robot.branch_id
            d               = bisection_direction(robot.pos, ref_parent_pos, child_sinks, sinks)

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
    r:            Robot,
    robots:       List[Robot],
    touched:      Optional[List[bool]],
    sinks:        List[Sink],
    candidate_sid: Optional[int] = None,
) -> Optional[Vec2]:
    """
    Guidance direction emitted by a NETWORK robot toward movers.

    STRAIGHT: guide toward child(ren) if subtree is not done.
    PIVOT:    guide toward the unfinished branch(es) — unless the asker has a
              specific candidate_sid (a Traitor), in which case guide
              straight to whichever side actually leads to that sink,
              ignoring the generic needs/charge-bias priority.
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

    if candidate_sid is not None:
        star_sinks, port_sinks = split_sinks_star_port(r.pos, parent.pos, r.sink_indices, sinks)
        if candidate_sid in star_sinks:
            return star_dir
        if candidate_sid in port_sinks:
            return port_dir
        # candidate not under this pivot at all — fall through to generic logic

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
        # Both sides still need growth: steer toward whichever side's sinks
        # are, in total, more charge-starved (Dutch-auction style bias).
        star_sinks, port_sinks = split_sinks_star_port(r.pos, parent.pos, r.sink_indices, sinks)
        star_sum = (sum(sinks[sid].charging_level for sid in star_sinks)
                    if star_sinks else float("inf"))
        port_sum = (sum(sinks[sid].charging_level for sid in port_sinks)
                    if port_sinks else float("inf"))
        return port_dir if port_sum < star_sum else star_dir
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
        r.ticks_since_sink_touch = 0
        r.pivot_cooldown = 0
        r.old_parent_id      = None
        r.old_parent_cooldown = 0
        r.old_sink_id        = None
        r.old_sink_cooldown  = 0
        r.traitor_grace      = 0
        r.candidate_cooldown = 0

    print(f"[RELEASE] Released subtree rooted at R{root_id}: {to_release}")


def tick_pivot_cooldowns(robots: List[Robot]):
    """
    Decrement cooldown on any network robot that has one.
    When it reaches 0, restore max_children based on node_type AND how many
    sinks are currently assigned — a pivot with only 1 sink stays at 1 child.
    """
    for r in robots:
        if r.role != Role.NETWORK or r.pivot_cooldown <= 0:
            continue
        r.pivot_cooldown -= 1
        if r.pivot_cooldown == 0 and not r.sees_sink:
            # Don't clobber a robot that's actively sees_sink (capacity=0 is
            # its own invariant, not ours to override) — only restore for a
            # robot that genuinely isn't touching anything right now.
            is_pivot_with_two_sinks = (r.node_type == NodeType.PIVOT
                                       and len(r.sink_indices) >= 2)
            r.max_children = 2 if is_pivot_with_two_sinks else 1
            print(f"[COOLDOWN] R{r.rid} ({r.node_type}) cooldown expired "
                  f"→ max_children={r.max_children} (sinks={r.sink_indices})")


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

def calc_pivot_metric(
    r: Robot,
    robots: List[Robot],
    sinks: List[Sink],
    target_angle: float = 120.0,
) -> Optional[float]:
    """
    Branching-angle criterion: returns the sum of absolute deviations of the
    star and port branch angles from target_angle.
    target_angle defaults to 120° (equilateral); in phase 2 it is set to the
    geometry-derived adb from pivot_single_sink().
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

    return abs(theta_star - target_angle) + abs(theta_port - target_angle)


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
# Traitor metrics & eligibility
# =============================================================================

def hop_count_to_own_sink(r: Robot, robots: List[Robot]) -> Optional[int]:
    """
    Walk down r's descendant chain (children_ids[0] — chains are single-child
    before a pivot split) counting steps until a sees_sink==True node is
    found. Returns that depth (0 if r itself is the sink-toucher). Returns
    None if a childless, non-sees_sink leaf is reached first (a dead-end that
    never reached its sink — deliberately scored worse than a genuine
    sink-sitter, not the same as hop_count=0).
    """
    depth = 0
    node  = r
    while True:
        if node.sees_sink:
            return depth
        if not node.children_ids:
            return None
        node   = robots[node.children_ids[0]]
        depth += 1


def chain_toucher_dwell_ticks(r: Robot, robots: List[Robot]) -> Optional[int]:
    """
    Same downward walk as hop_count_to_own_sink, but returns the sink-toucher's
    ticks_since_sink_touch instead of the hop depth. Any robot along an
    already-touched chain shares the SAME dwell-based hysteresis as the
    literal toucher — pulling a mid-chain robot shortens the chain by one R
    step exactly like pulling the leaf would, so it carries the identical
    risk of un-touching the sink and needs the identical protection.
    """
    node = r
    while True:
        if node.sees_sink:
            return node.ticks_since_sink_touch
        if not node.children_ids:
            return None
        node = robots[node.children_ids[0]]


def pivot_distance(r: Robot, robots: List[Robot], sid: int) -> int:
    """
    Walk up from r through parent_id, counting NodeType.PIVOT ancestors
    crossed, until reaching an ancestor whose sink_indices contains sid.
    Always terminates: the SOURCE robot's sink_indices holds every sink.
    """
    count = 0
    node  = r
    while node.parent_id is not None:
        node = robots[node.parent_id]
        if sid in node.sink_indices:
            return count
        if node.node_type == NodeType.PIVOT:
            count += 1
    return count


def is_traitor_eligible(
    r:          Robot,
    robots:     List[Robot],
    touched:    List[bool],
    sinks:      List[Sink],
    pos_thresh: float = RECRUIT_SETTLE_THRESH,
) -> bool:
    """
    A robot can become a Traitor-candidate once its branch's job is done:
    a settled STRAIGHT NETWORK robot carrying exactly one sink that is either
    currently touched OR already charged to an acceptable level (PIVOTs
    excluded — structurally load-bearing). The charge-level fallback matters
    under robot scarcity: if the chain's own tip is itself the one that
    defects, touched[] flips False the instant it leaves (compute_touched_sinks
    only scans NETWORK robots), which would otherwise permanently strand the
    rest of the chain — never touched live again, never eligible again, with
    no spare robot around to refill the tip. Not leaf-only: any robot along
    an eligible chain qualifies — hop_count_to_own_sink differentiates which
    one is cheapest to pull (closest to the sink first) when the chain is
    still actually connected; convert_to_traitor splices the gap when a
    mid-chain robot leaves.
    """
    if r.role != Role.NETWORK or r.node_type != NodeType.STRAIGHT:
        return False
    if r.parent_id is None:
        return False
    if len(r.sink_indices) != 1:
        return False
    sid = r.sink_indices[0]
    if not touched[sid] and sinks[sid].charging_level < TRAITOR_ELIGIBLE_CHARGE_THRESHOLD:
        return False
    if r.desired_dir_d is None or r.desired_R is None:
        return False

    parent   = robots[r.parent_id]
    void_pos = parent.pos + r.desired_dir_d * r.desired_R
    return (r.pos - void_pos).length() <= pos_thresh


def convert_to_traitor(robot: Robot, target_sid: int, robots: List[Robot], sinks: List[Sink]):
    """
    Detach robot from its parent and turn it into a Traitor chasing
    target_sid. max_children is reset to 1 because a sink-touching candidate
    has max_children=0, and ACCEPT_CHILD never restores it on reattachment —
    skipping this would permanently cripple the robot once it rejoins.

    If robot had a child (it was a mid-chain link, not the leaf), the gap is
    spliced: the child is reattached directly to robot's old parent, with its
    desired_dir_d/void_pos recomputed relative to that new parent — chains are
    single-child here, so this one splice is always sufficient (deeper
    descendants' own desired_* stay relative to the unmoved child, untouched).
    """
    old_parent_id = robot.parent_id
    child_id       = robot.children_ids[0] if robot.children_ids else None

    if old_parent_id is not None:
        parent = robots[old_parent_id]
        if robot.rid in parent.children_ids:
            parent.children_ids.remove(robot.rid)

        if child_id is not None:
            child           = robots[child_id]
            child.parent_id = old_parent_id
            parent.children_ids.append(child_id)

            grandparent_ref = (robots[parent.parent_id].pos
                               if parent.parent_id is not None
                               else parent.pos - Vec2(1, 0))
            child.desired_R     = R
            child.desired_dir_d = safe_normalize(
                bisection_direction(parent.pos, grandparent_ref, child.sink_indices, sinks)
            )
            child.void_pos_vis   = parent.pos + child.desired_dir_d * child.desired_R
            child.max_children   = 0
            child.pivot_cooldown = SPLICE_COOLDOWN_TICKS

    robot.parent_id              = None
    robot.old_parent_id          = old_parent_id
    robot.old_parent_cooldown    = OLD_PARENT_REVISIT_COOLDOWN_TICKS
    robot.old_sink_id            = robot.sink_indices[0] if robot.sink_indices else None
    robot.old_sink_cooldown      = OLD_SINK_REVISIT_COOLDOWN_TICKS
    robot.role                   = Role.TRAITOR
    robot.node_type              = NodeType.STRAIGHT
    robot.children_ids           = []
    robot.sink_indices           = [target_sid]
    robot.sees_sink              = False
    robot.max_children           = 1
    robot.desired_R              = None
    robot.desired_dir_d          = None
    robot.void_pos_vis           = None
    robot.guidance_lock_robot_id = None
    robot.guidance_lock_frames   = 0
    robot.traitor_grace          = TRAITOR_GRACE_TICKS
    robot.outbox.clear()

    print(f"[TRAITOR] Robot {robot.rid} defected, targeting sink {target_sid}")


# =============================================================================
# Sink placement
# =============================================================================

def make_demo_sinks(source_pos: Vec2) -> List[Sink]:
    """Place NUM_SINKS sinks randomly in a rectangle above the source."""
    import random
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
                sinks.append(Sink(i, pos, charging_level=0.0))
                placed = True
                break

            if candidate_min > best_min_dist:
                best_min_dist = candidate_min
                best_candidate = pos

        if not placed:
            if best_candidate is None:
                best_candidate = Vec2(random.uniform(x_min, x_max),
                                      random.uniform(y_min, y_max))
            sinks.append(Sink(i, best_candidate, charging_level=0.0))
            print(f"[SINK] Warning: sink {i} placed with min_dist={best_min_dist:.1f} "
                  f"(< {min_sep:.1f})")

    return sinks
