"""
Assistive Functions for Multi-Sink Network Formation Simulation
Paper Version - Geometry helpers, pivot logic, guidance, messaging, and sink handling
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

from pygame.math import Vector2 as Vec2

from constants_paper_version import (
    Robot, Sink, Role, NodeType,
    R, COMM_RANGE, SPEED_MAX, SINK_TOUCH_DIST,
    K_R, K_THETA, POS_THRESH,
    SINK_TOUCH_SETTLEMENT_THRESH, GUIDANCE_STICK_FRAMES,
    NUM_SINKS, SINK_SPACE_WIDTH, SINK_SPACE_HEIGHT, SINK_MIN_SEP,
    BOID_NEIGHBOR_RADIUS, BOID_SEP_RADIUS,
    BOID_SEP_WEIGHT, BOID_COH_WEIGHT, BOID_ALIGN_WEIGHT,
    RECRUIT_SETTLE_THRESH,
)


# =========================
# Geometry Helpers
# =========================
def safe_normalize(v: Vec2) -> Vec2:
    """Safely normalize a vector, returning zero vector if too small."""
    if v.length_squared() < 1e-12:
        return Vec2(0, 0)
    return v.normalize()


def angle_between_deg(a: Vec2, b: Vec2) -> float:
    """Calculate angle between two vectors in degrees."""
    if a.length_squared() < 1e-12 or b.length_squared() < 1e-12:
        return 0.0
    au = a.normalize()
    bu = b.normalize()
    dot = max(-1.0, min(1.0, au.x * bu.x + au.y * bu.y))
    return math.degrees(math.acos(dot))


def cross2(a: Vec2, b: Vec2) -> float:
    """Calculate 2D cross product (z-component of 3D cross product)."""
    return a.x * b.y - a.y * b.x


def split_sinks_star_port(
    node_pos: Vec2,
    parent_pos: Vec2,
    sink_ids: List[int],
    sinks: List[Sink]
) -> Tuple[List[int], List[int]]:
    """
    Split sink_ids into STARBOARD / PORT relative to the incoming direction parent->node.
    STARBOARD = cross(incoming, node->sink) > 0
    PORT      = cross(incoming, node->sink) < 0
    """
    incoming = node_pos - parent_pos  # parent->node
    incoming_u = safe_normalize(incoming)
    if incoming_u.length_squared() < 1e-12:
        # degenerate; dump all to one side
        return sink_ids[:], []

    star: List[int] = []
    port: List[int] = []
    for sid in sink_ids:
        v = sinks[sid].pos - node_pos
        vu = safe_normalize(v)
        s = cross2(incoming_u, vu)
        if s > 0:
            star.append(sid)
        elif s < 0:
            port.append(sid)
        else:
            # exactly on line -> assign to smaller side
            if len(star) <= len(port):
                star.append(sid)
            else:
                port.append(sid)
    return star, port


def avg_dir_to_sinks(node_pos: Vec2, sink_ids: List[int], sinks: List[Sink]) -> Vec2:
    """Calculate average direction from node to a set of sinks."""
    s = Vec2(0, 0)
    for sid in sink_ids:
        d = sinks[sid].pos - node_pos
        if d.length_squared() > 1e-12:
            s += d.normalize()
    if s.length_squared() < 1e-12:
        return Vec2(1, 0)
    return s.normalize()


def bisection_direction(
    node_pos: Vec2,
    parent_pos: Vec2,
    sink_ids: List[int],
    sinks: List[Sink]
) -> Vec2:
    """
    Calculate bisection direction for a robot.
    
    Split sinks into star/port relative to parent->node,
    compute average vector for each side, then return bisector.
    Fallback: if one side empty => avg of all sinks.
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


# =========================
# Robot Behaviors (step methods)
# =========================
def robot_step(robot: Robot, dt: float, robots: List[Robot], sinks: List[Sink], touched: List[bool]):
    """
    Main step function for a robot - dispatches to appropriate behavior.
    Records previous position and updates velocity.
    """
    # record previous position to compute instantaneous velocity after moving
    robot.prev_pos = Vec2(robot.pos)

    if robot.role == Role.SOURCE:
        _source_behavior(robot, sinks)
    elif robot.parent_id is None:
        _free_behavior(robot, dt, robots, sinks, touched)
    else:
        _attached_behavior(robot, dt, robots, sinks, touched)
        # AL child recruitment runs after the standard void controller has settled AL
        if robot.is_additive_leg:
            _try_recruit_al_child(robot, robots)

    # update velocity estimate
    if dt > 0:
        robot.vel = (robot.pos - robot.prev_pos) / dt
    else:
        robot.vel = Vec2(0, 0)


def _source_behavior(robot: Robot, sinks: List[Sink]):
    """Behavior for SOURCE robot - recruits children toward sinks."""
    if robot.has_free_slot():
        robot.broadcast("ASK_PARENT")

    for msg in robot.inbox:
        if msg.msg_type == "PARENT_STATUS" and robot.has_free_slot():
            if msg.payload.get("has_parent", True):
                continue

            # Only accept if the free robot chose US as its preferred parent
            if msg.payload.get("responding_to") != robot.rid:
                continue

            child_id = msg.sender_id
            robot.children_ids.append(child_id)

            # Direction for child: average towards all sinks (source has no parent)
            d = avg_dir_to_sinks(robot.pos, robot.sink_indices, sinks)
            child_sinks = robot.sink_indices[:]
            
            # Assign branch ID to child (source's child stays on branch "1")
            child_branch_id = "1"

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


def _free_behavior(robot: Robot, dt: float, robots: List[Robot], sinks: List[Sink], touched: List[bool]):
    """Behavior for free MOVING robot - follows guidance and responds to recruitment."""
    # Priority 1: Check for nearby NETWORK robots with free slots asking for children
    valid_askers = []
    for msg in robot.inbox:
        if msg.msg_type != "ASK_PARENT":
            continue
        
        asker_id = msg.sender_id
        asker = robots[asker_id]
        
        # Check if this is a pivot asking for specific side
        payload = msg.payload
        if "need_star" in payload and "need_port" in payload:
            # Pivot is asking for specific side(s)
            need_star = payload["need_star"]
            need_port = payload["need_port"]
            incoming_u = Vec2(payload["incoming_x"], payload["incoming_y"])
            
            # Determine which side we're on relative to the pivot
            v_pc = robot.pos - asker.pos
            v_pc_u = safe_normalize(v_pc)
            side_sgn = cross2(incoming_u, v_pc_u)
            
            my_side_is_star = side_sgn > 0
            
            # Only respond if pivot needs children on our side
            if (my_side_is_star and need_star) or (not my_side_is_star and need_port):
                valid_askers.append(asker_id)
        else:
            # Non-pivot recruiter, always respond
            valid_askers.append(asker_id)

    if valid_askers:
        # Someone asked for children - recruitment overrides any guidance lock
        robot.guidance_lock_robot_id = None
        robot.guidance_lock_frames = 0

        best_id = min(valid_askers, key=lambda aid: (robots[aid].pos - robot.pos).length())
        # Address the response to the chosen recruiter only
        robot.broadcast("PARENT_STATUS", has_parent=False, responding_to=best_id)
        _simple_move_towards(robot, robots[best_id].pos, dt)
    else:
        # Follow guidance with stickiness
        locked_guidance = None
        locked_guide_robot: Optional[Robot] = None

        if robot.guidance_lock_robot_id is not None and robot.guidance_lock_frames > 0:
            if 0 <= robot.guidance_lock_robot_id < len(robots):
                guide = robots[robot.guidance_lock_robot_id]
                if guide.role == Role.NETWORK:
                    g = guidance_dir_from_robot(guide, robots, touched)
                    if g is None:
                        g = guide.pos - robot.pos
                    if g.length_squared() > 1e-12:
                        locked_guidance = g
                        locked_guide_robot = guide

        if locked_guidance is None:
            robot.guidance_lock_robot_id = None
            robot.guidance_lock_frames = 0

            best_guidance = None
            best_score = -1
            best_dist = float("inf")
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
                    best_score = branch_val
                    best_dist = dist
                    best_guidance = g
                    best_guide_robot = other

            if best_guidance is not None:
                robot.guidance_lock_robot_id = best_guide_robot.rid if best_guide_robot is not None else None
                robot.guidance_lock_frames = GUIDANCE_STICK_FRAMES
                locked_guidance = best_guidance
                locked_guide_robot = best_guide_robot

        if locked_guidance is not None:
            guidance_dir = safe_normalize(locked_guidance)

            if locked_guide_robot is not None:
                guide_R = locked_guide_robot.desired_R if locked_guide_robot.desired_R is not None else R
                goal_target = locked_guide_robot.pos + guidance_dir * guide_R
                speed = max(locked_guide_robot.vel.length(), SPEED_MAX)
            else:
                goal_target = robot.pos + guidance_dir * R
                speed = SPEED_MAX

            if robot.guidance_lock_frames > 0:
                robot.guidance_lock_frames -= 1
        else:
            # No guidance — goal is to drift back toward the source
            goal_target = robots[0].pos
            speed = SPEED_MAX

        # Boid-steered movement: separation + cohesion + goal-heading (modified alignment)
        move_dir = _boid_move_direction(robot, robots, goal_target)
        _move_towards_with_speed(robot, robot.pos + move_dir * R, speed, dt)

    # Process ACCEPT_AL_CHILD
    for msg in robot.inbox:
        if msg.msg_type == "ACCEPT_AL_CHILD" and msg.payload.get("child_id") == robot.rid:
            al_parent_id = msg.payload["parent_id"]
            al_parent = robots[al_parent_id]
            sink_pos = sinks[msg.payload["sink_id"]].pos
            robot.is_al_child = True
            robot.al_child_parent_id = al_parent_id
            robot.parent_id = al_parent_id          # standard void controller needs this
            robot.role = Role.NETWORK
            robot.max_children = 0
            robot.desired_R = float(msg.payload["R"])
            robot.desired_dir_d = safe_normalize(sink_pos - al_parent.pos)
            robot.sink_indices = [msg.payload["sink_id"]]
            robot.branch_id = ""
            al_parent.al_child_id = robot.rid
            al_parent.max_children += 1
            if robot.rid not in al_parent.children_ids:
                al_parent.children_ids.append(robot.rid)
            print(f"[AL CHILD] Robot {robot.rid} child of AL {al_parent_id} toward sink {msg.payload['sink_id']}")

    # Process ACCEPT_ADDITIVE_LEG
    for msg in robot.inbox:
        if msg.msg_type == "ACCEPT_ADDITIVE_LEG" and msg.payload.get("child_id") == robot.rid:
            parent_id = msg.payload["parent_id"]
            robot.parent_id = parent_id
            robot.role = Role.NETWORK
            robot.is_additive_leg = True
            robot.max_children = 0
            robot.desired_R = float(msg.payload["R"])
            robot.desired_dir_d = Vec2(msg.payload["d_x"], msg.payload["d_y"])
            robot.sink_indices = [msg.payload["sink_id"]]
            robot.branch_id = ""
            robots[parent_id].max_children += 1
            if robot.rid not in robots[parent_id].children_ids:
                robots[parent_id].children_ids.append(robot.rid)
            # Immediately remove the covered sink from N1's normal children and their subtrees
            # so they stop pursuing it — the AL is now responsible for it.
            sid = msg.payload["sink_id"]
            n1 = robots[parent_id]
            for cid in n1.children_ids:
                if cid == robot.rid:  # skip the AL itself
                    continue
                remove_sink_from_subtree(robots, cid, sid)
            # Also remove from N1 itself
            if sid in n1.sink_indices:
                n1.sink_indices.remove(sid)
            print(f"[ADDITIVE LEG] Robot {robot.rid} child of {parent_id} toward sink {sid}, removed sink from normal subtree")

    # Process ACCEPT_CHILD
    for msg in robot.inbox:
        if msg.msg_type == "ACCEPT_CHILD" and msg.payload.get("child_id") == robot.rid:
            parent_id = msg.payload["parent_id"]
            parent_robot = robots[parent_id]
            robot.parent_id = parent_id
            robot.role = Role.NETWORK
            robot.desired_R = float(msg.payload["R"])
            d = Vec2(msg.payload["d_x"], msg.payload["d_y"])
            robot.desired_dir_d = safe_normalize(d)
            robot.sink_indices = list(msg.payload.get("sink_indices", robot.sink_indices))
            robot.branch_id = msg.payload.get("branch_id", "1")
            parent_type = parent_robot.node_type
            parent_parent_type = robots[parent_robot.parent_id].node_type if parent_robot.parent_id is not None else "NONE"
            print(f"[NETWORK] Robot {robot.rid} accepted as child of Robot {parent_id} "
                  f"(branch: {robot.branch_id}, parent_type={parent_type}, "
                  f"grandparent_type={parent_parent_type}, sinks={robot.sink_indices[:8]})")


def _attached_behavior(robot: Robot, dt: float, robots: List[Robot], sinks: List[Sink], touched: List[bool]):
    """Behavior for attached NETWORK robot - maintains void position and recruits children."""
    # 1) Local void controller
    if robot.desired_R is not None and robot.desired_dir_d is not None and robot.parent_id is not None:
        parent = robots[robot.parent_id]
        r = parent.pos - robot.pos
        dist = r.length()
        if dist > 1e-9:
            u = r / dist
            u_perp = Vec2(-u.y, u.x)
            u_star = -robot.desired_dir_d

            e_r = dist - robot.desired_R
            e_theta = cross2(u, u_star)

            v = (K_R * e_r) * u + (-K_THETA * e_theta) * u_perp
            if v.length() > SPEED_MAX:
                v = v.normalize() * SPEED_MAX

            robot.pos += v * dt
            robot.void_pos_vis = parent.pos + robot.desired_dir_d * robot.desired_R

    # Leg robots never recruit normal children and never provide guidance
    if robot.is_additive_leg or robot.is_al_child:
        return

    # Decide if I should recruit
    done = all(touched[sid] for sid in robot.sink_indices) if robot.sink_indices else False

    # Settlement check: robot must be close to its void position before recruiting
    is_settled = False
    if robot.desired_dir_d is not None and robot.desired_R is not None and robot.parent_id is not None:
        void_pos = robots[robot.parent_id].pos + robot.desired_dir_d * robot.desired_R
        is_settled = (robot.pos - void_pos).length() <= RECRUIT_SETTLE_THRESH

    if robot.node_type == NodeType.PIVOT:
        # Pivot recruits aggressively if has free slot and not done
        leaf_with_zero_sink = len(robot.sink_indices) == 0 and not robot.children_ids
        can_recruit = is_settled and (not done) and robot.has_free_slot() and (not leaf_with_zero_sink)
        
        if can_recruit:
            parent_ref = robots[robot.parent_id].pos if robot.parent_id is not None else robot.pos - Vec2(1, 0)
            incoming = robot.pos - parent_ref
            incoming_u = safe_normalize(incoming)
            
            has_star_child = False
            has_port_child = False
            for existing_child_id in robot.children_ids:
                existing_child = robots[existing_child_id]
                v_ec = existing_child.pos - robot.pos
                v_ec_u = safe_normalize(v_ec)
                existing_side_sgn = cross2(incoming_u, v_ec_u)
                if existing_side_sgn > 0:
                    has_star_child = True
                else:
                    has_port_child = True
            
            need_star = not has_star_child
            need_port = not has_port_child
            robot.broadcast("ASK_PARENT", need_star=need_star, need_port=need_port, 
                         incoming_x=float(incoming_u.x), incoming_y=float(incoming_u.y))
    else:
        # STRAIGHT node recruitment logic
        leaf_with_zero_sink = len(robot.sink_indices) == 0 and not robot.children_ids
        if done:
            can_recruit = is_settled and robot.has_free_slot() and (not leaf_with_zero_sink)
        else:
            # Only block recruitment if ALL assigned sinks are within touch distance
            # If there are distant sinks, we still need to recruit to reach them
            has_distant_sinks = any(
                (sinks[sid].pos - robot.pos).length() > SINK_TOUCH_DIST
                for sid in robot.sink_indices
            )
            all_sinks_near = not has_distant_sinks and len(robot.sink_indices) > 0
            can_recruit = is_settled and robot.has_free_slot() and (not all_sinks_near) and (not leaf_with_zero_sink)

    if can_recruit and robot.node_type != NodeType.PIVOT:
        robot.broadcast("ASK_PARENT")

    for msg in robot.inbox:
        if msg.msg_type == "PARENT_STATUS" and can_recruit:
            if msg.payload.get("has_parent", True):
                continue

            # Only accept if the free robot chose US as its preferred parent
            if msg.payload.get("responding_to") != robot.rid:
                continue

            # Re-check free slot here, after all filtering, so multiple messages
            # in the same tick cannot each pass before the first append runs
            if not robot.has_free_slot():
                break

            child_id = msg.sender_id
            
            if robot.parent_id is None:
                ref_parent_pos = robot.pos - Vec2(1, 0)
            else:
                ref_parent_pos = robots[robot.parent_id].pos

            if robot.node_type == NodeType.PIVOT:
                parent_ref = robots[robot.parent_id].pos if robot.parent_id is not None else robot.pos - Vec2(1, 0)
                incoming = robot.pos - parent_ref
                incoming_u = safe_normalize(incoming)
                v_pc = robots[child_id].pos - robot.pos
                v_pc_u = safe_normalize(v_pc)
                side_sgn = cross2(incoming_u, v_pc_u)
                
                existing_sides = {"STAR": False, "PORT": False}
                for existing_child_id in robot.children_ids:
                    existing_child = robots[existing_child_id]
                    v_ec = existing_child.pos - robot.pos
                    v_ec_u = safe_normalize(v_ec)
                    existing_side_sgn = cross2(incoming_u, v_ec_u)
                    if existing_side_sgn > 0:
                        existing_sides["STAR"] = True
                    else:
                        existing_sides["PORT"] = True
                
                child_side = "STAR" if side_sgn > 0 else "PORT"
                
                if existing_sides[child_side]:
                    continue
                
                star_sinks, port_sinks = split_sinks_star_port(robot.pos, parent_ref, robot.sink_indices, sinks)
                
                if side_sgn > 0:
                    child_sinks = star_sinks
                    child_branch_id = robot.branch_id + "1"
                else:
                    child_sinks = port_sinks
                    child_branch_id = robot.branch_id + "0"
                
                d = bisection_direction(robot.pos, ref_parent_pos, child_sinks, sinks)
            else:
                child_branch_id = robot.branch_id
                child_sinks = robot.sink_indices[:]
                d = bisection_direction(robot.pos, ref_parent_pos, robot.sink_indices, sinks)
            
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

    # ---- Additive leg recruitment ----
    # Conditions: has a regular child, >=2 assigned sinks, no additive leg yet,
    # settled, sees a hop-1 mover that itself sees an untouched sink.
    # Only a plain network robot (not an additive leg itself) can recruit an additive leg.
    # Also exclude AL children from this path entirely.
    is_plain_network = (not robot.is_additive_leg and not robot.is_al_child)
    parent_has_al = (
        robot.parent_id is not None
        and robots[robot.parent_id].additive_leg_child_id is not None
    )
    has_network_child = any(
        robots[cid].role == Role.NETWORK and not robots[cid].is_additive_leg
        for cid in robot.children_ids
    )
    al_eligible = (
        is_plain_network
        and robot.node_type == NodeType.STRAIGHT
        and not parent_has_al
        and robot.additive_leg_child_id is None
        and len(robot.sink_indices) >= 2
        and has_network_child
        and is_settled
    )
    if al_eligible:
        for other in robots:
            if other.role != Role.MOVING or other.parent_id is not None:
                continue
            if (other.pos - robot.pos).length() > COMM_RANGE:
                continue
            # find an untouched sink visible from the mover
            target_sink_id = None
            for sid in robot.sink_indices:
                if touched[sid]:
                    continue
                if (sinks[sid].pos - other.pos).length() <= R:
                    target_sink_id = sid
                    break
            if target_sink_id is None:
                continue
            # recruit this mover as additive leg
            sink_pos = sinks[target_sink_id].pos
            d = safe_normalize(sink_pos - robot.pos)
            robot.additive_leg_child_id = other.rid
            robot.broadcast(
                "ACCEPT_ADDITIVE_LEG",
                child_id=other.rid,
                parent_id=robot.rid,
                R=R,
                d_x=float(d.x),
                d_y=float(d.y),
                sink_id=target_sink_id,
            )
            break


def _try_recruit_al_child(robot: Robot, robots: List[Robot]):
    """
    Called after _attached_behavior for additive leg robots.
    Once settled, recruits one free mover as AL child (using standard ACCEPT_CHILD flow
    but with the AL's assigned sink as the only sink_index).
    """
    if not robot.sink_indices or robot.al_child_id is not None:
        return

    # Must be settled at void position first
    if robot.desired_dir_d is None or robot.desired_R is None or robot.parent_id is None:
        return
    parent = robots[robot.parent_id]
    void_pos = parent.pos + robot.desired_dir_d * robot.desired_R
    if (robot.pos - void_pos).length() > RECRUIT_SETTLE_THRESH:
        return

    sid = robot.sink_indices[0]
    for other in robots:
        if other.role != Role.MOVING or other.parent_id is not None:
            continue
        if other.is_additive_leg or other.is_al_child:
            continue
        if (other.pos - robot.pos).length() > COMM_RANGE:
            continue
        robot.al_child_id = other.rid
        robot.broadcast(
            "ACCEPT_AL_CHILD",
            child_id=other.rid,
            parent_id=robot.rid,
            R=R,
            sink_id=sid,
        )
        break


def _simple_move_towards(robot: Robot, target: Vec2, dt: float):
    """Move robot towards target at maximum speed."""
    _move_towards_with_speed(robot, target, SPEED_MAX, dt)


def _move_towards_with_speed(robot: Robot, target: Vec2, speed: float, dt: float):
    """Move robot towards target at specified speed."""
    d = target - robot.pos
    if d.length_squared() > 1e-12 and speed > 1e-9:
        u = d.normalize()
        robot.vel = u * speed
        robot.pos += robot.vel * dt
    else:
        robot.vel = Vec2(0, 0)


def _boid_move_direction(robot: Robot, robots: List[Robot], goal_target: Vec2) -> Vec2:
    """
    Compute Reynolds boid movement direction for a free moving robot.

    - Separation : repel from flock-mates that are too close
    - Cohesion   : steer toward the center of nearby flock-mates
    - Alignment  : replaced by goal-heading (direction to goal_target)

    Only other MOVING robots (no parent) count as flock-mates.
    Returns a normalised direction vector.
    """
    sep = Vec2(0, 0)
    center = Vec2(0, 0)
    count = 0

    for other in robots:
        if other.rid == robot.rid:
            continue
        if other.role != Role.MOVING or other.parent_id is not None or other.is_additive_leg or other.is_al_child:
            continue
        dist = (other.pos - robot.pos).length()
        if dist > BOID_NEIGHBOR_RADIUS:
            continue
        count += 1
        center += other.pos
        if dist < BOID_SEP_RADIUS and dist > 1e-9:
            # Stronger push the closer the neighbour
            sep += safe_normalize(robot.pos - other.pos) * (BOID_SEP_RADIUS / dist)

    # Modified alignment: each robot heads toward its own goal
    goal_dir = safe_normalize(goal_target - robot.pos)

    if count == 0:
        return goal_dir  # alone — just head to goal

    coh_dir = safe_normalize(center / count - robot.pos)
    sep_dir = safe_normalize(sep)

    combined = (BOID_SEP_WEIGHT * sep_dir
                + BOID_COH_WEIGHT * coh_dir
                + BOID_ALIGN_WEIGHT * goal_dir)

    if combined.length_squared() < 1e-12:
        return goal_dir
    return combined.normalize()


# =========================
# Pivot Metric & Local Candidacy
# =========================
def calc_pivot_metric(r: Robot, robots: List[Robot], sinks: List[Sink]) -> Optional[float]:
    """
    Calculate pivot metric for a robot.
    
    Uses the 120° criterion: compare angles between incoming direction
    and average directions to star/port sinks.
    """
    if r.parent_id is None:
        return None
    if len(r.sink_indices) < 2:
        return None

    parent = robots[r.parent_id]
    star, port = split_sinks_star_port(r.pos, parent.pos, r.sink_indices, sinks)
    if not star or not port:
        return None

    incoming = parent.pos - r.pos  # robot->parent
    incoming_u = safe_normalize(incoming)
    if incoming_u.length_squared() < 1e-12:
        return None

    v_star = avg_dir_to_sinks(r.pos, star, sinks)
    v_port = avg_dir_to_sinks(r.pos, port, sinks)

    theta_star = angle_between_deg(incoming_u, v_star)
    
    theta_port = angle_between_deg(incoming_u, v_port)

    return abs(theta_star - 120.0) + abs(theta_port - 120.0)


def is_local_pivot_candidate(
    rid: int,
    robots: List[Robot],
    errors: Dict[int, Optional[float]],
    pos_thresh: float = POS_THRESH,
) -> bool:
    """
    Check if a robot is a valid local pivot candidate.
    
    Requirements:
    - Must be NETWORK role with parent and at least one child
    - Must be close to its own void position
    - At least one child must be close to its void
    - Error < parent error (if parent is STRAIGHT) and < all child/grandchild errors
    """
    r = robots[rid]
    if r.role != Role.NETWORK:
        return False
    if r.parent_id is None or not r.children_ids:
        return False

    my_err = errors.get(rid)   
    if my_err is None:
        return False

    # close to my void
    if r.desired_dir_d is None or r.desired_R is None:
        return False
    parent = robots[r.parent_id]
    my_target = parent.pos + r.desired_dir_d * r.desired_R
    if (r.pos - my_target).length() > pos_thresh:
        return False

    # at least one child settled
    any_child_settled = False
    for cid in r.children_ids:
        c = robots[cid]
        if c.desired_dir_d is None or c.desired_R is None:
            continue
        c_target = r.pos + c.desired_dir_d * c.desired_R
        if (c.pos - c_target).length() <= pos_thresh:
            any_child_settled = True
            break
    if not any_child_settled:
        return False

    # Parent comparison
    p_err = errors.get(r.parent_id)
    parent = robots[r.parent_id]
    if parent.node_type == NodeType.STRAIGHT:
        if p_err is not None and my_err >= p_err:
            return False

    # Must be better than all children and grandchildren
    for cid in r.children_ids:
        c_err = errors.get(cid)
        if c_err is not None and my_err >= c_err:
            return False

        child_node = robots[cid]
        for gcid in child_node.children_ids:
            gc_err = errors.get(gcid)
            if gc_err is not None and my_err >= gc_err:
                return False

    return True


# =========================
# Guidance (multi-sink)
# =========================
def guidance_dir_from_robot(r: Robot, robots: List[Robot], touched: Optional[List[bool]]) -> Optional[Vec2]:
    """
    Calculate guidance direction from a NETWORK robot.
    
    Rules:
    - STRAIGHT nodes: if subtree done, no guidance; else guide to children
    - PIVOT nodes: guide toward unfinished branch(es)
    """
    if r.role != Role.NETWORK:
        return None

    if r.is_additive_leg or r.is_al_child:
        return None

    if not r.sink_indices:
        return None

    if touched is None:
        if r.children_ids:
            v = Vec2(0, 0)
            for cid in r.children_ids:
                v += (robots[cid].pos - r.pos)
            return safe_normalize(v) if v.length_squared() > 1e-12 else None
        if r.parent_id is not None:
            return safe_normalize(robots[r.parent_id].pos - r.pos)
        return None

    def subtree_done(node: Robot) -> bool:
        if not node.sink_indices:
            return True
        return all(touched[sid] for sid in node.sink_indices)

    my_subtree_done = subtree_done(r)

    # STRAIGHT node logic
    if r.node_type == NodeType.STRAIGHT:
        if my_subtree_done:
            return None
        else:
            if not r.children_ids:
                if r.parent_id is not None:
                    return safe_normalize(r.pos - robots[r.parent_id].pos)
                return None
            else:
                v = Vec2(0, 0)
                for cid in r.children_ids:
                    v += (robots[cid].pos - r.pos)
                return safe_normalize(v) if v.length_squared() > 1e-12 else None

    # PIVOT node logic
    else:
        if my_subtree_done:
            return None
        else:
            if r.parent_id is None:
                return None
            parent = robots[r.parent_id]
            incoming = r.pos - parent.pos
            incoming_u = safe_normalize(incoming)
            if incoming_u.length_squared() < 1e-12:
                return None
            
            star_dir = Vec2(-incoming_u.y, incoming_u.x)
            port_dir = Vec2(incoming_u.y, -incoming_u.x)
            
            has_star_child = False
            has_port_child = False
            star_child_done = True
            port_child_done = True
            
            for cid in r.children_ids:
                c = robots[cid]
                v_pc = c.pos - r.pos
                v_pc_u = safe_normalize(v_pc)
                sgn = cross2(incoming_u, v_pc_u)
                if sgn > 0:
                    has_star_child = True
                    if not subtree_done(c):
                        star_child_done = False
                else:
                    has_port_child = True
                    if not subtree_done(c):
                        port_child_done = False
            
            star_needs = not has_star_child or not star_child_done
            port_needs = not has_port_child or not port_child_done
            
            if star_needs and port_needs:
                return star_dir
            elif star_needs:
                return star_dir
            elif port_needs:
                return port_dir
            else:
                return None


# =========================
# Hop Count (moving robots only)
# =========================
def compute_hop_counts(robots: List[Robot]):
    """
    BFS from every NETWORK robot through the communication graph.
    Only MOVING robots receive a hop_count (>= 1).
    NETWORK robots are the seeds (distance 0, but we don't store that on them).
    SOURCE robots are ignored entirely.
    """
    for r in robots:
        r.hop_count = None

    queue: deque = deque()
    for r in robots:
        if r.role == Role.NETWORK:
            r.hop_count = 0          # seed; not displayed
            queue.append(r.rid)

    while queue:
        rid = queue.popleft()
        current = robots[rid]
        for other in robots:
            if other.hop_count is not None:
                continue
            if other.role != Role.MOVING:
                continue
            if (other.pos - current.pos).length() <= 2*R:
                other.hop_count = current.hop_count + 1
                queue.append(other.rid)


# =========================
# Messaging
# =========================
def deliver_messages(robots: List[Robot]):
    """Deliver messages from robot outboxes to other robots within communication range."""
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


# =========================
# Sink Touching / Done
# =========================
def node_sees_any_sink(pos: Vec2, sinks: List[Sink]) -> bool:
    """Check if a position is within touching distance of any sink."""
    for s in sinks:
        if (s.pos - pos).length() <= SINK_TOUCH_DIST:
            return True
    return False


def remove_sink_from_subtree(robots: List[Robot], root_id: int, sink_id: int):
    """Recursively remove a sink from a robot and all its descendants."""
    stack = [root_id]
    while stack:
        rid = stack.pop()
        r = robots[rid]
        if sink_id in r.sink_indices:
            r.sink_indices.remove(sink_id)
        stack.extend(r.children_ids)


def restore_sink_to_subtree(robots: List[Robot], root_id: int, sink_id: int):
    """Recursively add a sink back to a robot and all its descendants (if not already present)."""
    stack = [root_id]
    while stack:
        rid = stack.pop()
        r = robots[rid]
        if sink_id not in r.sink_indices:
            r.sink_indices.append(sink_id)
        stack.extend(r.children_ids)


def compute_touched_sinks(robots: List[Robot], sinks: List[Sink]) -> List[bool]:
    """Compute which sinks are touched by settled network robots."""
    touched = [False] * len(sinks)
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
            if (s.pos - r.pos).length() <= SINK_TOUCH_DIST:
                touched[s.sid] = True
    return touched


# =========================
# Retargeting (critical for recursion)
# =========================
def intersect_list(a: List[int], b: List[int]) -> List[int]:
    """Return intersection of two lists, preserving order from first list."""
    sb = set(b)
    return [x for x in a if x in sb]


def retarget_subtree_intersection(
    robots: List[Robot],
    sinks: List[Sink],
    root_id: int,
    allowed_sinks: List[int],
):
    """
    Retarget a subtree to only include sinks in allowed_sinks.
    Uses intersection to preserve splits from deeper pivots.
    Also recomputes desired_dir_d for each node.
    """
    stack = [root_id]
    allowed_set = set(allowed_sinks)

    while stack:
        rid = stack.pop()
        r = robots[rid]

        if r.sink_indices:
            new_list = [sid for sid in r.sink_indices if sid in allowed_set]
            r.sink_indices = new_list

        if r.parent_id is not None and r.desired_R is not None:
            parent = robots[r.parent_id]
            if parent.parent_id is None:
                ref_parent_pos = parent.pos - Vec2(1, 0)
            else:
                ref_parent_pos = robots[parent.parent_id].pos

            d = bisection_direction(parent.pos, ref_parent_pos, r.sink_indices, sinks)
            r.desired_dir_d = d
            r.void_pos_vis = parent.pos + d * r.desired_R

        stack.extend(r.children_ids)


def assign_branch_id_subtree(robots: List[Robot], root_id: int, new_branch_id: str):
    """Apply a branch_id to a node and all descendants."""
    stack = [root_id]
    while stack:
        rid = stack.pop()
        r = robots[rid]
        r.branch_id = new_branch_id
        stack.extend(r.children_ids)


# =========================
# Pivot Enforcement
# =========================
def enforce_pivot_split(
    pivot_id: int,
    robots: List[Robot],
    sinks: List[Sink],
):
    """
    Enforce pivot split on the chosen pivot robot.
    
    Operations:
    - Set as PIVOT with max_children=2
    - Split sink_indices into star/port
    - Assign children to sides based on position or touched sink
    - Release duplicate children on same side
    """
    p = robots[pivot_id]
    
    if p.node_type == NodeType.PIVOT:
        return
    
    p.node_type = NodeType.PIVOT
    p.max_children = 2

    if len(p.sink_indices) == 0:
        return

    if p.parent_id is None or len(p.sink_indices) < 2:
        return

    parent = robots[p.parent_id]
    star_sinks, port_sinks = split_sinks_star_port(p.pos, parent.pos, p.sink_indices, sinks)

    if not star_sinks or not port_sinks:
        return

    incoming = p.pos - parent.pos
    incoming_u = safe_normalize(incoming)
    if incoming_u.length_squared() < 1e-12:
        return

    star_sink_set = set(star_sinks)
    port_sink_set = set(port_sinks)

    def get_touched_sink_side(child: Robot) -> Optional[str]:
        for s in sinks:
            if (s.pos - child.pos).length() <= SINK_TOUCH_DIST:
                if s.sid in star_sink_set:
                    return "STAR"
                elif s.sid in port_sink_set:
                    return "PORT"
        return None

    side_to_children: Dict[str, List[int]] = {"STAR": [], "PORT": []}

    for cid in p.children_ids:
        c = robots[cid]
        
        touched_side = get_touched_sink_side(c)
        
        if touched_side is not None:
            assigned_side = touched_side
            print(f"[pivot-split] child={cid} assigned to {assigned_side} based on touched sink")
        else:
            v_pc = c.pos - p.pos
            v_pc_u = safe_normalize(v_pc)
            sgn = cross2(incoming_u, v_pc_u)
            assigned_side = "STAR" if sgn > 0 else "PORT"
        
        if assigned_side == "STAR":
            side_to_children["STAR"].append(cid)
            retarget_subtree_intersection(robots, sinks, cid, star_sinks)
            assign_branch_id_subtree(robots, cid, p.branch_id + "1")
        else:
            side_to_children["PORT"].append(cid)
            retarget_subtree_intersection(robots, sinks, cid, port_sinks)
            assign_branch_id_subtree(robots, cid, p.branch_id + "0")

    def release_extras(child_list: List[int]):
        if len(child_list) <= 1:
            return child_list
        child_list.sort()
        keep = child_list[0]
        for cid in child_list[1:]:
            child = robots[cid]
            child.parent_id = None
            child.role = Role.MOVING
            child.node_type = NodeType.STRAIGHT
            child.desired_R = None
            child.desired_dir_d = None
            child.void_pos_vis = None
            child.children_ids.clear()
            child.max_children = 1
            child.sink_indices = list(range(len(sinks)))

            if cid in p.children_ids:
                p.children_ids.remove(cid)

            print(f"[pivot-release] pivot={pivot_id} released child={cid} keep={keep}")
        return [keep]

    if star_sinks and port_sinks:
        side_to_children["STAR"] = release_extras(side_to_children["STAR"])
        side_to_children["PORT"] = release_extras(side_to_children["PORT"])

    def orient_child(cid: int, side_sinks: List[int]):
        child = robots[cid]
        child.parent_id = pivot_id
        child.role = Role.NETWORK
        child.node_type = NodeType.STRAIGHT
        child.desired_R = R
        dir_vec = bisection_direction(p.pos, parent.pos, side_sinks, sinks)
        child.desired_dir_d = safe_normalize(dir_vec)
        child.void_pos_vis = p.pos + child.desired_dir_d * child.desired_R
        child.max_children = 1

    for cid in side_to_children["STAR"]:
        orient_child(cid, star_sinks)
    for cid in side_to_children["PORT"]:
        orient_child(cid, port_sinks)


# =========================
# Demo Sink Placement
# =========================
def make_demo_sinks(source_pos: Vec2) -> List[Sink]:
    """Create randomly placed sinks in a configurable space above the source."""
    sinks: List[Sink] = []
    min_sep = SINK_MIN_SEP * COMM_RANGE
    max_attempts = 1000
    
    x_min = source_pos.x - SINK_SPACE_WIDTH / 2
    x_max = source_pos.x + SINK_SPACE_WIDTH / 2
    y_min = source_pos.y - SINK_SPACE_HEIGHT
    y_max = source_pos.y - 150
    
    min_dist_from_source = 150
    
    for i in range(NUM_SINKS):
        best_candidate: Optional[Vec2] = None
        best_min_dist = -1.0
        placed = False
        
        for _ in range(max_attempts):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            pos = Vec2(x, y)
            
            dist_from_source = (pos - source_pos).length()
            if dist_from_source < min_dist_from_source:
                continue
            
            if not sinks:
                candidate_min_dist = float("inf")
            else:
                candidate_min_dist = min((pos - s.pos).length() for s in sinks)
            
            if candidate_min_dist >= min_sep:
                sinks.append(Sink(i, pos))
                placed = True
                break
            
            if candidate_min_dist > best_min_dist:
                best_min_dist = candidate_min_dist
                best_candidate = pos
        
        if not placed:
            if best_candidate is None:
                best_candidate = Vec2(random.uniform(x_min, x_max), random.uniform(y_min, y_max))
            sinks.append(Sink(i, best_candidate))
            print(f"[make_demo_sinks] warning: placed sink {i} with min_dist={best_min_dist:.1f} (< {min_sep:.1f})")
    
    return sinks
