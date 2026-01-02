from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pygame
from pygame.math import Vector2 as Vec2

# =========================
# CONFIG
# =========================
WIDTH, HEIGHT = 900, 680
FPS = 60

R = 40.0                 # desired separation distance (void radius)
COMM_RANGE = 1.6 * R      # communication radius; must be > R
SPEED_MAX = 50        # max speed
SINK_TOUCH_DIST = COMM_RANGE

K_R = 25.0
K_THETA = 20.0

POS_THRESH = 3.0          # settled threshold for local pivot eligibility
PIVOT_THRESHOLD = 28.0    # smaller => harder to become pivot

MAX_ROBOTS = 35
SPAWN_INTERVAL = 5.0

NUM_SINKS = 3            # <======== change this
SINK_RING_R = 280.0       # radius for demo sink placement

random.seed(2)

# =========================
# Types / Roles
# =========================
class Role:
    MOVING = "MOVING"
    NETWORK = "NETWORK"
    SOURCE = "SOURCE"

class NodeType:
    STRAIGHT = "STRAIGHT"
    PIVOT = "PIVOT"

# =========================
# World objects
# =========================
@dataclass
class Sink:
    sid: int
    pos: Vec2

@dataclass
class Message:
    sender_id: int
    msg_type: str
    payload: Dict[str, Any]

# =========================
# Geometry helpers
# =========================
def safe_normalize(v: Vec2) -> Vec2:
    if v.length_squared() < 1e-12:
        return Vec2(0, 0)
    return v.normalize()

def angle_between_deg(a: Vec2, b: Vec2) -> float:
    if a.length_squared() < 1e-12 or b.length_squared() < 1e-12:
        return 0.0
    au = a.normalize()
    bu = b.normalize()
    dot = max(-1.0, min(1.0, au.x * bu.x + au.y * bu.y))
    return math.degrees(math.acos(dot))

def cross2(a: Vec2, b: Vec2) -> float:
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
            # exactly on line -> assign later (keep simple: add to smaller side)
            if len(star) <= len(port):
                star.append(sid)
            else:
                port.append(sid)
    return star, port

def avg_dir_to_sinks(node_pos: Vec2, sink_ids: List[int], sinks: List[Sink]) -> Vec2:
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
    Your definition:
      - Split sinks into star/port relative to parent->node
      - avg vector star, avg vector port
      - unit each, add => bisector
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
# Robot
# =========================
@dataclass
class Robot:
    rid: int
    pos: Vec2

    role: str = Role.MOVING
    node_type: str = NodeType.STRAIGHT

    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    max_children: int = 1

    # sinks assigned to this subtree
    sink_indices: List[int] = field(default_factory=list)

    inbox: List[Message] = field(default_factory=list)
    outbox: List[Message] = field(default_factory=list)

    desired_R: Optional[float] = None
    desired_dir_d: Optional[Vec2] = None
    void_pos_vis: Optional[Vec2] = None

    def has_free_slot(self) -> bool:
        return len(self.children_ids) < self.max_children

    def broadcast(self, msg_type: str, **payload):
        self.outbox.append(Message(self.rid, msg_type, dict(payload)))

    def step(self, dt: float, robots: List["Robot"], sinks: List[Sink], touched: List[bool]):
        if self.role == Role.SOURCE:
            self._source_behavior(dt, robots, sinks, touched)
        elif self.parent_id is None:
            self._free_behavior(dt, robots, sinks)
        else:
            self._attached_behavior(dt, robots, sinks, touched)

    # -------------------
    # Behaviors
    # -------------------
    def _source_behavior(self, dt: float, robots: List["Robot"], sinks: List[Sink], touched: List[bool]):
        # Source just recruits (one child). You can make it a pivot too later, but keep it simple.
        if self.has_free_slot():
            self.broadcast("ASK_PARENT")

        for msg in self.inbox:
            if msg.msg_type == "PARENT_STATUS" and self.has_free_slot():
                if msg.payload.get("has_parent", True):
                    continue

                child_id = msg.sender_id
                self.children_ids.append(child_id)

                # Direction for child: average towards all sinks (source has no parent)
                d = avg_dir_to_sinks(self.pos, self.sink_indices, sinks)
                child_sinks = self.sink_indices[:]

                self.broadcast(
                    "ACCEPT_CHILD",
                    child_id=child_id,
                    parent_id=self.rid,
                    R=R,
                    d_x=float(d.x),
                    d_y=float(d.y),
                    sink_indices=child_sinks,
                )

    def _free_behavior(self, dt: float, robots: List["Robot"], sinks: List[Sink]):
        # If someone asks for children, reply "no parent" and move toward closest asker.
        askers: List[int] = [m.sender_id for m in self.inbox if m.msg_type == "ASK_PARENT"]

        if askers:
            self.broadcast("PARENT_STATUS", has_parent=False)

            best_id = min(askers, key=lambda aid: (robots[aid].pos - self.pos).length())
            self._simple_move_towards(robots[best_id].pos, dt)
        else:
            # Follow guidance from nearest NETWORK robot in range; else go toward source-ish.
            best_guidance = None
            best_dist = float("inf")
            for other in robots:
                if other.rid == self.rid or other.role != Role.NETWORK:
                    continue
                dist = (other.pos - self.pos).length()
                if dist > COMM_RANGE:
                    continue
                g = guidance_dir_from_robot(other, robots, sinks, touched=None)  # guidance doesn't need sinks here
                if g is None:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_guidance = g

            if best_guidance is not None:
                self._simple_move_towards(self.pos + best_guidance * R, dt)
            else:
                # drift toward screen center (source lives there)
                self._simple_move_towards(Vec2(WIDTH * 0.45, HEIGHT * 0.65), dt)

        # Process ACCEPT_CHILD
        for msg in self.inbox:
            if msg.msg_type == "ACCEPT_CHILD" and msg.payload.get("child_id") == self.rid:
                self.parent_id = msg.payload["parent_id"]
                self.role = Role.NETWORK
                self.desired_R = float(msg.payload["R"])
                d = Vec2(msg.payload["d_x"], msg.payload["d_y"])
                self.desired_dir_d = safe_normalize(d)
                self.sink_indices = list(msg.payload.get("sink_indices", self.sink_indices))

    def _attached_behavior(self, dt: float, robots: List["Robot"], sinks: List[Sink], touched: List[bool]):
        # 1) Local void controller (same as your 2-sink code)
        if self.desired_R is not None and self.desired_dir_d is not None and self.parent_id is not None:
            parent = robots[self.parent_id]
            r = parent.pos - self.pos
            dist = r.length()
            if dist > 1e-9:
                u = r / dist
                u_perp = Vec2(-u.y, u.x)
                u_star = -self.desired_dir_d

                e_r = dist - self.desired_R
                e_theta = cross2(u, u_star)

                v = (K_R * e_r) * u + (-K_THETA * e_theta) * u_perp
                if v.length() > SPEED_MAX:
                    v = v.normalize() * SPEED_MAX

                self.pos += v * dt
                self.void_pos_vis = parent.pos + self.desired_dir_d * self.desired_R

        # 2) Decide if I should recruit
        done = all(touched[sid] for sid in self.sink_indices) if self.sink_indices else False

        if self.node_type == NodeType.PIVOT:
            # pivot recruits only if NOT done and still has free slot
            can_recruit = (not done) and self.has_free_slot()
        else:
            # straight recruits only if NOT done and has free slot and doesn't see a sink too close
            can_recruit = (not done) and self.has_free_slot() and (not node_sees_any_sink(self.pos, sinks))

        if can_recruit:
            self.broadcast("ASK_PARENT")

        for msg in self.inbox:
            if msg.msg_type == "PARENT_STATUS" and can_recruit:
                if msg.payload.get("has_parent", True):
                    continue

                child_id = msg.sender_id
                self.children_ids.append(child_id)

                # For the link (self -> child), compute direction using YOUR bisection rule,
                # which needs self's parent position as reference.
                if self.parent_id is None:
                    ref_parent_pos = self.pos - Vec2(1, 0)
                else:
                    ref_parent_pos = robots[self.parent_id].pos

                d = bisection_direction(self.pos, ref_parent_pos, self.sink_indices, sinks)
                child_sinks = self.sink_indices[:]

                self.broadcast(
                    "ACCEPT_CHILD",
                    child_id=child_id,
                    parent_id=self.rid,
                    R=R,
                    d_x=float(d.x),
                    d_y=float(d.y),
                    sink_indices=child_sinks,
                )

    def _simple_move_towards(self, target: Vec2, dt: float):
        d = target - self.pos
        if d.length_squared() > 1e-12:
            self.pos += d.normalize() * SPEED_MAX * dt

# =========================
# Pivot metric & local candidacy
# =========================
def calc_pivot_metric(r: Robot, robots: List[Robot], sinks: List[Sink]) -> Optional[float]:
    """
    Use your criterion but generalized:

    incoming = robot->parent
    split assigned sinks into star/port wrt parent->robot
    then compare 120° between incoming (robot->parent) and avg_star, avg_port.
    """
    if r.parent_id is None:
        return None
    if len(r.sink_indices) < 2:
        return None

    parent = robots[r.parent_id]
    star, port = split_sinks_star_port(r.pos, parent.pos, r.sink_indices, sinks)
    if not star or not port:
        return None

    incoming = parent.pos - r.pos     # robot->parent
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
    Same rule as your 2-sink version:

    - must be NETWORK, have parent and at least one child
    - must be close to its own void
    - at least one child close to its void
    - error < parent error and < all child errors
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

    p_err = errors.get(r.parent_id)
    if p_err is not None and my_err >= p_err:
        return False

    for cid in r.children_ids:
        c_err = errors.get(cid)
        if c_err is not None and my_err >= c_err:
            return False

    return True

# =========================
# Guidance (multi-sink)
# =========================
def guidance_dir_from_robot(r: Robot, robots: List[Robot], sinks: List[Sink], touched: Optional[List[bool]]) -> Optional[Vec2]:
    """
    MOVING robots follow NETWORK robots.

    If touched is provided:
      - prefer directing toward children whose subtrees are NOT done.
      - if everything below is done, guide back to parent.

    If touched is None (free robots call with touched=None), just follow the tree direction heuristically.
    """
    if r.role != Role.NETWORK:
        return None

    # If we don't have touched info, fallback to child direction if exists else parent direction.
    if touched is None:
        if r.children_ids:
            v = Vec2(0, 0)
            for cid in r.children_ids:
                v += (robots[cid].pos - r.pos)
            return safe_normalize(v) if v.length_squared() > 1e-12 else None
        if r.parent_id is not None:
            return safe_normalize(robots[r.parent_id].pos - r.pos)
        return None

    # With touched info, prefer unfinished children.
    def subtree_done(node: Robot) -> bool:
        return bool(node.sink_indices) and all(touched[sid] for sid in node.sink_indices)

    unfinished_children = [cid for cid in r.children_ids if not subtree_done(robots[cid])]

    if unfinished_children:
        # steer toward the "best" unfinished child (closest in angle / or just average)
        v = Vec2(0, 0)
        for cid in unfinished_children:
            v += (robots[cid].pos - r.pos)
        return safe_normalize(v) if v.length_squared() > 1e-12 else None

    # No unfinished children: guide back to parent if possible.
    if r.parent_id is not None:
        return safe_normalize(robots[r.parent_id].pos - r.pos)

    return None

# =========================
# Messaging
# =========================
def deliver_messages(robots: List[Robot]):
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
# Sink touching / done
# =========================
def node_sees_any_sink(pos: Vec2, sinks: List[Sink]) -> bool:
    for s in sinks:
        if (s.pos - pos).length() <= SINK_TOUCH_DIST:
            return True
    return False

def compute_touched_sinks(robots: List[Robot], sinks: List[Sink]) -> List[bool]:
    touched = [False] * len(sinks)
    for r in robots:
        if r.role != Role.NETWORK:
            continue
        for s in sinks:
            if (s.pos - r.pos).length() <= SINK_TOUCH_DIST:
                touched[s.sid] = True
    return touched

# =========================
# Retargeting (critical for recursion)
# =========================
def intersect_list(a: List[int], b: List[int]) -> List[int]:
    sb = set(b)
    return [x for x in a if x in sb]

def retarget_subtree_intersection(
    robots: List[Robot],
    sinks: List[Sink],
    root_id: int,
    allowed_sinks: List[int],
):
    """
    IMPORTANT:
    We do NOT overwrite everything blindly, because deeper pivots might have already split more.
    So we intersect each node.sink_indices with allowed_sinks.

    Also we recompute desired_dir_d for each node after sink set changes,
    using the parent's position (and parent's parent position for bisection).
    """
    stack = [root_id]
    allowed_set = set(allowed_sinks)

    while stack:
        rid = stack.pop()
        r = robots[rid]

        if r.sink_indices:
            new_list = [sid for sid in r.sink_indices if sid in allowed_set]
            if not new_list:
                # if intersection becomes empty (rare), keep old to avoid dead robots
                new_list = r.sink_indices[:]
            r.sink_indices = new_list

        # recompute link direction from parent to this node
        if r.parent_id is not None and r.desired_R is not None:
            parent = robots[r.parent_id]
            # reference parent of "parent" for bisection
            if parent.parent_id is None:
                ref_parent_pos = parent.pos - Vec2(1, 0)
            else:
                ref_parent_pos = robots[parent.parent_id].pos

            # direction for this edge uses THIS node's sink_indices
            d = bisection_direction(parent.pos, ref_parent_pos, r.sink_indices, sinks)
            r.desired_dir_d = d
            r.void_pos_vis = parent.pos + d * r.desired_R

        stack.extend(r.children_ids)

# =========================
# Pivot enforcement
# =========================
def enforce_pivot_split(
    pivot_id: int,
    robots: List[Robot],
    sinks: List[Sink],
):
    """
    For the chosen pivot:
      - set as PIVOT, max_children=2
      - split pivot.sink_indices into star/port
      - for each child: decide which side child lies on, and retarget that child subtree
        to the matching sink subset (via intersection so deeper pivots keep their splits).
      - if both children accidentally fall on same side while the other side has sinks,
        release the higher rid child (turn it MOVING again).
    """
    p = robots[pivot_id]
    p.node_type = NodeType.PIVOT
    p.max_children = max(2, p.max_children)

    if p.parent_id is None or len(p.sink_indices) < 2:
        return

    parent = robots[p.parent_id]
    star_sinks, port_sinks = split_sinks_star_port(p.pos, parent.pos, p.sink_indices, sinks)

    if not star_sinks or not port_sinks:
        # can't split -> behaves like straight effectively
        return

    # Determine side for each child
    incoming = p.pos - parent.pos  # parent->pivot
    incoming_u = safe_normalize(incoming)
    if incoming_u.length_squared() < 1e-12:
        return

    side_to_children: Dict[str, List[int]] = {"STAR": [], "PORT": []}

    for cid in p.children_ids:
        c = robots[cid]
        v_pc = c.pos - p.pos
        v_pc_u = safe_normalize(v_pc)
        sgn = cross2(incoming_u, v_pc_u)
        if sgn > 0:
            side_to_children["STAR"].append(cid)
            retarget_subtree_intersection(robots, sinks, cid, star_sinks)
        else:
            side_to_children["PORT"].append(cid)
            retarget_subtree_intersection(robots, sinks, cid, port_sinks)

    # If duplicates on one side and the other side exists, release extras.
    # Keep the lowest rid on that side (your policy: higher index becomes MOVING).
    def release_list(child_list: List[int]):
        if len(child_list) <= 1:
            return
        child_list.sort()
        keep = child_list[0]
        for cid in child_list[1:]:
            # detach cid (only the head; subtree remains attached to it — acceptable for this sim)
            child = robots[cid]
            child.parent_id = None
            child.role = Role.MOVING
            child.node_type = NodeType.STRAIGHT
            child.desired_R = None
            child.desired_dir_d = None
            child.void_pos_vis = None
            child.children_ids.clear()   # keep it simple: drop its subtree when releasing
            child.max_children = 1
            child.sink_indices = list(range(len(sinks)))

            if cid in p.children_ids:
                p.children_ids.remove(cid)

            print(f"[pivot-release] pivot={pivot_id} released child={cid} keep={keep}")

    # Only release duplicates if the opposite side is non-empty (otherwise duplicates are inevitable).
    if star_sinks and port_sinks:
        release_list(side_to_children["STAR"])
        release_list(side_to_children["PORT"])

# =========================
# Demo sink placement
# =========================
def make_demo_sinks(source_pos: Vec2) -> List[Sink]:
    sinks: List[Sink] = []
    # place sinks on an arc above the source
    start_ang = -160
    end_ang = -20
    for i in range(NUM_SINKS):
        t = 0.5 if NUM_SINKS == 1 else i / (NUM_SINKS - 1)
        ang = math.radians(start_ang + t * (end_ang - start_ang))
        pos = source_pos + Vec2(math.cos(ang), math.sin(ang)) * SINK_RING_R
        # small jitter so not perfectly symmetric
        pos += Vec2(random.uniform(-15, 15), random.uniform(-15, 15))
        sinks.append(Sink(i, pos))
    return sinks

# =========================
# Main
# =========================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Multi-sink network growth (recursive pivot splitting)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

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
        sink_indices=list(range(len(sinks))),
    )
    robots.append(r0)

    # Initial moving robot
    robots.append(Robot(
        rid=1,
        pos=SOURCE_POS + Vec2(-260, -50),
        role=Role.MOVING,
        node_type=NodeType.STRAIGHT,
        max_children=1,
        sink_indices=list(range(len(sinks))),
    ))

    sim_time = 0.0
    next_spawn_time = 1.0
    pivot_id: Optional[int] = None

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        sim_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # spawn
        if len(robots) < MAX_ROBOTS and sim_time >= next_spawn_time:
            rid = len(robots)
            spawn_pos = Vec2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
            robots.append(Robot(
                rid=rid,
                pos=spawn_pos,
                role=Role.MOVING,
                node_type=NodeType.STRAIGHT,
                max_children=1,
                sink_indices=list(range(len(sinks))),
            ))
            next_spawn_time += SPAWN_INTERVAL

        # clear outboxes
        for r in robots:
            r.outbox.clear()

        # touched sinks (global truth for "done")
        touched = compute_touched_sinks(robots, sinks)

        # compute errors
        errors: Dict[int, Optional[float]] = {}
        for r in robots:
            if r.role != Role.NETWORK or len(r.sink_indices) < 2:
                errors[r.rid] = None
            else:
                errors[r.rid] = calc_pivot_metric(r, robots, sinks)

        # pick a pivot using your local rule
        pivot_id = None
        best_err = None
        for r in robots:
            if not is_local_pivot_candidate(r.rid, robots, errors):
                continue
            E = errors[r.rid]
            if E is None:
                continue
            if best_err is None or E < best_err:
                best_err = E
                pivot_id = r.rid

        if best_err is None or best_err >= PIVOT_THRESHOLD:
            pivot_id = None

        # enforce pivot split (recursive behavior via sink set intersection)
        if pivot_id is not None:
            enforce_pivot_split(pivot_id, robots, sinks)

        # step robots
        for r in robots:
            r.step(dt, robots, sinks, touched)

        # deliver messages
        deliver_messages(robots)

        # =========================
        # DRAW
        # =========================
        screen.fill((30, 30, 30))

        # draw sinks
        for s in sinks:
            col = (220, 60, 60) if not touched[s.sid] else (60, 220, 60)
            pygame.draw.circle(screen, col, s.pos, 9)
            screen.blit(font.render(f"S{s.sid}", True, (240, 240, 240)),
                        (s.pos.x + 10, s.pos.y - 8))

        # draw source
        pygame.draw.circle(screen, (0, 200, 0), SOURCE_POS, 12)
        screen.blit(font.render("SOURCE", True, (255, 255, 255)),
                    (SOURCE_POS.x - 35, SOURCE_POS.y + 14))

        # draw robots
        for r in robots:
            if r.role == Role.SOURCE:
                color, rad = (0, 200, 0), 10
            elif pivot_id is not None and r.rid == pivot_id:
                color, rad = (255, 0, 255), 8
            elif r.role == Role.NETWORK:
                color, rad = (200, 200, 255), 7
            else:
                color, rad = (0, 160, 255), 7

            pygame.draw.circle(screen, color, r.pos, rad)

            # label
            screen.blit(font.render(f"R{r.rid}", True, (255, 255, 255)),
                        (r.pos.x + 6, r.pos.y - 10))

            # pivot metric text
            E = errors.get(r.rid)
            if E is not None:
                screen.blit(font.render(f"{E:.1f}", True, (255, 255, 0)),
                            (r.pos.x + 6, r.pos.y + 8))

            # void marker
            if r.void_pos_vis is not None:
                pygame.draw.circle(screen, (255, 255, 0), r.void_pos_vis, 4, 1)

            # show assigned sinks count (so you can see recursion)
            if r.role == Role.NETWORK and r.parent_id is not None:
                screen.blit(font.render(f"|S|={len(r.sink_indices)}", True, (160, 160, 160)),
                            (r.pos.x - 18, r.pos.y + 14))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
