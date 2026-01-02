from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import random
import pygame
from pygame.math import Vector2 as Vec2
WIDTH, HEIGHT = 800, 600
FPS = 60
R   = 40.0      # desired separation
COMM_RANGE = 1.5*R     # communication radius- it should be larger than R
SPEED_MAX  = 100.0     # max robot speed (pixels per second)
PIVOT_THRESHOLD = 70  # max pivot error to accept pivot role
POS_THRESH = 3   # position threshold for FINDING BEST LOCAL PIVOT
K_R      = 25   # radial gain
K_THETA  = 20   # angular/orbital gain
SINK_TOUCH_DIST = COMM_RANGE   # or maybe R, you can tune this based on COMM_RANGE
CURRENT_PIVOT_ID: Optional[int] = None
BOTH_BRANCHES_DONE: bool = False   # this just basically say that hey both branches are done, we use this later to reverse guidance vectors for some robots

class Role:
    MOVING  = "MOVING"
    NETWORK = "NETWORK"
    SOURCE  = "SOURCE"

class NodeType:
    STRAIGHT = "STRAIGHT"
    PIVOT    = "PIVOT"



# I started with a simple equilateral triangle setup for debugging, but you can change this later
SIDE = 200.0

# Base vertex (source) somewhere reasonable on screen
SOURCE_POS = Vec2(WIDTH/2 - SIDE/2, HEIGHT/2 + SIDE/4)+Vec2(-100,100)

# Equilateral triangle: base from SOURCE_POS to SINK1_POS, apex at SINK2_POS
SINK1_POS = SOURCE_POS + Vec2(SIDE, 0.0)+Vec2(10,100)
SINK2_POS = SOURCE_POS + Vec2(SIDE/2.0,
                              - (math.sqrt(3)/2.0) * SIDE)+Vec2(10,-170)

SINK3_POS = SOURCE_POS + Vec2(300,-100)



def split_sinks_port_star(point: Vec2, sink_indices: List[int]) -> tuple[list[int], list[int], Vec2, Vec2]:
    """
    Split sink_indices into (starboard, port) relative to v_rs = (SOURCE - point).
    Returns:
      star_ids, port_ids, star_dir_unit, port_dir_unit
    star_dir_unit is normalized sum of unit vectors to starboard sinks (same for port).
    """
    v_rs = SOURCE_POS - point
    if v_rs.length_squared() == 0:
        return [], [], Vec2(0,0), Vec2(0,0)
    v_rs_u = v_rs.normalize()

    star_ids: list[int] = []
    port_ids: list[int] = []
    star_vec = Vec2(0,0)
    port_vec = Vec2(0,0)

    for idx in sink_indices:
        v = SINK_POSITIONS[idx] - point
        if v.length_squared() == 0:
            continue
        u = v.normalize()
        cross = v_rs_u.x * u.y - v_rs_u.y * u.x  # z of 2D cross
        if cross > 0:
            star_ids.append(idx)
            star_vec += u
        elif cross < 0:
            port_ids.append(idx)
            port_vec += u
        else:
            # exactly on bisector: toss it into the smaller group
            if len(star_ids) <= len(port_ids):
                star_ids.append(idx); star_vec += u
            else:
                port_ids.append(idx); port_vec += u

    star_u = star_vec.normalize() if star_vec.length_squared() > 0 else Vec2(0,0)
    port_u = port_vec.normalize() if port_vec.length_squared() > 0 else Vec2(0,0)
    return star_ids, port_ids, star_u, port_u

SINK_POSITIONS = [SINK1_POS, SINK2_POS,SINK3_POS]   
# these are some functions for post procssign and have nothing do with the main logic. # here we just find the Fermat point of a triangle
def fermat_steiner_point(a: Vec2, b: Vec2, c: Vec2) -> Vec2:
    """
    For an equilateral triangle, the Fermat/Steiner point is just the centroid.
    Since we’re explicitly setting an equilateral geometry, just return (a+b+c)/3.
    """
    return (a + b + c) / 3.0
def rotate60(v: Vec2, sign: int = +1) -> Vec2:
    """Rotate vector v by ±60 degrees around the origin."""
    cos60 = 0.5
    sin60 = math.sqrt(3.0) / 2.0
    if sign > 0:
        return Vec2(
            cos60 * v.x - sin60 * v.y,
            sin60 * v.x + cos60 * v.y
        )
    else:
        return Vec2(
            cos60 * v.x + sin60 * v.y,
           -sin60 * v.x + cos60 * v.y
        )


def line_intersection(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> Optional[Vec2]:
    """
    Intersection of infinite lines p1–p2 and p3–p4.
    Returns None if lines are parallel / nearly parallel.
    """
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y
    x4, y4 = p4.x, p4.y

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None

    num_px = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    num_py = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)

    return Vec2(num_px / den, num_py / den)


def angle_at_vertex(a: Vec2, b: Vec2, c: Vec2) -> float:
    """Angle ABC in degrees (at vertex b)."""
    v1 = a - b
    v2 = c - b
    if v1.length_squared() == 0 or v2.length_squared() == 0:
        return 0.0
    v1u = v1.normalize()
    v2u = v2.normalize()
    dot = max(-1.0, min(1.0, v1u.x * v2u.x + v1u.y * v2u.y))
    return math.degrees(math.acos(dot))


def angle_between(u: Vec2, v: Vec2) -> float:
    """Angle between vectors u and v in degrees."""
    if u.length_squared() == 0 or v.length_squared() == 0:
        return 0.0
    uu = u.normalize()
    vv = v.normalize()
    dot = max(-1.0, min(1.0, uu.x * vv.x + uu.y * vv.y))
    return math.degrees(math.acos(dot))


def fermat_steiner_point(a: Vec2, b: Vec2, c: Vec2) -> Vec2:
    """
    Fermat / Steiner point (shortest 3-branch network) for triangle with vertices a, b, c.

    - If any interior angle >= 120°, the Fermat point is that vertex.
    - Otherwise:
        * Construct equilateral triangles on AB and AC (both ±60° orientations),
        * Intersect lines C–D and B–E for each orientation combo,
        * Pick the intersection where ∠APB, ∠BPC, ∠CPA are closest to 120°.
    """
    # 1) angle test for obtuse case
    A = angle_at_vertex(b, a, c)  # ∠BAC
    B = angle_at_vertex(a, b, c)  # ∠ABC
    C = angle_at_vertex(a, c, b)  # ∠ACB

    if A >= 120.0:
        return a
    if B >= 120.0:
        return b
    if C >= 120.0:
        return c

    # 2) acute case: use equilateral construction on AB and AC
    v_ab = b - a
    v_ac = c - a

    # equilateral vertices on AB
    D_plus  = a + rotate60(v_ab, +1)
    D_minus = a + rotate60(v_ab, -1)

    # equilateral vertices on AC
    E_plus  = a + rotate60(v_ac, +1)
    E_minus = a + rotate60(v_ac, -1)

    candidates: List[Vec2] = []

    # Try four orientation combinations: (D+,E+), (D+,E-), (D-,E+), (D-,E-)
    for D in (D_plus, D_minus):
        for E in (E_plus, E_minus):
            P = line_intersection(c, D, b, E)
            if P is None:
                continue

            # Check 120° property at P
            PA = a - P
            PB = b - P
            PC = c - P

            ang_APB = angle_between(PA, PB)
            ang_BPC = angle_between(PB, PC)
            ang_CPA = angle_between(PC, PA)

            err = (abs(ang_APB - 120.0) +
                   abs(ang_BPC - 120.0) +
                   abs(ang_CPA - 120.0))

            candidates.append((err, P))

    if not candidates:
        # Extremely degenerate; fall back to centroid
        return (a + b + c) / 3.0

    # 3) choose the intersection that best satisfies the 120° triple
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]








# list of the assigend sink for a robot can be changed so bisection direction is computed from that
def bisection_dir_from(point: Vec2, sink_indices: List[int]) -> Vec2:
    # side-based bisection (star_u + port_u), fallback to simple avg if one side is empty
    star_ids, port_ids, star_u, port_u = split_sinks_port_star(point, sink_indices)

    if star_u.length_squared() > 0 and port_u.length_squared() > 0:
        s = star_u + port_u
        return s.normalize() if s.length_squared() > 0 else Vec2(1,0)

    # fallback: normal average of all sinks (needed when sink set is already one-sided)
    s = Vec2(0,0)
    for idx in sink_indices:
        h = SINK_POSITIONS[idx] - point
        if h.length_squared() > 0:
            s += h.normalize()
    return s.normalize() if s.length_squared() > 0 else Vec2(1,0)


# this is a very simpel protocol for messaging between robots 
@dataclass
class Message:
    # simplest class possible for messages , sender , type , payload
    sender_id: int
    msg_type: str         
    payload: Dict[str, Any]


@dataclass
class Robot:
    rid: int
    pos: Vec2
    role: str = Role.MOVING
    node_type: str = NodeType.STRAIGHT
    parent_id: Optional[int] = None          # None = no parent yet
    owner_pivot: Optional[int] = None   # which pivot created my current sink set
    pivot_depth: int = 0                # how many splits happened upstream
    sink_indices: List[int] = field(default_factory=lambda: list(range(len(SINK_POSITIONS))))
 # default_factory is used to create a new list for each instance

    inbox: List[Message]  = field(default_factory=list)
    outbox: List[Message] = field(default_factory=list)

   
    desired_R: Optional[float] = None        # target distance to parent
    desired_dir_d: Optional[Vec2] = None     # parent -> sink direction

    
    max_children: int = 1
    children_ids: List[int] = field(default_factory=list)  # this means list of robot ids

    void_pos_vis: Optional[Vec2] = None
    branch_done: bool = False
    @staticmethod  # static method means that it does not depend on instance variables.

    
    def choose_sink_group_for_child(pivot_pos: Vec2, child_pos: Vec2, sink_indices: List[int]) -> List[int]:
            star_ids, port_ids, star_u, port_u = split_sinks_port_star(pivot_pos, sink_indices)

            # if one side is empty, no split possible -> give everything
            if not star_ids or not port_ids:
                return sink_indices[:]

            v_ps = SOURCE_POS - pivot_pos
            if v_ps.length_squared() == 0:
                return sink_indices[:]
            v_ps_u = v_ps.normalize()

            v_pc = child_pos - pivot_pos
            if v_pc.length_squared() == 0:
                # child sits on pivot -> choose the bigger side
                return star_ids if len(star_ids) >= len(port_ids) else port_ids

            v_pc_u = v_pc.normalize()
            child_side = v_ps_u.x * v_pc_u.y - v_ps_u.y * v_pc_u.x

            if child_side > 0:
                return star_ids
            if child_side < 0:
                return port_ids

            # exactly on bisector: choose side whose average direction is closer
            def angle(u: Vec2, v: Vec2) -> float:
                dot = max(-1.0, min(1.0, u.x * v.x + u.y * v.y))
                return math.degrees(math.acos(dot))

            a_star = angle(v_pc_u, star_u) if star_u.length_squared() > 0 else 1e9
            a_port = angle(v_pc_u, port_u) if port_u.length_squared() > 0 else 1e9
            return star_ids if a_star <= a_port else port_ids




    def choose_sink_for_child_side(pivot_pos: Vec2, child_pos: Vec2) -> int:
        """
        this code is basicaally to decide which sink to assign to a child of a pivot based on which side of the pivot the child is on
        Decide which sink (0 or 1) should be assigned to a child of a pivot,
        based on which side of the pivot it lies on relative to the pivot->source direction.
        """
        # reference direction: pivot -> source: means vector from pivot to source
        v_ps = SOURCE_POS - pivot_pos
        if v_ps.length_squared() == 0:
            # degenerate, just default to sink 0
            return 0
        v_ps_u = v_ps.normalize()
        # what direction is the child relative to pivot?

        # child side
        v_pc = child_pos - pivot_pos
        if v_pc.length_squared() == 0:
            # child is on top of pivot; arbitrary default
            child_side = 0.0
        else:
            v_pc_u = v_pc.normalize()
            child_side = v_ps_u.x * v_pc_u.y - v_ps_u.y * v_pc_u.x

        # sink sides
        sink_side = []
        for idx in [0, 1]:
            v_psink = SINK_POSITIONS[idx] - pivot_pos
            if v_psink.length_squared() == 0:
                side = 0.0
            else:
                v_psink_u = v_psink.normalize()
                side = v_ps_u.x * v_psink_u.y - v_ps_u.y * v_psink_u.x
            sink_side.append(side)

        # If child is on the same sign side as a sink, choose that sink.
        # child_side > 0 -> pick sink with side > 0, etc.
        if child_side > 0:
            if sink_side[0] > 0 and sink_side[1] <= 0:
                return 0
            if sink_side[1] > 0 and sink_side[0] <= 0:
                return 1
        elif child_side < 0:
            if sink_side[0] < 0 and sink_side[1] >= 0:
                return 0
            if sink_side[1] < 0 and sink_side[0] >= 0:
                return 1

        # Fallbacks: child exactly on the bisector, or sinks both on same side.
        # Choose the sink with smaller angle to child.
        best_idx = 0
        best_angle = None
        for idx in [0, 1]:
            v = SINK_POSITIONS[idx] - pivot_pos
            if v.length_squared() == 0 or v_pc.length_squared() == 0:
                angle = 0.0
            else:
                a = v.normalize()
                b = v_pc.normalize()
                dot = max(-1.0, min(1.0, a.x * b.x + a.y * b.y))
                angle = math.degrees(math.acos(dot))
            if best_angle is None or angle < best_angle:
                best_angle = angle
                best_idx = idx
        return best_idx

    def has_free_slot(self) -> bool:
        return len(self.children_ids) < self.max_children
    
    def assign_child_to_sink(parent: Robot, child: Robot, sink_index: int):
        """Force this child to be on parent's branch to a single sink."""
        child.parent_id = parent.rid  # already true, but harmless
        child.sink_indices = [sink_index]
        child.desired_R = R

        d = SINK_POSITIONS[sink_index] - parent.pos
        if d.length_squared() > 0:
            d = d.normalize()
        child.desired_dir_d = d

        # thiss part is just for visualization: where this child *should* converge to
        child.void_pos_vis = parent.pos + d * R
    def retarget_branch_to_sink(robots: List[Robot], root_id: int, sink_index: int):
        # this function is to retarget an entire branch of robots to a specific sink, why? becasue sometimes we want to retarget a branch when we find that the other branch is done
        stack = [root_id]
        while stack:
            rid = stack.pop()  # Last-In, First-Out for DFS traversal
            r = robots[rid]

            # Only this sink from now on
            r.sink_indices = [sink_index]

            # Recompute desired direction & void (relative to parent)
            if r.parent_id is not None:
                parent = robots[r.parent_id]
                d = SINK_POSITIONS[sink_index] - parent.pos
                if d.length_squared() > 0:
                    d = d.normalize()
                r.desired_R = R if r.desired_R is None else r.desired_R
                r.desired_dir_d = d
                r.void_pos_vis = parent.pos + d * r.desired_R

            # Continue with descendants
            stack.extend(r.children_ids)


    def broadcast(self, msg_type: str, **payload):
        self.outbox.append(Message(self.rid, msg_type, dict(payload)))

    def retarget_branch_to_sinkset(robots: List[Robot], root_id: int, sink_set: List[int]):
        stack = [root_id]
        while stack:
            rid = stack.pop()
            r = robots[rid]
            r.sink_indices = list(sink_set)

            if r.parent_id is not None:
                parent = robots[r.parent_id]
                d = bisection_dir_from(parent.pos, r.sink_indices)
                r.desired_R = R if r.desired_R is None else r.desired_R
                r.desired_dir_d = d
                r.void_pos_vis = parent.pos + d * r.desired_R

            stack.extend(r.children_ids)

      

    



    def step_behavior(self, dt: float, robots: List["Robot"]):


        # we have three major behaviors based on role, 
        # sourrce behavior which is only for source robot
        # free behavior which is for robots that are not attached to the network yet
        # attached behavior which is for robots that are already part of the network
        if self.role == Role.SOURCE:
            self._source_behavior(dt)
        elif self.parent_id is None:
            self._free_behavior(dt, robots)
        else:
            self._attached_behavior(dt, robots)

   
    def _source_behavior(self, dt: float):
        # If source has free child slot, advertise
        if self.has_free_slot():
            self.broadcast("ASK_PARENT")

        # Process replies
        for msg in self.inbox:
            if msg.msg_type == "PARENT_STATUS" and self.has_free_slot():
                if not msg.payload["has_parent"]:
                    child_id = msg.sender_id
                    self.children_ids.append(child_id)

                    # Decide direction & sinks for THIS child
                    if self.node_type == NodeType.PIVOT:
                        # Pivot: 2 children, one for Sink1, one for Sink2
                        if len(self.children_ids) == 1:
                            chosen_sink = 0          # first child -> Sink1
                        elif len(self.children_ids) == 2:
                            chosen_sink = 1          # second child -> Sink2
                        else:
                            continue  # no more capacity

                        sink_pos = SINK_POSITIONS[chosen_sink]
                        d = sink_pos - self.pos
                        if d.length_squared() > 0:
                            d = d.normalize()

                        child_sinks = [chosen_sink]  # child only cares about ONE sink

                    else:
                        
                        d = bisection_dir_from(self.pos, self.sink_indices)
                        child_sinks = self.sink_indices[:]  

                    self.broadcast(
                        "ACCEPT_CHILD",
                        child_id=child_id,
                        parent_id=self.rid,
                        R=R,
                        d_x=float(d.x),
                        d_y=float(d.y),
                        sink_indices=child_sinks   
                    )
                            # DEBUG
                    print(f"[ASK_PARENT] rid={self.rid} role={self.role}")



    
    def _free_behavior(self, dt: float, robots: List["Robot"]):
        askers: List[int] = []

        # detect who is advertising
        for msg in self.inbox:
            if msg.msg_type == "ASK_PARENT":
                askers.append(msg.sender_id)

        if askers:
            # somebody wants children
            self.broadcast("PARENT_STATUS", has_parent=False)

            best_id = None
            best_dist = float("inf")
            for aid in askers:
                parent = robots[aid]
                dvec = parent.pos - self.pos
                dist = dvec.length()
                if dist < best_dist:
                    best_dist = dist
                    best_id = aid

            if best_id is not None:
                self._simple_move_towards(robots[best_id].pos, dt)

        else:
            # no ASK_PARENT messages instrad  follow guidance from nearest network robot
            best_guidance = None
            best_dist = float("inf")

            for other in robots:
                if other.rid == self.rid:
                    continue
                if other.role != Role.NETWORK:
                    continue

                dvec = other.pos - self.pos
                dist = dvec.length()
                if dist > COMM_RANGE:
                    continue

               # we should use current pivot here.
                g = guidance_dir_from_robot(other, robots, pivot_id=CURRENT_PIVOT_ID)
                if g is None:
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_guidance = g

            if best_guidance is not None:
                target = self.pos + best_guidance * R
                self._simple_move_towards(target, dt)
            else:
                # last resort
                self._simple_move_towards(SOURCE_POS, dt)

       # this part is to process ACCEPT_CHILD messages
        for msg in self.inbox:
            if msg.msg_type == "ACCEPT_CHILD" and msg.payload["child_id"] == self.rid:
                self.parent_id = msg.payload["parent_id"]
                self.role      = Role.NETWORK
                self.desired_R = float(msg.payload["R"])
                d = Vec2(msg.payload["d_x"], msg.payload["d_y"])
                if "owner_pivot" in msg.payload:
                    self.owner_pivot = msg.payload["owner_pivot"]
                if "pivot_depth" in msg.payload:
                    self.pivot_depth = int(msg.payload["pivot_depth"])

                if d.length_squared() > 0:
                    d = d.normalize()
                self.desired_dir_d = d

                if "sink_indices" in msg.payload:
                    self.sink_indices = list(msg.payload["sink_indices"])


    def _attached_behavior(self, dt: float, robots: List["Robot"]):
        # 1) move into void relative to parent using local controller


        if self.desired_R is not None and self.desired_dir_d is not None:
            parent = robots[self.parent_id]
            r = parent.pos - self.pos
            dist = r.length()
            if dist > 1e-6:
                u = r / dist
                u_perp = Vec2(-u.y, u.x)
                u_star = -self.desired_dir_d 

                e_r = dist - self.desired_R
                cross = u.x * u_star.y - u.y * u_star.x
                e_theta = cross

                v_r = K_R * e_r * u
                v_theta = -K_THETA * e_theta * u_perp
                v = v_r + v_theta

                vlen = v.length()
                if vlen > SPEED_MAX:
                    v = v * (SPEED_MAX / vlen)

                self.pos += v * dt

                # ideal void location for drawing (where *this robot* wants to be)
                self.void_pos_vis = parent.pos + self.desired_dir_d * self.desired_R

        # 2) recruit a child if we still have capacity
        # 2) recruit a child if we still have capacity AND do NOT see a sink
        # in _attached_behavior
                # 2) recruit a child if we still have capacity
        #    - for normal nodes: must NOT see a sink and NOT be on finished branch
        #    - for pivot: ignore node_sees_sink; we still want to fill the other branch
        if self.node_type == NodeType.PIVOT:
            can_recruit = self.has_free_slot() 
        else:
            can_recruit = (
                self.has_free_slot()
                and not node_sees_sink(self)
                and not self.branch_done
            )
                # DEBUG
        if self.rid == CURRENT_PIVOT_ID:
            print(
                f"[pivot] rid={self.rid} children={self.children_ids} "
                f"max_children={self.max_children} "
                f"has_free_slot={self.has_free_slot()} "
                f"branch_done={self.branch_done} "
                f"sees_sink={node_sees_sink(self)}"
            )


       


        if can_recruit:
            self.broadcast("ASK_PARENT")

        for msg in self.inbox:
            if msg.msg_type == "PARENT_STATUS" and can_recruit:
                

                if not msg.payload["has_parent"]:
                    child_id = msg.sender_id
                    self.children_ids.append(child_id)

                    # Decide direction & sinks for THIS child
                    if self.node_type == NodeType.PIVOT:
                        # choose sink based on which side of the pivot the child is on
                        child = robots[child_id]
                        chosen_sink =Robot.choose_sink_for_child_side(self.pos, child.pos)

                        # direction from pivot to that sink
                        sink_pos = SINK_POSITIONS[chosen_sink]
                        d = sink_pos - self.pos
                        if d.length_squared() > 0:
                            d = d.normalize()

                        # child and all its descendants should ONLY care about this sink
                        child_sinks = [chosen_sink]

                    else:
                        # Straight node: bisection of its own sink set,
                        # and pass the same sink set down unchanged
                        d = bisection_dir_from(self.pos, self.sink_indices)
                        child_sinks = self.sink_indices[:]

                    # Send local void spec to the child
                    if self.node_type == NodeType.PIVOT:
                        new_owner = self.rid
                        new_depth = self.pivot_depth + 1
                    else:
                        new_owner = self.owner_pivot
                        new_depth = self.pivot_depth
                    self.broadcast(
                        "ACCEPT_CHILD",
                        child_id=child_id,
                        parent_id=self.rid,
                        R=R,
                        d_x=float(d.x),
                        d_y=float(d.y),
                        sink_indices=child_sinks,
                        owner_pivot=new_owner,
                        pivot_depth=new_depth
                    )

    def _simple_move_towards(self, target: Vec2, dt: float):
        d = target - self.pos
        if d.length_squared() > 1e-6:
            v = d.normalize() * SPEED_MAX
            self.pos += v * dt

def calc_pivot_metric(robot: Robot) -> Optional[float]:
    v_rs = SOURCE_POS - robot.pos
    if v_rs.length_squared() == 0:
        return None
    v_rs_u = v_rs.normalize()

    star_ids, port_ids, star_u, port_u = split_sinks_port_star(robot.pos, robot.sink_indices)

    if not star_ids or not port_ids:
        return None  # can't be a pivot if you can't split into two sides

    def angle_deg(a: Vec2, b: Vec2) -> float:
        dot = max(-1.0, min(1.0, a.x * b.x + a.y * b.y))
        return math.degrees(math.acos(dot))

    theta_star = angle_deg(v_rs_u, star_u)
    theta_port = angle_deg(v_rs_u, port_u)
    return abs(theta_star - 120.0) + abs(theta_port - 120.0)


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



def assign_child_to_sink(parent: Robot, child: Robot, sink_index: int):
    Robot.assign_child_to_sink(parent, child, sink_index)


def retarget_branch_to_sink(robots: List[Robot], root_id: int, sink_index: int):
   
    Robot.retarget_branch_to_sink(robots, root_id, sink_index)

def elect_pivots_hierarchical(robots: List[Robot], errors: Dict[int, Optional[float]]) -> List[int]:
    full_set_size = len(SINK_POSITIONS)
    best_in_group: Dict[tuple, int] = {}

    for r in robots:
        if r.role != Role.NETWORK:
            continue
        if len(r.sink_indices) < 2:
            continue

        E = errors.get(r.rid, None)
        if E is None or E >= PIVOT_THRESHOLD:
            continue
        if not is_local_pivot_candidate(r.rid, robots, errors):
            continue

        # Root pivot ONLY from full sink set (your “first pivot is 3 sinks” rule)
        if r.owner_pivot is None and len(r.sink_indices) != full_set_size:
            continue

        key = (r.owner_pivot, tuple(sorted(r.sink_indices)))
        if key not in best_in_group or E < errors[best_in_group[key]]:
            best_in_group[key] = r.rid

    return list(best_in_group.values())

def is_local_pivot_candidate(
    rid: int,
    robots: List[Robot],
    errors: Dict[int, Optional[float]],
    pos_thresh: float = POS_THRESH,
) -> bool:
    """
    Local rule:

    - I must be NETWORK and have a parent and at least one child.
    - I and at least one child must be close to our void targets.
    - My pivot error must be < parent_error and < all child_errors.
    """
    r = robots[rid]

    # Must be in the network
    if r.role != Role.NETWORK:
        return False

    my_err = errors.get(rid, None)
    if my_err is None:
        return False

    # Need both parent and at least one child to compare against
    if r.parent_id is None or not r.children_ids:
        return False

    # --- 1) me close to my target void ---
    if r.desired_dir_d is None or r.desired_R is None:
        return False

    parent = robots[r.parent_id]
    my_target = parent.pos + r.desired_dir_d * r.desired_R
    if (r.pos - my_target).length() > pos_thresh:
        return False

    # --- 2) at least one child close to its target void ---
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

    # --- 3) my_error < parent_error (if parent has an error) ---
    p_err = errors.get(r.parent_id, None)
    if p_err is not None and my_err >= p_err:
        return False

    # --- 4) my_error < child_error for all children that have an error ---
    for cid in r.children_ids:
        c_err = errors.get(cid, None)
        if c_err is not None and my_err >= c_err:
            return False

    return True
def guidance_dir_from_robot(r: Robot, robots: List[Robot], pivot_id: Optional[int]) -> Optional[Vec2]:
    """
    Guidance direction for MOVING robots.

    - On unfinished branch (branch_done == False):
        forward, parent -> child (outwards toward sinks).
    - On finished branch (branch_done == True):
        backward, child -> parent (toward pivot).
    - At pivot:
        only consider children on the NOT-finished branch; guide into that side.

    After both branches have touched their sinks (BOTH_BRANCHES_DONE == True),
    guidance vectors from NETWORK robots with rid < pivot_id are reversed
    (single minus sign).
    """
    global BOTH_BRANCHES_DONE

    def maybe_flip(v: Vec2) -> Vec2:
        # Only flip if:
        # - both branches are done
        # - we actually have a pivot
        # - this robot's id is less than the pivot id
        if BOTH_BRANCHES_DONE and pivot_id is not None and r.rid < pivot_id:
            return -v
        return v

    if r.role != Role.NETWORK:
        return None

    # --- Pivot special case: steer into the not-full (unfinished) branch ---
    if pivot_id is not None and r.rid == pivot_id:
        v = Vec2(0, 0)
        count = 0
        for cid in r.children_ids:
            c = robots[cid]
            if c.branch_done:
                continue  # skip finished branch
            v += (c.pos - r.pos)  # pivot -> child on unfinished side
            count += 1
        if count == 0 or v.length_squared() == 0:
            return None
        return maybe_flip(v).normalize()

    # --- Non-pivot nodes ---

    if r.branch_done:
        # finished branch: guide back to parent (toward pivot)
        if r.parent_id is None:
            return None
        parent = robots[r.parent_id]
        v = parent.pos - r.pos        # child -> parent
        if v.length_squared() == 0:
            return None
        return maybe_flip(v).normalize()
    else:
        # unfinished branch: guide outwards along children if possible
        if r.children_ids:
            v = Vec2(0, 0)
            for cid in r.children_ids:
                c = robots[cid]
                v += (c.pos - r.pos)   # to children
            if v.length_squared() == 0:
                return None
            return maybe_flip(v).normalize()
        else:
            # leaf but not finished: roughly forward (away from parent)
            if r.parent_id is None:
                return None
            parent = robots[r.parent_id]
            v = r.pos - parent.pos     # parent -> this node
            if v.length_squared() == 0:
                return None
            return maybe_flip(v).normalize()


def update_branch_done(robots: List[Robot], pivot_id: Optional[int]):
    """
    Mark branch_done = True for nodes on a branch that touched a sink,
    from the leaf up to (but NOT including) the pivot.
    """
    # reset
    for r in robots:
        r.branch_done = False

    if pivot_id is None:
        return

    frontier: List[int] = []

    # seeds: NETWORK nodes that actually see a sink
    for r in robots:
        if r.role != Role.NETWORK:
            continue
        if node_sees_sink(r):
            frontier.append(r.rid)

    # propagate upwards until pivot
    while frontier:
        rid = frontier.pop()
        if rid == pivot_id:
            continue  # do not mark pivot or go below

        r = robots[rid]
        if r.branch_done:
            continue

        r.branch_done = True  # this node is on a finished branch

        if r.parent_id is None or r.parent_id == pivot_id:
            continue
        frontier.append(r.parent_id)
SINK_TOUCH_DIST = COMM_RANGE   # you can tune this

def node_sees_sink(r: Robot) -> bool:
    for sink_pos in SINK_POSITIONS:
        if (sink_pos - r.pos).length() <= SINK_TOUCH_DIST:
            return True
    return False
def release_extra_pivot_children(robots: List[Robot], pivot_id: int):
    pivot = robots[pivot_id]
    if len(pivot.children_ids) <= 2:
        return

    # Determine which side each child belongs to by looking at its sink set
    star_keep = None
    port_keep = None
    star_list = []
    port_list = []

    star_ids, port_ids, _, _ = split_sinks_port_star(pivot.pos, pivot.sink_indices)

    for cid in pivot.children_ids:
        c = robots[cid]
        # If child's sink set intersects mostly with star_ids => star side
        star_overlap = len(set(c.sink_indices) & set(star_ids))
        port_overlap = len(set(c.sink_indices) & set(port_ids))
        if star_overlap >= port_overlap:
            star_list.append(cid)
        else:
            port_list.append(cid)

    # Keep lowest rid in each side, release others
    for side_list in (star_list, port_list):
        side_list.sort()
        if len(side_list) <= 1:
            continue
        keep = side_list[0]
        for cid in side_list[1:]:
            child = robots[cid]
            child.parent_id = None
            child.role = Role.MOVING
            child.desired_R = None
            child.desired_dir_d = None
            child.void_pos_vis = None
            child.branch_done = False
            child.sink_indices = list(range(len(SINK_POSITIONS)))  # reset to "all sinks"
            pivot.children_ids.remove(cid)
            print(f"[pivot-release] pivot={pivot_id} released child={cid} keep={keep}")

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Automatic chain + pivot selection + 2 child voids")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    robots: List[Robot] = []

    # R0: source
    r0 = Robot(rid=0, pos=SOURCE_POS, role=Role.SOURCE, max_children=1)
    robots.append(r0)

    # First moving robot at t=0
    r1 = Robot(rid=1, pos=SOURCE_POS + Vec2(-220, -60), max_children=1)
    robots.append(r1)

    MAX_ROBOTS      = 20
    SPAWN_INTERVAL  = 7
    next_spawn_time = 1

    sim_time = 0.0
    running = True
    pivot_id: Optional[int] = None
    steiner_pos = fermat_steiner_point(SOURCE_POS, SINK1_POS, SINK2_POS)

    while running:
        dt = clock.tick(FPS) / 1000.0
        sim_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        
        if len(robots) < MAX_ROBOTS and sim_time >= next_spawn_time:
            rid = len(robots)

            # random position inside the screen
            spawn_pos = Vec2(
                random.uniform(0, WIDTH),
                random.uniform(0, HEIGHT)
            )

            robots.append(Robot(rid=rid, pos=spawn_pos, max_children=1))
            next_spawn_time += SPAWN_INTERVAL

        # --- 1) compute and cache errors for all robots ---
        errors: Dict[int, Optional[float]] = {}
        for r in robots:
            if r.role != Role.NETWORK:
                errors[r.rid] = None
            else:
                errors[r.rid] = calc_pivot_metric(r)

        # --- 2) local pivot selection: I must beat parent & children and be settled ---
        pivot_ids = elect_pivots_hierarchical(robots, errors)
        pivot_id = pivot_ids[0] if pivot_ids else None   # temporary: keep your old code working

        best_local_err: Optional[float] = None
        # reset first
        for r in robots:
            r.node_type = NodeType.STRAIGHT
            r.max_children = 1

        for pid in pivot_ids:
            pivot = robots[pid]
            pivot.node_type = NodeType.PIVOT
            pivot.max_children = 2

            for cid in pivot.children_ids:
                child = robots[cid]
                sink_set = Robot.choose_sink_group_for_child(pivot.pos, child.pos, pivot.sink_indices)
                Robot.retarget_branch_to_sinkset(robots, root_id=cid, sink_set=sink_set)

            release_extra_pivot_children(robots, pid)

        for r in robots:
            if not is_local_pivot_candidate(r.rid, robots, errors):
                continue

            E = errors[r.rid]
            if E is None:
                continue

            # if multiple local winners exist, pick the one with smallest error
            if best_local_err is None or E < best_local_err:
                best_local_err = E
                pivot_id = r.rid

        # optional: still enforce a global error threshold
        if best_local_err is None or best_local_err >= PIVOT_THRESHOLD:
            pivot_id = None


        
        if pivot_id is not None:
            p = robots[pivot_id]
            pygame.draw.circle(screen, (80,80,80), p.pos, COMM_RANGE, 1)
            pivot = robots[pivot_id]
            pivot.node_type = NodeType.PIVOT
            pivot.max_children = 2

            # For each child of the pivot, decide which sink it is on,
            # then retarget the ENTIRE branch under that child to that sink.
            for cid in pivot.children_ids:
                child = robots[cid]
                   
                # Decide sink based on side of the triangle (same logic as your method)
                sink_set = Robot.choose_sink_group_for_child(pivot.pos, child.pos, pivot.sink_indices)
                Robot.retarget_branch_to_sinkset(robots, root_id=cid, sink_set=sink_set)


                      # After retargeting, ensure at most one child per sink
            release_extra_pivot_children(robots, pivot_id)

   
        update_branch_done(robots, pivot_id)
            # after computing pivot_id:

        # after computing pivot_id:
        global CURRENT_PIVOT_ID, BOTH_BRANCHES_DONE
        CURRENT_PIVOT_ID = pivot_id

        # One-time detection, we should check if we  have both sinks been touched by some NETWORK robot?
        if not BOTH_BRANCHES_DONE:
            sinks_touched = [False] * len(SINK_POSITIONS)

            for r in robots:
                if r.role != Role.NETWORK:
                    continue
                for idx, sink_pos in enumerate(SINK_POSITIONS):
                    if (sink_pos - r.pos).length() <= SINK_TOUCH_DIST:
                        sinks_touched[idx] = True

            if all(sinks_touched):
                BOTH_BRANCHES_DONE = True
                print(">>> Both branches done, guidance will be reversed.")
       
        
       
        for r in robots:
            r.outbox.clear()

        for r in robots:
            r.step_behavior(dt, robots)

        deliver_messages(robots)

        
        screen.fill((30, 30, 30))

      
        pygame.draw.line(screen, (100,100,100), SOURCE_POS, SINK1_POS, 2)
        pygame.draw.line(screen, (100,100,100), SINK1_POS,  SINK2_POS, 2)
        pygame.draw.line(screen, (100,100,100), SINK2_POS,  SOURCE_POS, 2)

        pygame.draw.circle(screen, (0,200,0), SOURCE_POS, 10)
        screen.blit(font.render("Source", True, (255,255,255)),
                    (SOURCE_POS.x-30, SOURCE_POS.y+12))
        pygame.draw.circle(screen, (200,0,0), SINK1_POS, 8)
        screen.blit(font.render("Sink1", True, (255,255,255)),
                    (SINK1_POS.x-15, SINK1_POS.y+10))
        pygame.draw.circle(screen, (200,0,0), SINK2_POS, 8)
        screen.blit(font.render("Sink2", True, (255,255,255)),
                    (SINK2_POS.x-15, SINK2_POS.y-20))
        pygame.draw.circle(screen, (200,0,0), SINK3_POS, 8)
        screen.blit(font.render("Sink3", True, (255,255,255)),
                    (SINK3_POS.x+8, SINK3_POS.y-10))
        pygame.draw.circle(screen, (0, 255, 0), steiner_pos, 6)
        screen.blit(font.render("Steiner", True, (0, 255, 0)),
                    (steiner_pos.x + 8, steiner_pos.y - 10))
        
        for r in robots:
            if r.role == Role.SOURCE:
                color = (0,200,0); radius = 10
            elif r.rid == pivot_id:
                color = (255,0,255); radius = 8
            elif r.role == Role.NETWORK:
                color = (200,200,255); radius = 7
            else:
                color = (0,150,255); radius = 7

            pygame.draw.circle(screen, color, r.pos, radius)
            label = f"R{r.rid}"
            screen.blit(font.render(label, True, (255,255,255)),
                        (r.pos.x+6, r.pos.y-10))

            if r.void_pos_vis is not None:
                pygame.draw.circle(screen, (255,255,0), r.void_pos_vis, 4, 1)

            E = calc_pivot_metric(r)
            if E is not None:
                txt = f"{E:.1f}"
                screen.blit(font.render(txt, True, (255,255,0)),
                            (r.pos.x+6, r.pos.y+8))

        
        if pivot_id is not None:
            p = robots[pivot_id].pos
            for idx in [0, 1]:
                d = SINK_POSITIONS[idx] - p
                if d.length_squared() == 0:
                    continue
                d = d.normalize()
                void_child_pos = p + d * R
                pygame.draw.circle(screen, (255,255,0), void_child_pos, 6, 2)

        pygame.display.flip()

    pygame.quit()
if __name__ == "__main__":
    main()




# if(
#(norm(my_pos, my_target_pos) < thresh))
#&&(norm(child_pos, child_target_pos)
#&&(my_error < parent_error)
#&&(my_error < child_error)
#{
 #  become_pivot()
#}