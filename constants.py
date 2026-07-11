"""
Constants and Configuration for Multi-Sink Network Formation Simulation
Core Version - No additive leg.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pygame.math import Vector2 as Vec2

# =========================
# SIMULATION CONFIGURATION
# =========================
WIDTH, HEIGHT = 1000,1000
FPS = 120

# =========================
# NETWORK FORMATION PARAMETERS
# =========================
R = 20                      # desired separation distance (void radius)
COMM_RANGE = 2 * R                  # communication radius; must be > R
SPEED_MAX = 75                      # max speed
SINK_TOUCH_DIST = 1 * R             # distance to consider sink "touched"

# Void controller gains
K_R = 50                            # radial gain
K_THETA = 70                        # angular gain

# Pivot eligibility thresholds
POS_THRESH = 1                      # settled threshold for local pivot eligibility
PIVOT_THRESHOLD =60    # raw angular error threshold (E); robot must have E <= this
PIVOT_SCORE_THRESHOLD = max(0.0, 1.0 - PIVOT_THRESHOLD / 120.0)  # normalized: 1 - E/120
SINK_TOUCH_SETTLEMENT_THRESH = 3 # robot must be this close to void pos to count as touching
RECRUIT_SETTLE_THRESH = 1           # robot must be this close to its void pos before recruiting

# Pivot scoring mode:
#   0 = default: score = max(0, 1 - E/120)  (no sink proximity bonus)
#   1 = score-based: score = max(0, 1 - E/120) + 0.2 * sink_in_desired_range
PIVOT_SCORE_MODE = 0
# Robot spawning
MAX_ROBOTS = 65
SPAWN_INTERVAL = 20                 # sim-time between flock releases

# Flock spawning (circle formation)
FLOCK_SIZE = 12                      # robots per flock
FLOCK_RADIUS = 1.0                  # circle radius as multiple of R

# Guidance
GUIDANCE_STICK_FRAMES = 0           # short lock to keep guidance priority

# =========================
# BOID FLOCKING PARAMETERS
# =========================
BOID_NEIGHBOR_RADIUS = 2 * R
BOID_SEP_RADIUS      = 1.8 * R
BOID_SEP_WEIGHT      = 1.5 * R
BOID_COH_WEIGHT      = 0.5
BOID_ALIGN_WEIGHT    = 2.0

# =========================
# SINK CONFIGURATION
# =========================
NUM_SINKS = 5
SINK_SPACE_WIDTH  = 2500/5
SINK_SPACE_HEIGHT = 2500/5
SINK_MIN_SEP = 5                    # multiplied by COMM_RANGE

# =========================
# VISUALIZATION SETTINGS
# =========================
VISUALIZE_ON = True
PIVOT_MARKER_SIZE  = 25
ROBOT_MARKER_SIZE  = 12
SINK_MARKER_SIZE   = 10
SOURCE_MARKER_SIZE = 18

# =========================
# RANDOM SEED
# =========================
random.seed(2)


# =========================
# ENUMERATIONS
# =========================
class Role:
    MOVING  = "MOVING"
    NETWORK = "NETWORK"
    SOURCE  = "SOURCE"


class NodeType:
    STRAIGHT = "STRAIGHT"
    PIVOT    = "PIVOT"


# =========================
# DATA CLASSES
# =========================
@dataclass
class Sink:
    sid: int
    pos: Vec2
    charging_level: float =100


@dataclass
class Obstacle:
    pos:    Vec2
    radius: float


@dataclass
class Message:
    sender_id: int
    msg_type:  str
    payload:   Dict[str, Any]


@dataclass
class Robot:
    """
    Robot agent in the network formation simulation.

    Key fields
    ----------
    rid             : unique ID
    pos             : current position
    role            : MOVING | NETWORK | SOURCE
    node_type       : STRAIGHT | PIVOT
    parent_id       : tree parent (None if free)
    children_ids    : list of child robot IDs
    max_children    : capacity; set to 0 when robot sees a sink
    branch_id       : binary path from root ("1", "10", "11", "100", …)
    sink_indices    : sinks assigned to this subtree
    sees_sink       : True when robot is within void-position + sink-touch distance
    """
    rid: int
    pos: Vec2

    role:      str = Role.MOVING
    node_type: str = NodeType.STRAIGHT

    parent_id:    Optional[int] = None
    children_ids: List[int]     = field(default_factory=list)
    max_children: int           = 1

    branch_id:    str       = "1"
    sink_indices: List[int] = field(default_factory=list)

    inbox:  List[Message] = field(default_factory=list)
    outbox: List[Message] = field(default_factory=list)

    desired_R:     Optional[float] = None
    desired_dir_d: Optional[Vec2]  = None
    void_pos_vis:  Optional[Vec2]  = None

    vel:      Vec2 = field(default_factory=lambda: Vec2(0, 0))
    prev_pos: Vec2 = field(default_factory=lambda: Vec2(0, 0))

    # Guidance lock (moving robots)
    guidance_lock_robot_id: Optional[int] = None
    guidance_lock_frames:   int           = 0

    # Hop count (moving robots only; None = unreachable)
    hop_count: Optional[int] = None

    # Sink-seeing state (triggers zero capacity and propagates "done" signal)
    sees_sink: bool = False

    # Branch-pruning cooldown: steps remaining before pivot can recruit on freed side
    pivot_cooldown: int = 0

    # Battery reports accumulated at a pivot: {"STAR": level, "PORT": level}
    battery_reports: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.prev_pos = Vec2(self.pos)
        self.vel      = Vec2(0, 0)

    def has_free_slot(self) -> bool:
        return len(self.children_ids) < self.max_children

    def broadcast(self, msg_type: str, **payload):
        self.outbox.append(Message(self.rid, msg_type, dict(payload)))
