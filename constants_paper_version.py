"""
Constants and Configuration for Multi-Sink Network Formation Simulation
Paper Version - Constants, Configuration Parameters, and Data Classes
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pygame.math import Vector2 as Vec2

# =========================
# SIMULATION CONFIGURATION
# =========================
WIDTH, HEIGHT = 800, 800
FPS = 120

# =========================
# NETWORK FORMATION PARAMETERS
# =========================
R = 60                   # desired separation distance (void radius)
COMM_RANGE = 3 * R        # communication radius; must be > R
SPEED_MAX =200           # max speed
SINK_TOUCH_DIST = 1*R       # distance to consider sink touched

# Void controller gains
K_R = 100                # radial gain
K_THETA = 150          # angular gain

# Pivot eligibility thresholds
POS_THRESH = 2            # settled threshold for local pivot eligibility
PIVOT_THRESHOLD =60      # smaller => harder to become pivot
SINK_TOUCH_SETTLEMENT_THRESH = 1  # allow sink touch when within ~1-2 units of void

# Robot spawning
MAX_ROBOTS = 500
SPAWN_INTERVAL =5

# Guidance
GUIDANCE_STICK_FRAMES = 0 # short lock to keep recruitment priority over guidance

# =========================
# SINK CONFIGURATION
# =========================
NUM_SINKS = 10            # number of sinks to place
SINK_SPACE_WIDTH = 2000   # Width of random sink placement region
SINK_SPACE_HEIGHT = 2000 # Height of random sink placement region
SINK_MIN_SEP = 5          # Minimum separation between sinks (multiplied by COMM_RANGE)
SINK_RING_R = 400         # radius for demo sink placement (legacy)

# =========================
# VISUALIZATION SETTINGS
# =========================
VISUALIZE_ON = False       # Set to False for headless mode (faster simulation)
PIVOT_MARKER_SIZE = 25    # Size for pivot robots in plots
ROBOT_MARKER_SIZE = 12    # Size for regular network robots in plots
SINK_MARKER_SIZE = 10     # Size for sinks in plots
SOURCE_MARKER_SIZE = 18   # Size for source in plots

# =========================
# RANDOM SEED
# =========================
random.seed(2)


# =========================
# ENUMERATIONS / ROLE CLASSES
# =========================
class Role:
    """Robot roles in the network formation algorithm."""
    MOVING = "MOVING"
    NETWORK = "NETWORK"
    SOURCE = "SOURCE"


class NodeType:
    """Node types for network robots."""
    STRAIGHT = "STRAIGHT"
    PIVOT = "PIVOT"


# =========================
# DATA CLASSES
# =========================
@dataclass
class Sink:
    """Represents a sink/target node in the network."""
    sid: int
    pos: Vec2


@dataclass
class Message:
    """Message passed between robots for communication."""
    sender_id: int
    msg_type: str
    payload: Dict[str, Any]


@dataclass
class Robot:
    """
    Robot agent in the network formation simulation.
    
    Attributes:
        rid: Unique robot identifier
        pos: Current position (pygame Vector2)
        role: Current role (MOVING, NETWORK, or SOURCE)
        node_type: Node type (STRAIGHT or PIVOT)
        parent_id: ID of parent robot in tree (None if unattached)
        children_ids: List of child robot IDs
        max_children: Maximum number of children allowed
        branch_id: Binary tree branch ID ("1" for root, "10"/"11" for pivot children, etc.)
        sink_indices: List of sink IDs assigned to this subtree
        inbox: Received messages
        outbox: Messages to send
        desired_R: Desired distance from parent
        desired_dir_d: Desired direction from parent
        void_pos_vis: Visualization position for void
        vel: Current velocity
        prev_pos: Previous position (for velocity calculation)
        guidance_lock_robot_id: ID of robot providing guidance
        guidance_lock_frames: Remaining frames to follow guidance
    """
    rid: int
    pos: Vec2

    role: str = Role.MOVING
    node_type: str = NodeType.STRAIGHT

    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    max_children: int = 1

    # Binary tree branch ID: "1" for root/pre-pivot, "10"/"11" for pivot children, etc.
    branch_id: str = "1"

    # Sinks assigned to this subtree
    sink_indices: List[int] = field(default_factory=list)

    inbox: List[Message] = field(default_factory=list)
    outbox: List[Message] = field(default_factory=list)

    desired_R: Optional[float] = None
    desired_dir_d: Optional[Vec2] = None
    void_pos_vis: Optional[Vec2] = None
    
    # Velocity estimate and previous position (used to match speeds)
    vel: Vec2 = field(default_factory=lambda: Vec2(0, 0))
    prev_pos: Vec2 = field(default_factory=lambda: Vec2(0, 0))

    # Guidance lock
    guidance_lock_robot_id: Optional[int] = None
    guidance_lock_frames: int = 0

    def __post_init__(self):
        """Initialize prev_pos to starting pos."""
        self.prev_pos = Vec2(self.pos)
        self.vel = Vec2(0, 0)

    def has_free_slot(self) -> bool:
        """Check if robot can accept more children."""
        return len(self.children_ids) < self.max_children

    def broadcast(self, msg_type: str, **payload):
        """Add a message to the outbox for broadcasting."""
        self.outbox.append(Message(self.rid, msg_type, dict(payload)))
