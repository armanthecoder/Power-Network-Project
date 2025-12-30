from __future__ import annotations 
import math 
from dataclasses import dataclass,field
from typing import List, Optional, Tuple
import numpy as np

import random 
import pygame
from pygame.math import Vector2 as Vec2

WIDTH, HEIGHT = 800, 600
FPS = 60
R   = 40.0      # desired separation
COMM_RANGE = 1.5*R     # communication radius- it should be larger than R
SPEED_MAX  = 100.0     # max robot speed (pixels per second)
PIVOT_THRESHOLD = 20  # max pivot error to accept pivot role
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



SIDE = 200.0

# Base vertex (source) somewhere reasonable on screen
SOURCE_POS = Vec2(WIDTH/2 - SIDE/2, HEIGHT/2 + SIDE/4)+Vec2(-100,100)

# Equilateral triangle: base from SOURCE_POS to SINK1_POS, apex at SINK2_POS
SINK1_POS = SOURCE_POS + Vec2(SIDE, 0.0)+Vec2(10,100)
SINK2_POS = SOURCE_POS + Vec2(SIDE/2.0,
                              - (math.sqrt(3)/2.0) * SIDE)+Vec2(10,-170)





def bisection_dir_from(point: Vec2, sink_indices: List[int]) -> Vec2:
    # this is a function that we compute the bisection direction from assigned sinks
    # inputs are the current positoion of the robot and the list of assigned sink indices
    s = Vec2(0, 0)
    for idx in sink_indices:
        h = SINK_POSITIONS[idx] - point
        if h.length_squared() > 0:
            s += h.normalize()
    if s.length_squared() == 0: 
        return Vec2(1, 0)
    return s.normalize()
# the bisection line 