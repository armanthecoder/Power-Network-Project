import pygame
from pygame.math import Vector2 as Vec2

class Robot:
    def __init__(self,rid,pos:Vec2):
        self.rid=rid
        self.pos=Vec2(pos)
        self.vel=Vec2(0,0)
        self.assigned_sinks=None
        self.nodetype='straight'
        self.role='moving'
        self.dir=Vec2(0,0)



    def step (self,dt:float):
        "in each step I want to update robot step"
        self.pos+=self.vel*dt
    def draw(self,screen,font):
        "draw this robot"
        pygame.draw.circle(screen,(0,150,225),self.pos,8)
   
    def bisection_dir(self):
        "compute the bisection direction from assigned sinks"
        a=Vec2(0,0)
        if self.assigned_sinks is None or len(self.assigned_sinks)==0:
            a= Vec2(0,0)
        else:
            for s in self.assigned_sinks:
             a=a+normalize(s - self.pos)
        self.dir=normalize(a) 
class Sink:
    def __init__(self,pos:Vec2):
        self.pos=Vec2(pos)
    def draw(self,screen,font):
        "draw this sink"
        pygame.draw.circle(screen,(225,150,0),self.pos,12)
