import pygame
from pygame.math import Vector2 as Vec2
import Robot
from Robot import Robot,Sink


class Env:
    def __init__(self):
        self.robots:list[Robot]=[]
        self.next_id=0
        self.sinks:list[Sink]=[]
    def add_robot(self, pos: Vec2):
        r=Robot(self.next_id,pos)
        self.next_id=self.next_id+1
        self.robots.append(r)
    def step(self,dt:float):
        for r in self.robots:
           r.step(dt)
        
    def draw(self,screen,font):
        "draw everything in the environemt "
        for r in self.robots:
         r.draw(screen,font)
        for s in self.sinks:
            s.draw(screen,font)
    def add_sink(self,pos:Vec2):
        s=Sink(pos)
        self.sinks.append(s)


# assisting functions

def normalize(v:Vec2)->Vec2:
    l=v.length()
    if l>0:
        return v/l
    else:
        return Vec2(0,0)
