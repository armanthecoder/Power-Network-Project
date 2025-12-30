import pygame
from pygame.math import Vector2 as Vec2
import pygame
from env import Env    # this uses the Env class from env.py


# assisting functions

def normalize(v:Vec2)->Vec2:
    l=v.length()
    if l>0:
        return v/l
    else:
        return Vec2(0,0)

        


    
def main():
    pygame.init()
    screen_width=800
    screen_height=600
    screen=pygame.display.set_mode((screen_width,screen_height))
    pygame .display.set_caption("Env+Robot example")
    clock=pygame.time.Clock()
    print(clock)
    font = pygame.font.SysFont(None, 24)
    env=Env()
    env.add_robot(Vec2(100,300))
    env.add_robot(Vec2(200,300))
    env.add_sink(Vec2(300,300))
    running=True
    a=0
    while running:
        dt=clock.tick(60)/1000
        a=dt+a
        print(f"{a:.5f}")
        for event in pygame.event.get():
            print(event)
            if event.type==pygame.QUIT:
                 running=False
        
         
        env.step(dt)
        screen.fill((20,20,20))
        env.draw(screen,font)
        pygame.display.flip()

    pygame.quit()


if __name__=="__main__":
      main()