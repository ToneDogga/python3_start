# tests learning python
#

import math
import time
import brickpi3
import pygame
import random
import turtle



#if __name__ == "__main__":
#    main()


# robots recusive algorithm testing
# global absolute coordinates of the robots position, posx, posy
#
# global absolute coordinates of the robots goal - goal_x, goal_y
#


global pos
pos=[[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,1,1,0,0,0,1,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,1,0,0,1],[0,0,1,0,0,0,0,1,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]


posx=0  # x position of robot
posy=0   # y position of robot
distance_travelled=0   # total distance travelled by robot

seek_call_count=0

max_x=9   # size of pos
min_x=0
max_y=9
min_y=0

global goal_x, goal_y
goal_x=5
goal_y=0

#goal_x=random.randint(0,9)
#goal_y=random.randint(0,9)
pos[goal_x][goal_y]=0



def circ(diameter):
    turtle.circle(diameter)
    turtle.getscreen()._root.mainloop()
    
   



def distance(x,y):
    return math.sqrt(x*x+y*y)

def random_move():
    return random.randint(1,4)

def valid_move(x,y):
    if x<min_x or x>max_x or y<min_y or y>max_y:
        print("invalid move")
        return 0
    else:
        return 1

def reached_goal(x,y):
    if (x==goal_x) and (y==goal_y):
        return 1

def display_pos(x,y):
    input("next?")
    print("\r")
    for rows in range(min_y,max_y+1):
        rowstring=""
        for cols in range(min_x,max_x+1):
            if (rows==y) and (cols==x):
                string="X"
            elif (rows==goal_y) and (cols==goal_x):
                string="G"
            else:    
                string=str(pos[cols][rows])
              #  print(rows,cols,string)
                
            rowstring=rowstring+string
        print(rowstring)
    
    


def calcx_from_angle(angle,dist):
    return(dist*math.cos(math.radians(90-angle)))


def calcy_from_angle(angle,dist):
    return(dist*math.cos(math.radians(angle)))


def seek(x,y,n):
    # recursive seek function test
    # x,y is current robot position, goal_x and goal_y

    display_pos(x,y)
    
    print("n=",n," current position (",x,",",y,") =",pos[x][y])

    if (x==goal_x) and (y==goal_y):
        return 1

    if valid_move(x,y):
  #  curr_x=x
  #  curr_y=y
  #  print("seek call count= ",n," x= ",x," y= ",y," goalx= ",goal_x," goaly= ",goal_y," distance to goal=",distance(goal_x-x,goal_y-y))

        toss2=random_move()
        if toss2==1 or toss2==3:   # bias to moving x wise first.  other wise bias moving Y first
            if x<goal_x:
                
                    if (pos[x+1][y]==0):
                        seek(x+1,y,n+1)
                    else:
                        print("can't move. pos[",x+1,"][",y,"]  is a wall. Stuck")
                        stuck=True
                        while stuck:
                            toss=random_move()
                            if toss==1:
                                print("random move x,y+1. x,y=",x,y)
                                if y+1<=max_y:
                                    seek(x,y+1,n+1)
                                    stuck=False
                                else:
                                    print("x random move out of range. aborted.")
                                    stuck=True
                            elif toss==2:
                                print("random move x,y-1. x,y=",x,y)
                                if y-1>=min_y:
                                    seek(x,y-1,n+1)
                                    stuck=False
                                else:
                                    print("x random move out of range. aborted.")
                                    stuck=True
                            elif toss==3:
                                print("random move x-1,y.  x,y=",x,y)
                                if x-1>=min_x:
                                    seek(x-1,y,n+1)
                                    stuck=False
                                else:
                                    print("x random move out of range. aborted.")
                                    stuck=True
                            elif toss==4:
                                print("random move x-1,y. x,y=",x,y)
                                if x-1>=min_x:
                                    seek(x-1,y,n+1)
                                    stuck=False
                                else:
                                    print("x random move out of range. aborted.")
                                    stuck=True
                                
                            else:
                                print("random error")
                                stuck=False
                
            elif x>goal_x:
                
                    if (pos[x-1][y]==0):
                        seek(x-1,y,n+1)
                    else:
                        print("can't move. pos[",x-1,"][",y,"] is a wall. stuck.")
                        stuck=True
                        while stuck:
                            toss=random_move()
                            if toss==1:
                                print("random move x,y+1. x,y=",x,y)
                                if y+1<=max_y:
                                    seek(x,y+1,n+1)
                                    stuck=False
                                else:
                                    print("x Random move out of range. abort.")
                                    stuck=True
                            elif toss==2:
                                print("random move x,y-1. x,y=",x,y)
                                if y-1>=min_y:
                                    seek(x,y-1,n+1)
                                    stuck=False
                                else:
                                    print("x random move out of range. abort.")
                                    stuck=True
                            elif toss==3: 
                                print("random move x+1,y.  x,y=",x,y)
                                if x+1<=max_x:
                                    seek(x+1,y,n+1)
                                    stuck=False
                                else:
                                    print("x random move out of range. abort.")
                                    stuck=True
                            elif toss==4:
                                print("random move x+1,y.  x,y=",x,y)
                                if x+1<=max_x:
                                    seek(x+1,y,n+1)
                                    stuck=False
                                else:
                                    print("x random move out of range. abort.")
                                    stuck=True
                            else:    
                                print("random error")
                                stuck=False
                            
            else:   # x==goal_x
                     if (pos[x][y+1]==0):
                        seek(x,y+1,n+1)
                     else:
                        print("can't move. pos[",x,"][",y+1,"]  is a wall. stuck")
                        stuck=True
                        while stuck:
                            toss=random_move()
                            if toss==1:
                                print("random move x+1,y. x,y=",x,y)
                                if x+1<=max_x:
                                    seek(x+1,y,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            elif toss==2:
                                print("random move x-1,y. x,y=",x,y)
                                if x-1>=min_x:
                                    seek(x-1,y,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            elif toss==3:
                                print("random move x,y-1.  x,y=",x,y)
                                if y-1>=min_y:
                                    seek(x,y-1,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            elif toss==4:
                                print("random move y-1,y.  x,y=",x,y)
                                if y-1>=min_y:
                                    seek(x,y-1,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            else:
                                print("Random error")
                                stuck=False


               
                
            
        else:    
            if y<goal_y:
                
                    if (pos[x][y+1]==0):
                        seek(x,y+1,n+1)
                    else:
                        print("can't move. pos[",x,"][",y+1,"]  is a wall. stuck")
                        stuck=True
                        while stuck:
                            toss=random_move()
                            if toss==1:
                                print("random move x,y-1. x,y=",x,y)
                                if y-1>=min_y:
                                    seek(x,y-1,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            elif toss==2:
                                print("random move x,y-1. x,y=",x,y)
                                if y-1>=min_y:
                                    seek(x,y-1,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            elif toss==3:   
                                print("random move x+1,y. x,y=",x,y)
                                if x+1<=max_x:
                                    seek(x+1,y,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            elif toss==4:
                                print("random move x-1,y.  x,y=",x,y)
                                if x-1>=min_x:
                                    seek(x-1,y,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            else:    
                                print("random error")
                                stuck=False
                            
            
            elif y>goal_y:
            
                    if (pos[x][y-1]==0):
                        seek(x,y-1,n+1)
                    else:
                        print("can't move. pos[",x,"][",y-1,"]  is a wall. stuck")
                        stuck=True
                        while stuck:
                            toss=random_move()
                            if toss==1:
                                print("random move x,y+1. x,y=",x,y)
                                if y+1<=max_y:
                                    seek(x,y+1,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            elif toss==2:
                                print("random move x,y+1. x,y=",x,y)
                                if y+1<=max_y:
                                    seek(x,y+1,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            elif toss==3:
                                print("random move x+1,y.  x,y=",x,y)
                                if x+1<=max_x:
                                    seek(x+1,y,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            elif toss==4:
                                print("random move x-1,y.  x,y=",x,y)
                                if x-1>=min_x:
                                    seek(x-1,y,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            else:
                                print("Random error")
                                stuck=False

            else:  # y==goal_y                          
                     if (pos[x+1][y]==0):
                        seek(x+1,y,n+1)
                     else:
                        print("can't move. pos[",x+1,"][",y,"]  is a wall. stuck")
                        stuck=True
                        while stuck:
                            toss=random_move()
                            if toss==1:
                                print("random move x,y+1. x,y=",x,y)
                                if y+1<=max_y:
                                    seek(x,y+1,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            elif toss==2:
                                print("random move x,y-1. x,y=",x,y)
                                if y-1>=min_y:
                                    seek(x,y-1,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            elif toss==3:
                                print("random move x-1,y.  x,y=",x,y)
                                if x-1>=min_x:
                                    seek(x-1,y,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            elif toss==4:
                                print("random move x-1,y.  x,y=",x,y)
                                if x-1>=min_x:
                                    seek(x-1,y,n+1)
                                    stuck=False
                                else:
                                    print("y random move out of range. abort.")
                                    stuck=True
                            else:
                                print("Random error")
                                stuck=False
                        
    else:
        print("invalid move x=",x,y)
        


    

def main():
   # print(pos)
 #  while True:
        print("starting position=(",posx,",",posy,") - goal=(",goal_x,",",goal_y,")")
        if seek(posx,posy,1):
            print("goal found.")
        
  #  for angle in range(0,90,1):
  #      d=distance(posx,posy)
  #      print("posx= ",posx,"posy= ",posy," dist= ",distance(posx,posy),"angle= ",angle," angle x= ",calcx_from_angle(angle,distance(posx,posy))," angle y= ",calcy_from_angle(angle,distance(posx,posy)))
  #      posx=posx+1
    # for i in range(1,10):
     #   print(random.randint(1,15))
        
    # print(math.pi)
    # circ(50)
    
    
main()





