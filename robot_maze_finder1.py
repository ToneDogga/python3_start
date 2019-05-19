#Anthony Paech's Raspbery pi / Brick Pi Lego robot project in python started 19/4/19
# by anthony paech
#
#
#!/usr/bin/env python
#
# https://www.dexterindustries.com/BrickPi/
# https://github.com/DexterInd/BrickPi3
#

from __future__ import print_function # use python 3 syntax but make it compatible with python 2
from __future__ import division       #                           ''

import math
#import pygame
#import turtle
import random
import time     # import the time library for the sleep function
#import brickpi3 # import the BrickPi3 drivers
#import pygame
import sys





class Robot:
    def __init__(self, height,width,o=["F","R","B","L"]):
        self.w = width
        self.h = height
        self.orient = o

        self.maze = []
        self.generate_maze()
        self.load_maze()


    def load_maze(self):
        fobj = open("maze.txt")
        h=0
        for line in fobj:
            length=len(line)
        #    print("y=",h," length=",length)
            for w in range(0,length-1):
                self.maze[h][w]=int(line[w:w+1])
            #    print("w=",w," h=",h,"line=",line[w:w+1])
            h+=1    
        fobj.close()



    def generate_maze(self):
    
   #     Creates a nested list to represent the game board
    
        for i in range(self.h):
            self.maze.append([0]*self.w)

           
          

    def print_maze(self):
        print("  ")
        print("------")
        for elem in self.maze:
            print(elem)
        print("------")
        print("  ")



    def generate_legal_moves(self, cur_pos):
        """
        Generates a list of legal moves for the knight to take next
        """
        possible_pos = []
        move_offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]

  # offsets for a knights tour      
  #      move_offsets = [(1, 2), (1, -2), (-1, 2), (-1, -2),
  #                      (2, 1), (2, -1), (-2, 1), (-2, -1)]

        for move in move_offsets:
            new_h = cur_pos[0] + move[0]
            new_w = cur_pos[1] + move[1]

            if (new_h >= self.h):
                continue
            elif (new_h < 0):
                continue
            elif (new_w >= self.w):
                continue
            elif (new_w < 0):
                continue
            else:
                possible_pos.append((new_h, new_w))

        return possible_pos




    
    def sort_closest_neighbors_to_goal(self, to_visit, goal):
        """
        sort the neighbours in terms of which one best advances trip to the goal
        
        """
        neighbor_list = self.generate_legal_moves(to_visit)
        empty_neighbours = []

        for neighbor in neighbor_list:
            np_value = self.maze[neighbor[0]][neighbor[1]]
            if np_value == 0:
                empty_neighbours.append(neighbor)
      #  print("goal check empty neighbours",empty_neighbours)        


        distances = []
        for empty in empty_neighbours:
      #      print("goal check empty=",empty)
            distance = [empty, 0]
            moves = self.generate_legal_moves(empty)
            for m in moves:
          #      print("goal check m=",m,"distance[1]=",distance[1],"m[0]=",m[0]," m[1]=",m[1]," goal[0]=",goal[0]," goal[1]=",goal[1],"self.maze[m[0]][m[1]]=",self.maze[m[0]][m[1]])
          #      print("m in moves m=",m," self.maze[m[0]][m[1]]=",self.maze[m[0]][m[1]])
                if self.maze[m[0]][m[1]] == 0:
                    hdist=abs(goal[0]-m[0])
                    wdist=abs(goal[1]-m[1])
                    if hdist>wdist:
                        distance[1]=hdist
                    else:
                        distance[1]=wdist
               

            distances.append(distance)


        distances_sort = sorted(distances, key = lambda s: s[1])  # find the move that reduces the distance to the goal the most
        sorted_neighbours = [s[0] for s in distances_sort]
    #    print("distances:",distances,"distances sort:",distances_sort," sorted neighbours",sorted_neighbours)
    #    input("next?")
        return sorted_neighbours
    





    def move(self, n, path, to_visit, goal):
        
       # Recursive definition of RobotPath. Inputs are as follows:
       # n = current depth of search tree
       # path = current path taken
       # to_visit = node to visit
       # goal = node to finish on
    

        self.maze[to_visit[0]][to_visit[1]] = n
        path.append(to_visit) #append the newest vertex to the current point
    #    print("step no=",n," Moving to: ", to_visit, "goal=",goal)
          #  input("next?")
        if to_visit==goal: # goal reached
            self.print_maze()
            print(path)
            print("goal reached",goal,"n=",n)
            sys.exit(1)


        if n == self.w * self.h: #if every grid is filled
            self.print_maze()
            print(path)
            print("Done!")
            sys.exit(1)

        
        #self.print_maze()
        #input("next?")

        sorted_neighbours=self.sort_closest_neighbors_to_goal(to_visit,goal)
        for neighbor in sorted_neighbours:
            self.move(n+1, path, neighbor,goal)

            #If we exit this loop, all neighbours failed so we reset
        self.maze[to_visit[0]][to_visit[1]] = 0
        try:
            path.pop()
            print("Going back to: ", path[-1])
        except IndexError:
            print("No path found")
            sys.exit(1)
        

        return False


def main():
    #Define the size of grid. We are currently solving for an 8x8 grid
  #        height is first, then width

    rp = Robot(20,20)
      
    #config_colour_sensor()

   # global start_h,start_w,gh,gw

    start_h=0
    start_w=0
    gh=19
    gw=19

    rp.print_maze()
    print("starting at=(",start_h,",",start_w,")  goal=(",gh,",",gw,") start?")
   # input()
    # recursive function starts the moving. ( move,path,starting pos, goal pos)
    rp.move(1, [], (start_h,start_w), (gh,gw))
    rp.print_maze()


main()
BP.reset_all()        # Unconfigure the sensors, disable the motors, and restore the LED to the control of the BrickPi3 firmware.







#if __name__ == '__main__':  









