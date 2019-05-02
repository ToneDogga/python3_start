# tests learning python
#

import math
import time
import brickpi3
import pygame
import random
import turtle








"""
This code generates the path required for a simulated robot to move through a maze with user-specified dimensions


"""
import sys

class RobotPath:
    def __init__(self, width, height):
        self.w = width
        self.h = height

        self.maze = []
        self.generate_maze()
       # self.load_maze()


    def load_maze(self):
        fobj = open("maze.txt")
        row=0
        for line in fobj:
            length=len(line)
            for n in range(0,length-1):
                self.maze[row][n]=int(line[n:n+1])
               # print(line[n:n+1])
            row+=1    
        fobj.close()



    def generate_maze(self):
    
   #     Creates a nested list to represent the game board
    
        for i in range(self.h):
            self.maze.append([0]*self.w)

        self.load_maze()


    #        for i in range(1,2):
   #         x=random.randint(0,self.w-1)
    #        y=random.randint(0,self.h-1)
        #    print("generate maze:,x,y=",x,y,i)
     #       self.maze[x][y]=1   # put 1's in the maze of zeros to block the path
            
          

    def print_maze(self):
        print("  ")
        print("------")
        for elem in self.maze:
            #print("elem=",elem)
            #if elem==goal:
            #    print("elem==goal")
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
            new_x = cur_pos[0] + move[0]
            new_y = cur_pos[1] + move[1]

            if (new_x >= self.h):
                continue
            elif (new_x < 0):
                continue
            elif (new_y >= self.w):
                continue
            elif (new_y < 0):
                continue
            else:
                possible_pos.append((new_x, new_y))

        return possible_pos

    def calc_distance_to_goal(self,to_visit,goal):
        xdist=goal[0]-to_visit[0]
        ydist=goal[1]-to_visit[1]
        #print("calc dist xdist=",xdist," ydist=",ydist)
        #dist=math.sqrt(xdist*xdist+ydist*ydist)
        #print("round dist=",int(dist))
        return round(xdist)



    
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
                    xdist=abs(goal[0]-m[0])
                    ydist=abs(goal[1]-m[1])
        #            rand=random.randint(1,3)
        #            if rand==range(1,2):
                    if xdist>ydist:
                        distance[1]=xdist
                    else:
                        distance[1]=ydist
      #                  if ydist>xdist:
      #                      distance[1]=ydist
      #                  else:
      #                      distance[1]=xdist
    #             else:
    #                   if xdist>ydist:
    #                       distance[1]=ydist
    #                   else:
    #                       distance[1]=xdist
      #                  if xdist>ydist:
      #                       distance[1]=ydist
      #                  else:
      #                      distance[1]=xdist
                

            distances.append(distance)


        distances_sort = sorted(distances, key = lambda s: s[1])  # find the move that reduces the distance to the goal the most
        sorted_neighbours = [s[0] for s in distances_sort]
    #    print("distances:",distances,"distances sort:",distances_sort," sorted neighbours",sorted_neighbours)
    #    input("next?")
        return sorted_neighbours
    





    def move(self, n, path, to_visit, goal):
        """
        Recursive definition of RobotPath. Inputs are as follows:
        n = current depth of search tree
        path = current path taken
        to_visit = node to visit
        goal = node to finish on
        """
        self.maze[to_visit[0]][to_visit[1]] = n
        path.append(to_visit) #append the newest vertex to the current point
        print("step no=",n," Moving to: ", to_visit, "goal=",goal)

        if to_visit==goal: # goal reached
            self.print_maze()
            print(path)
            print("goal reached",goal)
            sys.exit(1)


        if n == self.w * self.h: #if every grid is filled
            self.print_maze()
            print(path)
            print("Done!")
            sys.exit(1)

        else:
            if not n%25:               # every 25 steps stop and display progress
                self.print_maze()
                input("next?")
           # sorted_neighbours = self.sort_lonely_neighbors(to_visit)
            sorted_neighbours=self.sort_closest_neighbors_to_goal(to_visit,goal)
        #    print("to visit",to_visit," after checking goal")
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






if __name__ == '__main__':  
    #Define the size of grid. We are currently solving for an 8x8 grid
  #        
    rp = RobotPath(25,50)
    rp.print_maze()
    input("next?")
    rp.move(1, [], (1,1), (14,44))
    rp.print_maze()








