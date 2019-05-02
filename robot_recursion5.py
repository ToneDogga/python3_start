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
    def __init__(self, height,width):
        self.w = width
        self.h = height

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

      #  self.load_maze()


    #        for i in range(1,2):
   #         x=random.randint(0,self.w-1)
    #        y=random.randint(0,self.h-1)
        #    print("generate maze:,x,y=",x,y,i)
     #       self.maze[x][y]=1   # put 1's in the maze of zeros to block the path
            
          

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

    def calc_distance_to_goal(self,to_visit,goal):
        hdist=goal[0]-to_visit[0]
        wdist=goal[1]-to_visit[1]
        #print("calc dist xdist=",xdist," ydist=",ydist)
        #dist=math.sqrt(xdist*xdist+ydist*ydist)
        #print("round dist=",int(dist))
        return round(hdist)


    def physical_move(instruction_string):    



    def translate_to_physical_move(self,n,path,to_visit):
        # the physical robot can only move 4 ways
        # F - forward    
        # B - back      
        # L - turn left  
        # R - turn right
        #
        # to_visit[0]+1 == forward
        # to_visit[0]-1 == backward
        # to_visit[1]+1 == turn left, forward, turn right
        # to_visit[1]-1 === turn right, forward, turn left
        #
       #
        move_instruction=""
        print("physical move to ",to_visit)
        if n<2:
            curr_h=0
            curr_w=0
        else:    
            curr_h=path[-1][0]
            curr_w=path[-1][1]   
            print("n=",n,"current position before move is path[-1]=",path[-1])
            
        print("curr_h=",curr_h," curr_w=",curr_w)
        new_h=to_visit[0]
        new_w=to_visit[1]
            #
        print("new_h=",new_h," new_w=",new_w)
           #
        if new_h-curr_h==1:
               # move forward 1
            move_instruction="F"
        elif new_h-curr_h==-1:
               # move back 1
            move instruction="B"
        elif new_w-curr_w==1:
               # move left
            move_instruction="LFR"
        elif new_w-curr_w==-1:
               # move right
            move_instruction="RFL"
        else:
               # instrucion error
            print("instuction error path[-1]->to_visit")


        print("move_instruction=",move_instruction)
        input("next?")
        if physical_move(move_instruction):
            return True
        else:
            return False
        # return true is move successful
        

    
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
        """
        Recursive definition of RobotPath. Inputs are as follows:
        n = current depth of search tree
        path = current path taken
        to_visit = node to visit
        goal = node to finish on
    

        
        """
       # height is the [0] parameter, width is the [1]

        # time to translate move into robot language
        if self.translate_to_physical_move(n,path,to_visit):


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
        else:
            print("Robot physical move failed trying to get to ",to_visit)





if __name__ == '__main__':  
    #Define the size of grid. We are currently solving for an 8x8 grid
  #        height is first, then width
    rp = RobotPath(50,25)
    rp.print_maze()
    input("next?")
    rp.move(1, [], (1,12), (44,14))
    rp.print_maze()








