#Anthony Paech's Raspbery pi / Brick Pi Lego robot project in python started 19/4/19
# by anthony paech
#
#   Goal :  build a tracked vehicle controlled by a python code
#
# stage 1 a programmable robot that can follow instructions
# stage 2 is stage 1 that can stop if it hits an object
#  stage 3 is stage 2 that can see obstacals
# stage 4 is where it can navigate across a pool table dodging balls.
# stage 5 is where it is given a high level goal and it finds its way there
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
import brickpi3 # import the BrickPi3 drivers
import pygame
import sys


BP = brickpi3.BrickPi3() # Create an instance of the BrickPi3 class. BP will be the BrickPi3 object.
#move_size=180  # global variable forwards or backwards move.  this move size translates to 25m by 50 moves on the snooker table 
move_size=340
#turn_size=485 # global variable of motor degrees movement to turn the robot 90 degrees
turn_size=475
power_percent=50
speed_limit=250   # in degrees per second
wait_time=1.8 # seconds to wait after a move
colour = ["none", "Black", "Blue", "Green", "Yellow", "Red", "White", "Brown"]
ultra_safety_dist=2.5  # ultra sonic sensor miniumn distance in cm

# main logic
# motor functions
# define a motor reset function with no arguments
# define a stop function with no arguments
# define a forward function with the degrees of motor travel as an arg
# define a back function with the degrees of motor travel as an arg
# define a turn right function
# define a turn left function
# define a motor power stop function with no args
#
#
#
#
#
#
# sensor functions
#
#
#
#
#
#
#
#
# define a simple input function for robot commands with a string of commands
# F,B,L,R
# define a simple parser to check validity of the string
# define a main() loop
#   Enter robot commands string
#    parse command
#   if valid
#        execute commands
#
#   else
#        print invalid commands
#    
#
#   execute commands
#       reset motors
#       for command in range
#           case command
#       stop
#       motor power stop
#
# tests learning python
#



class Robot:
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
            curr_h=start_h
            curr_w=start_w
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
            move_instruction="B"
        elif new_w-curr_w==1:
               # move left
            move_instruction="LFR"
        elif new_w-curr_w==-1:
               # move right
            move_instruction="RFL"
        else:
               # instruction error
            print("instuction error path[-1]->to_visit")


        print("move_instruction=",move_instruction)
        self.execute_command(move_instruction)
        return True
        #else:
        #    return False
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
          #  input("next?")
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
        return False






    def enter_command(self):
        cstring=""
        while (len(cstring)==0):
            cstring=input("Enter Robot Command string: ")

  #  print("cstring= ",cstring)
  #  for letter in cstring:
  #      print(letter)

        return cstring.upper()


        
    def validate_command(self,cstr):

        flag=1
        print("validate command:",cstr)
        if cstr=="":
            flag=0
        else:    
            for l in cstr:
   #     print(l)
                if l=="L" or l=="R" or l=="F" or l=="B" or l=="T":
                    if flag==1:
                        flag=1
                    else:
                        flag=0
    #        print("flag=",flag)
                else:
                    flag=0
     #       print("flag=",flag)

        if flag:
            print("command string: ",cstr," is valid.")
        else:
            print("command string: ",cstr," is invalid.")

   
        return flag

    

    def motorA_reset(self):
        try:
            BP.offset_motor_encoder(BP.PORT_A, BP.get_motor_encoder(BP.PORT_A)) # reset encoder
            print("Motor A reset encoder done")
        except IOError as error:
            print(error)

        try:
            BP.set_motor_power(BP.PORT_A, BP.MOTOR_FLOAT)    # float motor A
            print("Motor A floated")
        except IOError as error:
            print(error)
        
        try:
            BP.set_motor_limits(BP.PORT_A, power_percent, speed_limit)          # optionally set a power limit (in percent) and a speed limit (in Degrees Per Second)
        except IOError as error:
            print(error)

    

    def motorB_reset(self):
        try:
            BP.offset_motor_encoder(BP.PORT_B, BP.get_motor_encoder(BP.PORT_B)) # reset encoder
            print("Motor B reset encoder done")
        except IOError as error:
            print(error)

        try:
            BP.set_motor_power(BP.PORT_B, BP.MOTOR_FLOAT)    # float motor B
            print("Motor B floated")
        except IOError as error:
            print(error)

        try:
            BP.set_motor_limits(BP.PORT_B, power_percent, speed_limit)          # optionally set a power limit (in percent) and a speed limit (in Degrees Per Second)
        except IOError as error:
            print(error)

    

    def motorC_reset(self):
        try:
            BP.offset_motor_encoder(BP.PORT_C, BP.get_motor_encoder(BP.PORT_C)) # reset encoder
            print("Motor C reset encoder done")
        except IOError as error:
            print(error)

        try:
            BP.set_motor_power(BP.PORT_C, BP.MOTOR_FLOAT)    # float motor C
            print("Motor C floated")
        except IOError as error:
            print(error)
        
        try:
            BP.set_motor_limits(BP.PORT_C, power_percent, speed_limit)          # optionally set a power limit (in percent) and a speed limit (in Degrees Per Second)
        except IOError as error:
            print(error)

    


    def motorD_reset(self):
        try:
            BP.offset_motor_encoder(BP.PORT_D, BP.get_motor_encoder(BP.PORT_D)) # reset encoder
            print("Motor D reset encoder done")
        except IOError as error:
            print(error)

        try:
            BP.set_motor_power(BP.PORT_D, BP.MOTOR_FLOAT)    # float motor D
            print("Motor D floated")
        except IOError as error:
            print(error)
        
        try:
            BP.set_motor_limits(BP.PORT_D, power_percent, speed_limit)          # optionally set a power limit (in percent) and a speed limit (in Degrees Per Second)
        except IOError as error:
            print(error)



    def turn_left90(self):
        print("turn_left90")
        try:
            target=BP.get_motor_encoder(BP.PORT_B)
            print("motor B encoder value before move: ",target, "calculated pos SB: ", target+(turn_size)) 
            BP.set_motor_position(BP.PORT_B, target+(turn_size))    # set motor B's target position
   
        except IOError as error:
            print(error)

 #   time.sleep(3)

    
     
        try:
            target=BP.get_motor_encoder(BP.PORT_C)
            print("motor C encoder value before move: ",target, "calculated pos SB: ", target-(turn_size)) 
            BP.set_motor_position(BP.PORT_C, target-(turn_size))    # set motor C's target position 
        except IOError as error:
            print(error)

        time.sleep(wait_time)

        print("motor B encoder value after move = ", BP.get_motor_encoder(BP.PORT_B))
        print("motor C encoder value after move = ", BP.get_motor_encoder(BP.PORT_C))


    #time.sleep(wait_time)


    def turn_right90(self):
        print("turn_right 90")
        try:
            target=BP.get_motor_encoder(BP.PORT_B)
            print("motor B encoder value before move: ",target, "calculated pos SB: ", target-(turn_size)) 
            BP.set_motor_position(BP.PORT_B, target-(turn_size))    # set motor B's target position
   
        except IOError as error:
            print(error)

   # time.sleep(1)

    
     
        try:
            target=BP.get_motor_encoder(BP.PORT_C)
            print("motor C encoder value before move: ",target, "calculated pos SB: ", target+(turn_size)) 
            BP.set_motor_position(BP.PORT_C, target+(turn_size))    # set motor C's target position 
        except IOError as error:
            print(error)

        time.sleep(wait_time)

        print("motor B encoder value after move = ", BP.get_motor_encoder(BP.PORT_B))
        print("motor C encoder value after move = ", BP.get_motor_encoder(BP.PORT_C))

    #time.sleep(wait_time)


    def turn_angle(angle):
        print("turn ",angle," degrees")
        angle_move=angle*(turn_size/90)
        try:
            target=BP.get_motor_encoder(BP.PORT_B)
            print("motor B encoder value before move: ",target, "calculated pos SB: ", target-(angle_move)) 
            BP.set_motor_position(BP.PORT_B, target-(angle_move))    # set motor B's target position
   
        except IOError as error:
            print(error)

   # time.sleep(1)

    
     
        try:
            target=BP.get_motor_encoder(BP.PORT_C)
            print("motor C encoder value before move: ",target, "calculated pos SB: ", target+(angle_move)) 
            BP.set_motor_position(BP.PORT_C, target+(angle_move))    # set motor C's target position 
        except IOError as error:
            print(error)

        time.sleep(wait_time)

        print("motor B encoder value after move = ", BP.get_motor_encoder(BP.PORT_B))
        print("motor C encoder value after move = ", BP.get_motor_encoder(BP.PORT_C))

    #time.sleep(wait_time)


    def move_forward(self):
        print("move_forward")
# forward movement distance and speed is set by global variables
#            print("Motor B target: %6d  Motor B position: %6d" % (target, BP.get_motor_encoder(BP.PORT_B)))
#        except IOError as error:
#            print(error)
        try:
            target=BP.get_motor_encoder(BP.PORT_B)
            print("motor B encoder value before move: ",target, "calculated pos SB: ", target+(move_size))
        #if not read_touch_sensor():
            BP.set_motor_position(BP.PORT_B, target+move_size)    # set motor B's target position
        #else:
        #    print("touch sensor triggered")
        except IOError as error:
            print(error)

        try:
            target=BP.get_motor_encoder(BP.PORT_C)
            print("motor C encoder value before move: ",target, "calculated pos SB: ", target+(move_size))
       # if not read_touch_sensor():
            BP.set_motor_position(BP.PORT_C, target+move_size)    # set motor C's target position
       # else:
        #    print("touch sensor triggered")
        except IOError as error:
            print(error)

        time.sleep(wait_time)

        print("motor B encoder value after move = ", BP.get_motor_encoder(BP.PORT_B))
        print("motor C encoder value after move = ", BP.get_motor_encoder(BP.PORT_C))

    #time.sleep(wait_time)

        

    def move_backward(self):
        print("Move_backward")    
# backward movement distance and speed is set by global variables

        try:
            target=BP.get_motor_encoder(BP.PORT_B)
            print("motor B encoder value before move: ",target, "calculated pos SB: ", target-(move_size)) 
            BP.set_motor_position(BP.PORT_B, target-move_size)    # set motor B's target position 
        except IOError as error:
            print(error)

        try:
            target=BP.get_motor_encoder(BP.PORT_C)
            print("motor C encoder value before move: ",target, "calculated pos SB: ", target-(move_size)) 
            BP.set_motor_position(BP.PORT_C, target-move_size)    # set motor C's target position 
        except IOError as error:
            print(error)

        time.sleep(wait_time)

        print("motor B encoder value after move = ", BP.get_motor_encoder(BP.PORT_B))
        print("motor C encoder value after move = ", BP.get_motor_encoder(BP.PORT_C))



    #time.sleep(wait_time)


    def config_touch_sensor(self):
# Configure for a touch sensor.
# If an EV3 touch sensor is connected, it will be configured for EV3 touch, otherwise it's configured for NXT touch.
# BP.set_sensor_type configures the BrickPi3 for a specific sensor.
# BP.PORT_1 specifies that the sensor will be on sensor port 1.
# BP.SENSOR_TYPE.TOUCH specifies that the sensor will be a touch sensor.
        print("config touch sensor")
        BP.set_sensor_type(BP.PORT_1, BP.SENSOR_TYPE.TOUCH)




    def read_touch_sensor(self):
 # read and display the sensor value
        # BP.get_sensor retrieves a sensor value.
        # BP.PORT_1 specifies that we are looking for the value of sensor port 1.

        # BP.get_sensor returns the sensor value (what we want to display).
        value=0
        try:
            value = BP.get_sensor(BP.PORT_1)
            print("Read touch sensor. value= ",value)
        except brickpi3.SensorError as error:
            print(error)   
        return value


    def config_colour_sensor(self):
    # Configure for an EV3 color sensor.
    # BP.set_sensor_type configures the BrickPi3 for a specific sensor.
    # BP.PORT_3 specifies that the sensor will be on sensor port 3.
    # BP.Sensor_TYPE.EV3_COLOR_REFLECTED specifies that the sensor will be an ev3 color sensor.
        BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_REFLECTED)

# BP.set_sensor_type(BP.PORT_1, BP.SENSOR_TYPE.EV3_COLOR_COLOR)


    
        # BP.get_sensor retrieves a sensor value.
        # BP.PORT_3 specifies that we are looking for the value of sensor port 3.
        # BP.get_sensor returns the sensor value (what we want to display).
        try:
            BP.get_sensor(BP.PORT_3)
        except brickpi3.SensorError:
            print("Configuring colour sensor...")
            error = True
            while error:
                time.sleep(0.1)
                try:
                    BP.get_sensor(BP.PORT_3)
                    error = False
                except brickpi3.SensorError:
                    error = True
            print("Configured.")




    def read_colour_sensor(self):    
        # read and display the sensor value
        # BP.get_sensor retrieves a sensor value.
        # BP.PORT_3 specifies that we are looking for the value of sensor port 3.
        # BP.get_sensor returns the sensor value (what we want to display).
        config_colour_sensor()
        print("read colour sensor")
        try:
            BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_REFLECTED)        # Configure for an EV3 color sensor in reflected mode.
            time.sleep(0.02)
            value1 = BP.get_sensor(BP.PORT_3)                                           # get the sensor value
            print("value1= ",value1,"reflected mode colour=",colour[value1])
            BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_AMBIENT)          # Configure for an EV3 color sensor in ambient mode.
            time.sleep(0.02)
            value2 = BP.get_sensor(BP.PORT_3)                                        # get the sensor value
            print("value2= ",value2,"ambient mode colour=")        
            BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_COLOR)            # Configure for an EV3 color sensor in color mode.
            time.sleep(0.02)
            value3 = BP.get_sensor(BP.PORT_3)                                        # get the sensor value
            print("value3= ",value3,"colour colour mode colour=",colour[value3])       
            BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_COLOR_COMPONENTS) # Configure for an EV3 color sensor in color components mode.
            time.sleep(0.02)
            value4 = BP.get_sensor(BP.PORT_3)                                        # get the sensor value
            print("value4= ",value4,"colour components mode colour=")

#        value = BP.get_sensor(BP.PORT_3)
      #  print(value4, colour[value4])                # print the color
        except brickpi3.SensorError as error:
            print(error)
        
     #   time.sleep(0.02)  # delay for 0.02 seconds (20ms) to reduce the Raspberry Pi CPU load.


#    try:
#        BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_REFLECTED)        # Configure for an EV3 color sensor in reflected mode.
#        time.sleep(0.02)
#        value1 = BP.get_sensor(BP.PORT_3)                                        # get the sensor value
#        BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_AMBIENT)          # Configure for an EV3 color sensor in ambient mode.
#        time.sleep(0.02)
#        value2 = BP.get_sensor(BP.PORT_3)                                        # get the sensor value
#        BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_COLOR)            # Configure for an EV3 color sensor in color mode.
#        time.sleep(0.02)
#        value3 = BP.get_sensor(BP.PORT_3)                                        # get the sensor value
#        BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_COLOR_COMPONENTS) # Configure for an EV3 color sensor in color components mode.
#        time.sleep(0.02)
#        value4 = BP.get_sensor(BP.PORT_3)                                        # get the sensor value
#        print("colour sensor readings - ",value1, "   ", value2, "   ", value3, "   ", value4)
              # print the color sensor values
#    except brickpi3.SensorError as error:
#        print(error)

    def config_ultrasonic_sensor(self):
# Configure for an EV3 color sensor.
# BP.set_sensor_type configures the BrickPi3 for a specific sensor.
# BP.PORT_4 specifies that the sensor will be on sensor port 4.
# BP.Sensor_TYPE.EV3_ULTRASONIC_CM specifies that the sensor will be an EV3 ultrasonic sensor.
        print("config ultra sonic sensor")
        BP.set_sensor_type(BP.PORT_4, BP.SENSOR_TYPE.EV3_ULTRASONIC_CM) # Configure for an EV3 ultrasonic sensor.



    def read_ultrasonic_sensor(self):
        # BP.get_sensor retrieves a sensor value.
        # BP.PORT_4 specifies that we are looking for the value of sensor port 4.
        # BP.get_sensor returns the sensor value (what we want to display).
        value=0
        try:
            value = BP.get_sensor(BP.PORT_4)
            print("read ultra sonic sensor value= ",value)                         # print the distance in CM
        except brickpi3.SensorError as error:
            print(error)

        return value


    
    def execute_command(self,com_str):
        time.sleep(wait_time)
        miss_command=True
        if self.validate_command(com_str):
            c_count=1
            print("execute commands: ", com_str)
            for l in com_str:
               
                time.sleep(0.05)  # delay for 0.05 seconds (50ms) to reduce the Raspberry Pi CPU load.

                print(c_count," - execute command: ",l)

                miss_command=False
                touch=self.read_touch_sensor()
                if touch:
                    print("touch sensor triggered")
                    print("forward off touch")
                    move_forward()
                    turn_forward_count=turn_forward_count+1
                    miss_command=True
                distance=self.read_ultrasonic_sensor()
                print("ultra sonic sensor dist= ", distance)
                if distance<ultra_safety_dist:  # distance to stop
                    print("ultra sonic sensor triggered. distance= ", distance)
                    print("backward off ultra sonic")
                
                    miss_command=True

  
       
            
                if not miss_command:
                    if l=="L":
                        print("turn left 90")
                        self.turn_left90()
                    elif l=="R":
                        print("turn right 90")      
                        self.turn_right90()
                    elif l=="F":
                        print("Forward")
                        self.move_forward()
                    elif l=="B":
                        print("backward")      
                        self.move_backward()
                    elif l=="T":
                        print("turn 45 degrees")
                        self.turn_angle(45)
                    else:
                        print("Invalid command= ",l)
                    c_count=c_count+1
                else:
                    print("command ",l," ignored, touch sensor or ultra sonic sensor triggered")
                  
            print(c_count-1," :Command(s) completed")
        return not miss_command
    
    




def main():
    #Define the size of grid. We are currently solving for an 8x8 grid
  #        height is first, then width

    rp = Robot(40,20)
      
    #config_colour_sensor()
    rp.config_touch_sensor()
    rp.config_ultrasonic_sensor()
    rp.motorB_reset()
    rp.motorC_reset()

    global start_h,start_w,gh,gw

    start_h=1
    start_w=1
    gh=1
    gw=10

    rp.print_maze()
    print("starting at=(",start_h,",",start_w,")  goal=(",gh,",",gw,") start?")
    input()
    # recursive function starts the moving. ( move,path,starting pos, goal pos)
    rp.move(1, [], (start_h,start_w), (gh,gw))
    rp.print_maze()

 #   try:
 #       while True:
 #           command_string=enter_command()
 #           if validate_command(command_string):
#              motorB_reset()
#               motorC_reset()
#              execute_command(command_string)
#           else:
#               print("command invalid")
    
#   except KeyboardInterrupt:
#       BP.reset_all()            

main()
BP.reset_all()        # Unconfigure the sensors, disable the motors, and restore the LED to the control of the BrickPi3 firmware.







#if __name__ == '__main__':  









