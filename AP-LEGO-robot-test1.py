# Anthony Paech's Raspbery pi / Brick Pi Lego robot project in python started 19/4/19
#
#
#   Goal :  build a tracked vehicle using the lego power function motor
# with a dual differential system and a stepper motor controlling the steering
# it uses proximity sensors to avoid collisions
#  Can I use the Ir control from python?
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

BP = brickpi3.BrickPi3() # Create an instance of the BrickPi3 class. BP will be the BrickPi3 object.
move_size=360  # global variable forwards or backwards move 
turn_size=485 # global variable of motor degrees movement to turn the robot 90 degrees
power_percent=50
speed_limit=250   # in degrees per second
wait_time=2.5  # seconds to wait after a move

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



def enter_command():
    cstring=""
    while (len(cstring)==0):
        cstring=input("Enter Robot Command string: ")

  #  print("cstring= ",cstring)
  #  for letter in cstring:
  #      print(letter)

    return cstring.upper()


        
def validate_command(cstr):

    flag=1
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

    

def motorA_reset():
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

    

def motorB_reset():
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

    

def motorC_reset():
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

    


def motorD_reset():
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



def turn_left90():
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


def turn_right90():
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


def move_forward():
    print("move_forward")
# forward movement distance and speed is set by global variables
#            print("Motor B target: %6d  Motor B position: %6d" % (target, BP.get_motor_encoder(BP.PORT_B)))
#        except IOError as error:
#            print(error)
    try:
        target=BP.get_motor_encoder(BP.PORT_B)
        print("motor B encoder value before move: ",target, "calculated pos SB: ", target+(move_size)) 
        BP.set_motor_position(BP.PORT_B, target+move_size)    # set motor B's target position 
    except IOError as error:
        print(error)

    try:
        target=BP.get_motor_encoder(BP.PORT_C)
        print("motor C encoder value before move: ",target, "calculated pos SB: ", target+(move_size))
        BP.set_motor_position(BP.PORT_C, target+move_size)    # set motor C's target position 
    except IOError as error:
        print(error)

    time.sleep(wait_time)

    print("motor B encoder value after move = ", BP.get_motor_encoder(BP.PORT_B))
    print("motor C encoder value after move = ", BP.get_motor_encoder(BP.PORT_C))

    #time.sleep(wait_time)

        

def move_backward():
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


    
def execute_command(com_str):
    c_count=1
    print("execute commands: ", com_str)
    for l in com_str:
               
        time.sleep(0.05)  # delay for 0.05 seconds (50ms) to reduce the Raspberry Pi CPU load.

        print(c_count," - execute command: ",l)
        if l=="L":
            print("turn left 90")
            turn_left90()
        elif l=="R":
            print("turn right 90")      
            turn_right90()
        elif l=="F":
            print("Forward")
            move_forward()
        elif l=="B":
            print("backward")      
            move_backward()
        elif l=="T":
            print("turn 45 degrees")
            turn_angle(45)
        else:
            print("Invalid command= ",l)
        c_count=c_count+1
                  
    print(c_count-1," :Commands completed")
    
    




#    while True:
        # Each of the following BP.get_motor_encoder functions returns the encoder value.
#        try:
#            target = BP.get_motor_encoder(BP.PORT_C) # read motor C's position
#        except IOError as error:
#            print(error)
        
#        BP.set_motor_position(BP.PORT_B, target)    # set motor B's target position to the current position of motor D
        
#        try:
#            print("Motor B target: %6d  Motor B position: %6d" % (target, BP.get_motor_encoder(BP.PORT_B)))
#        except IOError as error:
#            print(error)
        
 #       time.sleep(0.02)  # delay for 0.02 seconds (20ms) to reduce the Raspberry Pi CPU load.


# This code is an example for reading an EV3 infrared sensor connected to PORT_1 of the BrickPi3
# 
# Hardware: Connect an EV3 infrared sensor to BrickPi3 sensor port 1.
# 
# Results:  When you run this program, the infrared remote status will be printed.


# Configure for an EV3 color sensor.
# BP.set_sensor_type configures the BrickPi3 for a specific sensor.
# BP.PORT_1 specifies that the sensor will be on sensor port 1.
# BP.Sensor_TYPE.EV3_INFRARED_REMOTE specifies that the sensor will be an EV3 infrared sensor.
#BP.set_sensor_type(BP.PORT_1, BP.SENSOR_TYPE.EV3_INFRARED_REMOTE)

#try:
#    while True:
        # BP.get_sensor retrieves a sensor value.
        # BP.PORT_1 specifies that we are looking for the value of sensor port 1.
        # BP.get_sensor returns the sensor value (what we want to display).
#        try:
#            print(BP.get_sensor(BP.PORT_1))   # print the infrared values
#        except brickpi3.SensorError as error:
#            print(error)
        
#        time.sleep(0.02)  # delay for 0.02 seconds (20ms) to reduce the Raspberry Pi CPU load.

#except KeyboardInterrupt: # except the program gets interrupted by Ctrl+C on the keyboard.
#    BP.reset_all()        # Unconfigure the sensors, disable the motors, and restore the LED to the control of the BrickPi3 firmware.






#except KeyboardInterrupt: # except the program gets interrupted by Ctrl+C on the keyboard.



def main():
    command_string=enter_command()
    if validate_command(command_string):
        motorB_reset()
        motorC_reset()
        execute_command(command_string)
    else:
        print("command invalid")
    


main()
BP.reset_all()        # Unconfigure the sensors, disable the motors, and restore the LED to the control of the BrickPi3 firmware.



