# Anthony Paech's Raspbery pi / Brick Pi Lego robot project in python started 19/4/19
# by anthony paech
#
#   Goal :  build a tracked vehicle controlled by a python code
#
# stage 1 a programmable robot that can follow instructions
# stage 2 is stage 1 that can stop if it hits an object
#  stage 3 is stage 2 that can see obstacals
# stage 3 is where it can navigate across a pool table dodging balls.
# stage 3 is where it is given a high level goal and it finds its way there
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
move_size=80  # global variable forwards or backwards move.  this move size translates to 25m by 50 moves on the snooker table 
turn_size=485 # global variable of motor degrees movement to turn the robot 90 degrees
power_percent=50
speed_limit=250   # in degrees per second
wait_time=0.02  # seconds to wait after a move
colour = ["none", "Black", "Blue", "Green", "Yellow", "Red", "White", "Brown"]
ultra_safety_dist=2   # minimum distance in centimeters that the commands should stop working
backup_count_strategy_change=1   # if the robot has to backup off an obstcal either forward or reverse, this is the number of times it does that before turning 45 degrees on a new strategy
turn_avoid=33    # if the robot had to turn around an object, turn alternatively left or right 33 degrees


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


def config_touch_sensor():
# Configure for a touch sensor.
# If an EV3 touch sensor is connected, it will be configured for EV3 touch, otherwise it's configured for NXT touch.
# BP.set_sensor_type configures the BrickPi3 for a specific sensor.
# BP.PORT_1 specifies that the sensor will be on sensor port 1.
# BP.SENSOR_TYPE.TOUCH specifies that the sensor will be a touch sensor.
    print("config touch sensor")
    BP.set_sensor_type(BP.PORT_1, BP.SENSOR_TYPE.TOUCH)




def read_touch_sensor():
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


def config_colour_sensor():
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




def read_colour_sensor():    
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

def config_ultrasonic_sensor():
# Configure for an EV3 color sensor.
# BP.set_sensor_type configures the BrickPi3 for a specific sensor.
# BP.PORT_4 specifies that the sensor will be on sensor port 4.
# BP.Sensor_TYPE.EV3_ULTRASONIC_CM specifies that the sensor will be an EV3 ultrasonic sensor.
    print("config ultra sonic sensor")
    #BP.set_sensor_type(BP.PORT_4, BP.SENSOR_TYPE.EV3_ULTRASONIC_CM) # Configure for an EV3 ultrasonic sensor.
    BP.set_sensor_type(BP.PORT_4, BP.SENSOR_TYPE.EV3_INFRARED_PROXIMITY)


def read_ultrasonic_sensor():
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


    
def execute_command(com_str):
    c_count=1
    turn_back_count=0
    turn_forward_count=0
    turn_avoidance_angle=turn_avoid   # global
    turn_right_count=0
    turn_left_count=0     # obstale aviodance swerve counts
    turn_change=0    # if the turn count is odd, turn right, even turn left
    print("execute commands: ", com_str)
    for l in com_str:
               
        time.sleep(0.05)  # delay for 0.05 seconds (50ms) to reduce the Raspberry Pi CPU load.

        print(c_count," - execute command: ",l)

        miss_command=False
        touch=read_touch_sensor()
        if touch:
            print("touch sensor triggered")
            print("forward off touch")
            move_forward()
            turn_forward_count=turn_forward_count+1
            miss_command=True
        distance=read_ultrasonic_sensor()
        print("ultra sonic sensor dist= ", distance)
        if distance<ultra_safety_dist:  # distance to stop
            print("ultra sonic sensor triggered. distance= ", distance)
            print("backward off ultra sonic")
            move_backward()
            turn_back_count=turn_back_count+1
            miss_command=True

        if turn_right_count>0:
            print("get back on track. turn right count= ",turn_right_count," turning ",-(90-turn_avoidance_angle))
            turn_angle(turn_avoidance_angle)
            turn_right_count=0
        elif turn_left_count>0:
            print("get back on track turn left count= .", turn_left_count," turning ",90-turn_avoidance_angle)
            turn_angle(-turn_avoidance_angle)
            turn_left_count=0

            
        if (turn_forward_count+turn_back_count)>backup_count_strategy_change:
            if turn_change%2:    # odd or even?  change strategy
                turn_avoidance_angle=turn_avoid   #global  probably set to 33 degrees
                turn_right_count=turn_right_count+1
            else:
                turn_avoidance_angle=-turn_avoid
                turn_left_count=turn_left_count+1
            print("backup limit exceeded.  Turning ",turn_avoidance_angle," degrees...")
            turn_angle(turn_avoidance_angle)
            turn_change=turn_change+1
            turn_back_count=0
            turn_forward_count=0

       
            
        if not miss_command:
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
        else:
            print("command ",l," ignored, touch sensor or ultra sonic sensor triggered")
                  
    print(c_count-1," :Command(s) completed")

    
    




def main():
    #config_colour_sensor()
   # config_touch_sensor()
   
    distance=0
    move_size=100
    happy_dist=10
    config_ultrasonic_sensor()
    motorC_reset()
    motorB_reset()
    try:
        while True:
            distance=read_ultrasonic_sensor()
            time.sleep(0.1)
            if distance <50:
                speed=(distance-happy_dist)**3
            else:
                speed=80
                    
            print("distance=",distance," speed=",speed)
            BP.set_motor_power(BP.PORT_A + BP.PORT_B + BP.PORT_C + BP.PORT_D, speed)

            time.sleep(0.1)
    
    
    except KeyboardInterrupt:
        BP.reset_all()            

main()
BP.reset_all()        # Unconfigure the sensors, disable the motors, and restore the LED to the control of the BrickPi3 firmware.



