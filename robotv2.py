# Anthony Paech's Raspbery pi / Brick Pi Lego robot project in python started 19/4/19
# by anthony paech
#
#   Goal :  build a tracked vehicle controlled by a python code
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
import csv      # for importing the decision tables as .csv files
import time     # import the time library for the sleep function
import brickpi3 # import the BrickPi3 drivers

BP = brickpi3.BrickPi3() # Create an instance of the BrickPi3 class. BP will be the BrickPi3 object.
move_size=80  # global variable forwards or backwards move.  this move size translates to 25m by 50 moves on the snooker table 
turn_size=497 # with good batteries global variable of motor degrees movement to turn the robot 90 degrees
#turn_size=510 # with low batteries, 485 with good batteries
power_percent=50
speed_limit=350   # in degrees per second
wait_time=2  # seconds to wait after a move
colour = ["none", "Black", "Blue", "Green", "Yellow", "Red", "White", "Brown"]
wiggle_angle=15    # angle the robot turns each way off of straight to look around

# size of table x= 4200, y=8400
# position is from centre.  robot has a radius of about 100

class robot:
    def __init__(self,x=100,y=100,r=["F","R","B","L"],gx=1900,gy=3900):   #, m=[["L","R","-","T"],["T","-","L","R"],["R","L","T","-"],["-","T","R","L"]]):
        self.posx=x   # encoder position in real life
        self.posy=y
        self.r_angle=0      # as measured by gyro
        self.orient=r
        self.direction=0
        self.robot_radius=100
        self.goalx=gx
        self.goaly=gy

        self.tablex=x/100  # virtual position is the encoder position divided by 100
        self.tabley=y/100
        self.tablew = 40        # size of the virtual table list
        self.tableh = 80
        self.table = []
        
        self.tablegoalx=gx/100   #  virtual table goal.  a 100th of the encoder position
        self.tablegoaly=gy/100
        self.tablerobot_radius=1
        
        self.turn_matrix=[]    #m   #  2D decision matrix for turning.. the value is the diection to turn.  F, R, B or L
        self.turn_matrix_h=5    # 4 starting orientations
        self.turn_matrix_w=5     # 4 different positions relative to the goal +x, -x, +y, -y
  

        #  direction is the first element 0-3, the second element is +x, -x, +y, -y
        #  that is the movement needed to get closer to the goal.  ie is the posx>goalx the we need -x to get closer
        #  L-turn left 90, R - turn right 90, T - turn 180
        #
        #     +x   -x    +y   -y
        #  F    L    R   -     T
        #  R    T   -    L    R
        #  B    R   L    T     -
        #  L    -   T    R     L

        # sensor speed decision matrix
        self.sensor_speed_matrix=[]    # 2D decision matrix for speeds  a negative speed is forwards.  the value is the speed positive or negative
        self.sensor_speed_matrix_h=7    #   up to 6 sensor readings : Front, Right(NA), Back, Left(NA), Gyro (angle),Gyro(rate)
        self.sensor_speed_matrix_w=5    #   4 different ranges of distance or angle in degrees

        #      0    1-10  10-49  50>=
        #  F
        #  R    
        #  B    
        #  L
        #  G
        #  Y


        


  #  def load_table(self):
  #      fobj = open("table.txt")
  #      h=0
  #      for line in fobj:
  #          length=len(line)
  #      #    print("y=",h," length=",length)
  #          for w in range(0,length-1):
  #              self.maze[h][w]=int(line[w:w+1])
  #          #    print("w=",w," h=",h,"line=",line[w:w+1])
  #          h+=1    
  #      fobj.close()



    def generate_table(self,w,h):
    
   #     Creates a nested list to represent the game board
    
        for i in range(h):
            self.table.append([0]*w)

           
          

    def print_table(self):
        print("  ")
        print("------")
        for elem in self.table:
            print(elem)
        print("------")
        print("  ")


    def read_in_turn_decision_matrix(self):
        print("read in turn decision matrix")
        
        with open('turn_decision_matrix.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                self.turn_matrix.append(row)



    def print_turn_matrix(self):
        print(" Print turn decision matrix ")
        print("------------")
        for elem in self.turn_matrix:
            print(elem)
        print("-----------")
        print("  ")
          
 


    def read_in_sensor_speed_decision_matrix(self):
        print("read in sensor speed decision matrix")
        with open('sensor_speed_decision_matrix.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                self.sensor_speed_matrix.append(row)
        
       #      0    <10  10-49  50>=
        #  F
        #  R    
        #  B    
        #  L    



    def print_sensor_speed_matrix(self):
        print(" Print sensor speed decision matrix ")
        print("------")
        for elem in self.sensor_speed_matrix:
            print(elem)
        print("------")
        print("  ")
                                                                               

    def show_position(self):
        print("position=(",self.posx,",",self.posy,").  Orientation=",self.orient[self.direction])
        
      #  robotlog.write("position=(",self.posx,",",self.posy,")")

  #  def show_goal(self):
  #      print("goal=(",self.goalx,",",self.goaly,")")
      #  robotlog.write("goal=(",self.goalx,",",self.goaly,")")


    def calc_dist_to_goal2(self,goal):
        return round(math.sqrt((goal[0]-self.posx)**2+(goal[1]-self.posy)**2))


    def calc_dist_moved(self, b_last_move,c_last_move):
        b_en=BP.get_motor_encoder(BP.PORT_B)
        b_move=b_en-b_last_move
        c_en=BP.get_motor_encoder(BP.PORT_C)
        c_move=c_en-c_last_move
        ave_move=round((b_move+c_move)/2)
        print("[calc dist moved] ave_move=",ave_move," b encoder=",b_en," b_move=",b_move," c encoder=",c_en,"c_move=",c_move)

        return ave_move

    def reset_all(self):
        BP.reset_all()
    

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





    def config_touch_sensor(self):
        # Configure for a touch sensor.
        # If an EV3 touch sensor is connected, it will be configured for EV3 touch, otherwise it's configured for NXT touch.
        # BP.set_sensor_type configures the BrickPi3 for a specific sensor.
        # BP.PORT_1 specifies that the sensor will be on sensor port 1.
        # BP.SENSOR_TYPE.TOUCH specifies that the sensor will be a touch sensor.
        print("config touch sensor")
        BP.set_sensor_type(BP.PORT_2, BP.SENSOR_TYPE.TOUCH)




    def read_touch_sensor(self):
         # read and display the sensor value
        # BP.get_sensor retrieves a sensor value.
        # BP.PORT_1 specifies that we are looking for the value of sensor port 1.

        # BP.get_sensor returns the sensor value (what we want to display).
        value=False
        try:
            value = BP.get_sensor(BP.PORT_2)
         #   print("Read touch sensor. value= ",value)
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
        self.config_colour_sensor(BP)
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
        

    def configEV3_ultrasonic_sensorS1(self):
        # Configure for an EV3 color sensor.
        # BP.set_sensor_type configures the BrickPi3 for a specific sensor.
        # BP.PORT_3 specifies that the sensor will be on sensor port 3.
        # BP.Sensor_TYPE.EV3_ULTRASONIC_CM specifies that the sensor will be an EV3 ultrasonic sensor.
        print("config ultra sonic sensor EV3 S1")
        #BP.set_sensor_type(BP.PORT_4, BP.SENSOR_TYPE.EV3_ULTRASONIC_CM) # Configure for an EV3 ultrasonic sensor.
        BP.set_sensor_type(BP.PORT_1, BP.SENSOR_TYPE.EV3_INFRARED_PROXIMITY)



    def configNXT_ultrasonic_sensorS4(self):
        # Configure for an NXT ultrasonic sensor.
        # BP.set_sensor_type configures the BrickPi3 for a specific sensor.
        # BP.PORT_1 specifies that the sensor will be on sensor port 1.
        # BP.Sensor_TYPE.NXT_ULTRASONIC) specifies that the sensor will be an NXT ultrasonic sensor.
        print("config ultra sonic sensor NXT S4")
        #BP.set_sensor_type(BP.PORT_1, BP.SENSOR_TYPE.EV3_ULTRASONIC_CM) # Configure for an EV3 ultrasonic sensor.
        BP.set_sensor_type(BP.PORT_4, BP.SENSOR_TYPE.NXT_ULTRASONIC)


    def configNXT_ultrasonic_sensorS3(self):
        # Configure for an NXT ultrasonic sensor.
        # BP.set_sensor_type configures the BrickPi3 for a specific sensor.
        # BP.PORT_4 specifies that the sensor will be on sensor port 4.
        # BP.Sensor_TYPE.NXT_ULTRASONIC) specifies that the sensor will be an NXT ultrasonic sensor.
        print("config ultra sonic sensor NXT S3")
        #BP.set_sensor_type(BP.PORT_4, BP.SENSOR_TYPE.EV3_ULTRASONIC_CM) # Configure for an EV3 ultrasonic sensor.
        BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.NXT_ULTRASONIC)



    def readEV3_ultrasonic_sensorS1(self):
        # BP.get_sensor retrieves a sensor value.
        # BP.PORT_3 specifies that we are looking for the value of sensor port 3.
        # BP.get_sensor returns the sensor value (what we want to display).
        value=0
        try:
            value = BP.get_sensor(BP.PORT_1)
    #    print("read EV3 ultra sonic sensor value S3= ",value)                         # print the distance in CM
        except brickpi3.SensorError as error:
            print(error)

        return value

    def readNXT_ultrasonic_sensorS4(self):
        # BP.get_sensor retrieves a sensor value.
        # BP.PORT_1 specifies that we are looking for the value of sensor port 1.
        # BP.get_sensor returns the sensor value (what we want to display).
        value=0
        try:
            value = BP.get_sensor(BP.PORT_4)
            #    print("read ultra sonic sensor value S1= ",value)                         # print the distance in CM
        except brickpi3.SensorError as error:
            print(error)
   

        #if value>100:
        #    value=50

        return value


    def readNXT_ultrasonic_sensorS3(self):
        # BP.get_sensor retrieves a sensor value.
        # BP.PORT_4 specifies that we are looking for the value of sensor port 4.
        # BP.get_sensor returns the sensor value (what we want to display).
        value=0
        try:
            value = BP.get_sensor(BP.PORT_3)
            #    print("read ultra sonic sensor value S4= ",value)                         # print the distance in CM
        except brickpi3.SensorError as error:
            print(error)

     #   if value>100:
     #       value=50

        return value

    def configure_gyro(self):
        try:
            BP.set_sensor_type(BP.PORT_2, BP.SENSOR_TYPE.EV3_GYRO_ABS_DPS)
            print("configuring gyro")
        except brickpi3.SensorError as error:
            print(error)
            
    def gyro_angle(self):
        # BP.get_sensor retrieves a sensor value.
        # BP.PORT_1 specifies that we are looking for the value of sensor port 1.
        # BP.get_sensor returns the sensor value (what we want to display).
        angle=0
        try:
            angle=BP.get_sensor(BP.PORT_2)   # print the gyro sensor values
      #      print("gyro angle=",angle)
        except brickpi3.SensorError as error:
            print(error)

        return angle    

    def stop(self):
      #  print("stop")                
        BP.set_motor_power(BP.PORT_B + BP.PORT_C, 0)   # stop
        


 

    def turn_angle(self, angle,creep):  # if creep is true accuracy is checked, false it is not
        self.stop()   # stop completely to get a good gyro reading
    #    print("turn ",angle," degrees")
        time.sleep(0.2)  # wait for gyro to settle down
        angle_move=angle*(turn_size/90)
      #  print("angle_move=",angle_move," turn size=",turn_size)
        start_gyro=self.gyro_angle()[0]   # current gyro angle
      #  print("initial gyro angle=",start_gyro)

        time.sleep(0.3)
        try:
            target=BP.get_motor_encoder(BP.PORT_B)
      #      print("motor B encoder value before move: ",target, "calculated pos SB: ", target-(angle_move)) 
            BP.set_motor_position(BP.PORT_B, target-(angle_move))    # set motor B's target position
   
            target=BP.get_motor_encoder(BP.PORT_C)
     #       print("motor C encoder value before move: ",target, "calculated pos SB: ", target+(angle_move)) 
            BP.set_motor_position(BP.PORT_C, target+(angle_move))    # set motor C's target position 
        except IOError as error:
            print(error)

     #   print("turn finshed?")
        time.sleep(abs(angle_move/250))
        finish_gyro=self.gyro_angle()[0]   # current gyro angle
        current_turn_angle=finish_gyro-start_gyro
 #       print("finish gyro reading=",finish_gyro," gyro angle turned=",finish_gyro-start_gyro)
 
 #       time.sleep(wait_time)
    #    self.configure_gyro()   # reset the gyro, it gets out when turning

    # we may need a minor correction to get the turn angle according to the gyro exact

        
        while creep is True and current_turn_angle != angle:
        #    print("creep loop")
            if current_turn_angle<angle:
                creep=8
            else:
                creep=-8
                
            try:
                target=BP.get_motor_encoder(BP.PORT_B)
      #      print("motor B encoder value before move: ",target, "calculated pos SB: ", target-(angle_move)) 
                BP.set_motor_position(BP.PORT_B, target-creep)    # set motor B's target position
   
            except IOError as error:
                print(error)

   
            try:
                target=BP.get_motor_encoder(BP.PORT_C)
     #       print("motor C encoder value before move: ",target, "calculated pos SB: ", target+(angle_move)) 
                BP.set_motor_position(BP.PORT_C, target+creep)    # set motor C's target position 
            except IOError as error:
                print(error)


            time.sleep(0.2)
            finish_gyro=self.gyro_angle()[0]   # current gyro angle
            current_turn_angle=finish_gyro-start_gyro

    #    print("finish gyro reading=",finish_gyro," current turn angle=",current_turn_angle)


        # to keep orientation correct, only turn either -90 or +90 degrees
        if angle==90:   # turning clockwise (right)
            if self.direction<3:
                self.direction+=1
            else:
                self.direction=0
        elif angle==-90:
            if self.direction>0:            # turning anti clockwise (left)
                self.direction-=1
            else:
                self.direction=3
        #else:
         #   print("turn angle ",angle," unrecogised. orientation not updated.")



    #    print("motor B encoder value after move = ", BP.get_motor_encoder(BP.PORT_B))
    #    print("motor C encoder value after move = ", BP.get_motor_encoder(BP.PORT_C))

    #time.sleep(wait_time)



    def look_around(self,move):
        distance=[[0,0,0],[100,100,100],[100,100,100]]   # Front left, front right, back for three views straight, turned 2 degrees right, turned 2 degrees left
        try:
            distance[0][0]=self.readEV3_ultrasonic_sensorS1()  #left
            distance[0][1]=self.readNXT_ultrasonic_sensorS4()  #right
            distance[0][2]=self.readNXT_ultrasonic_sensorS3()  # back
       #     print("STRAIGHT look around left, right, back",distance)

        except IOError as error:
            print(error)


      #  print("distance=",distance," move%5=",move%5)
        if move%3==0:   # every 3rd move look left and right in a wiggle
            start_angle=self.gyro_angle()[0] 
      #  time.sleep(0.5)
            self.turn_angle(wiggle_angle,True)  # turn right 12 degrees for a another look
            time.sleep(0.4)
            try:
                distance[1][0]=self.readEV3_ultrasonic_sensorS1()  #left
                distance[1][1]=self.readNXT_ultrasonic_sensorS4()  #right
                distance[1][2]=self.readNXT_ultrasonic_sensorS3()  # back
        #        print("RIGHT look around left, right, back",distance)

            except IOError as error:
                print(error)

           # print("distance=",distance)
            #time.sleep(0.5)  
            self.turn_angle(-wiggle_angle*2,True)  # dont check accxuracy turn left 8 degrees fromt straight for a another look
            time.sleep(0.7)
            try:
                distance[2][0]=self.readEV3_ultrasonic_sensorS1()  #left
                distance[2][1]=self.readNXT_ultrasonic_sensorS4()  #right
                distance[2][2]=self.readNXT_ultrasonic_sensorS3()  # back
       #         print("LEFT look around left, right, back",distance)

            except IOError as error:
                print(error)

       #     print("distance=",distance)
            #time.sleep(0.5)
    
          
            self.turn_angle(wiggle_angle,True)  # turn right 12 degrees to be straight again straight for a another look
            time.sleep(0.4)
            finish_angle=self.gyro_angle()[0] 
            self.turn_angle(start_angle-finish_angle,True)  # correct any bias between the motors,  get the robot on the same angle as it startedturn right 12 degrees to be straight again straight for a another look
            time.sleep(0.4)        
            

            
        mindist=[100,100,100]
        mindist[0]=min(distance[0][0],distance[1][0],distance[2][0])
        mindist[1]=min(distance[0][1],distance[1][1],distance[2][1])
        mindist[2]=min(distance[0][2],distance[1][2],distance[2][2])
        print("min dist=",mindist)

        if mindist[0] < 10 or mindist[1]<10:  # front (left and right count as 1) sensor  sees something close
            obstacle_flag=True
        else:
            obstacle_flag=False



        return(mindist,obstacle_flag)

    
    def turn_decision(self,goal):
    #    print("turn decision.  current orientation=",self.orient[self.direction])
        turn=self.turn_matrix[self.direction+1]
    #    print("turn=",turn)
        if self.direction==0 or self.direction==2:   # forward or backwards orient
            if self.posx<=goal[0]:  # x
                e=1
            else:
                e=2
        else:                                   # left or right orientation
            if self.posy<=goal[1]:   # y
                e=3
            else:   
                e=4

        td=self.turn_matrix[self.direction+1][e]
   #     print("turn decision=",td," element=",e)

        if td=="L":
            self.turn_angle(-90,True)            # turn left.  True means check accuracy
        elif td=="R":
            self.turn_angle(90,True)       # right
        elif td=="T":
            self.turn_angle(90,True)   # right
            self.turn_angle(90,True)     # right
 
        return


    def speed_decision(self,distance):
     #   print("speed decision.  distance=",distance)
     #   front_distance=(distance[0]+distance[1])/2  # average the two readings
     #   rear_distance=distance[2]    # [0][x] is from the stright look
        #self.sensor_speed_matrix


        distance_choices_row=0
        front_sensor_row=1
        rear_sensor_row=3
        gyro_angle_row=5

        fspeed=0
        rspeed=0
        gspeed=0

        distance_choices=self.sensor_speed_matrix[distance_choices_row]        
        rear_choices=self.sensor_speed_matrix[rear_sensor_row]        
        front_choices=self.sensor_speed_matrix[front_sensor_row]
        gyro_angle_choices=self.sensor_speed_matrix[gyro_angle_row]
        
   #     print(" front choices=",front_choices," rear choices=",rear_choices," gyro choices=",gyro_angle_choices," distance choices=",distance_choices)


###################################
        # front sensors  (average the two readings)

        col=0
        for d in distance_choices:
            if col>0:
                start_range=int(d[0:3])
                finish_range=int(d[4:7])
      #          print("front distance choices col=",col," d=",d," start=",start_range," finish=",finish_range)
                if (distance[0]>=start_range and distance[0]<=finish_range) or (distance[1]>=start_range and distance[1]<=finish_range):
                    break

            col+=1

    #    print("column no chosen of front distance choices =",col)
        if col<=self.sensor_speed_matrix_w:
            fspeed=int(self.sensor_speed_matrix[front_sensor_row][col])  # using only front sensor
      #  speed=self.sensor_speed_matrix[rear_sensor_row][col]       # if rear sensor
        else:
            print("front Distance out of range")
            fspeed=0
            # how do I determine if the speeds recommended are different from the different sensors at the same time?



############################
            #   rear sensor

        col=0
        for d in distance_choices:
            if col>0:
                start_range=int(d[0:3])
                finish_range=int(d[4:7])
         #       print("rear distance choices col=",col," d=",d," start=",start_range," finish=",finish_range)
                if distance[2]>=start_range and distance[2]<=finish_range:
                    break

            col+=1

     #   print("column no chosen of rear distance choices =",col)
        if col<=self.sensor_speed_matrix_w:
            rspeed=int(self.sensor_speed_matrix[rear_sensor_row][col])  # using only front sensor
      #  speed=self.sensor_speed_matrix[rear_sensor_row][col]       # if rear sensor
        else:
            print("rear Distance out of range")
            rspeed=0
            # how do I determine if the speeds recommended are different from the different sensors at the same time?


######################################
            # Gyro sensor not used for speed changes

       # col=0
       # for d in distance_choices:
       #     if col>0:
       #         start_range=int(d[0:3])
       #         finish_range=int(d[4:7])
       #         print("gyro angle choices col=",col," d=",d," start=",start_range," finish=",finish_range)
       #         if gyro_angle>=start_range and gyro_angle<=finish_range:
       #             break

        #    col+=1

     #   print("column no chosen of gyro angle choices =",col)
        #if col<=self.sensor_speed_matrix_w:
        #    gspeed=int(self.sensor_speed_matrix[gyro_angle_row][col])  # using only front sensor
      #  speed=self.sensor_speed_matrix[rear_sensor_row][col]       # if rear sensor
       # else:
       #     print("gyro angle out of range")
       #     gspeed=0
            # how do I determine if the speeds recommended are different from the different sensors at the same time?

##################################3
            

       # print("speed decisions= f,r",fspeed,rspeed," obstacle flag=",obstacle_flag)
       

        speed=min(fspeed,rspeed)   # return the lowest speed recommended by the three sensors
   #     print("speed decision=",speed)
       # input("?")

        return speed






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

            if (new_h >= self.tableh):
                continue
            elif (new_h < 0):
                continue
            elif (new_w >= self.tablew):
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
          #  if to_visit==goal: # goal reached
        if to_visit[0]>=goal[0]-self.tablerobot_radius  and to_visit[0]<=goal[0]+self.tablerobot_radius and to_visit[1]>=goal[1]-self.tablerobot_radius and to_visit[1]<=goal[1]+self.tablerobot_radius:
           # self.print_table()
            print(path)
            print("goal reached",goal,"n=",n)
            sys.exit(1)


        if n == self.tablew * self.tableh: #if every grid is filled
           # self.print_maze()
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
         #   print("Going back to: ", path[-1])
        except IndexError:
            print("No path found")
            sys.exit(1)
        

        return False



    def goal_seek(self,goal):

        self.motorC_reset()
        self.motorB_reset()
        time.sleep(3)
        self.configure_gyro()
        time.sleep(5)
        
        d=0
        dmax=0
        speed=0
        b_move=0
        c_move=0
        b_en=0
        c_en=0
        b_last_move=0
        c_last_move=0
        move_size=100
        angle=[0,0]
        totalmoves=0
        move_no=0
    
  
        back_up_flag=False
        back_up_moveno=0
        obstacle_flag=False
        reverse_count=0
        reverse_flag=False
        goal_reached=False
    
        old_dist=[]
        goal_distance=[]
        position_log=[]
        table_log=[]


       # a function to move the robot to a x,y position (gx,gy)
        # this will be called recurively later
        # returns True is goal reached
        print("goal seek to:",goal)
        loop=True

        try:
            while loop:


#####################################
                # firstly, update the position based on the movement of the motor encoders
                b_en=BP.get_motor_encoder(BP.PORT_B)
                b_move=b_en-b_last_move
                c_en=BP.get_motor_encoder(BP.PORT_C)
                c_move=c_en-c_last_move
            #    print("B motor encoder=",b_en," C motor=",c_en)

                # motor encoder of 225 is equiventant to 17cm
                # 13.2 encoder per cm
                #
                ave_move=round((b_move+c_move)/2)
             #   print("ave_move=",ave_move," b encoder=",b_en," b_move=",b_move," c encoder=",c_en,"c_move=",c_move)
                if self.orient[self.direction]=="F":
                    self.posy-=ave_move
                    self.tabley=round(self.posy/100)
                elif self.orient[self.direction]=="R":
                    self.posx+=ave_move
                    self.tablex=round(self.posx/100)
                elif self.orient[self.direction]=="B":
                    self.posy+=ave_move
                    self.tabley=round(self.posy/100)
                elif self.orient[self.direction]=="L":
                    self.posx-=ave_move
                    self.tablex=round(self.posx/100)
                else:
                    print("invalid orientation")

                b_last_move=b_en
                c_last_move=c_en

##########################################
                #  update angle from start using gyro


                self.r_angle=self.gyro_angle()[0] 

###########################################
#               this is the main robot movement loop

                
                distance_to_goal=self.calc_dist_to_goal2(goal)
            
                if distance_to_goal<self.robot_radius*2:    # if robot is close enough to goal
                    print("goal reached!")
                    loop=False
                    goal_reached=True


##########################################################

                #  basic goal finding here.  turn when you start getting further from the goal.  doesn't work in a maze
                #if move_no>3 and distance_to_goal>goal_distance[3]:
                #    print("getting further away. distance to goal",distance_to_goal," >gd[3]=",goal_distance[3])
                #    self.turn_decision(goal)



######################################
                #  look around
                

                
                distance, obstacle_flag=self.look_around(move_no)
                
                if obstacle_flag:   # if obstacle seen, update the virtual table map
                    if self.orient[self.direction]=="F":
                        self.table[self.tabley][self.tablex+1]=1
                    elif self.orient[self.direction]=="R":
                        self.table[self.tabley-1][self.tablex]=1
                    elif self.orient[self.direction]=="B":
                        self.table[self.tabley][self.tablex-1]=1
                    elif self.orient[self.direction]=="L":
                        self.table[self.tabley+1][self.tablex]=1
                    else:
                        print("orientation not correct")
                    print("obstacle flag means a 1 on the virtual table, current position=",self.tablex,self.tabley)    
              #      self.print_table()
            


#######################################################


                # recursive function starts the moving. ( move,path,starting pos, goal pos)
                #rp.move(1, [], (start_h,start_w), (gh,gw))#####################

                

##############################################################
                
                speed=self.speed_decision(distance)


#############################################################
                #  now update the speed and position and then loop back


                
                time.sleep(0.5)
                
                
                BP.set_motor_power(BP.PORT_B + BP.PORT_C, -speed)   # going backwards is actually going forward

                table_log.append([self.tablex,self.tabley])
                position_log.append([self.posx,self.posy,self.orient[self.direction]])    
                goal_distance.append(distance_to_goal)
                old_dist.append(distance)

###############################################
               #  other housekeeping 
            
                print("move_no=",move_no,"position=",position_log[move_no],"table_log",table_log[move_no]," speed=",speed," angle=",self.r_angle,"goal=",goal)
               

                if move_no>3:
                    del old_dist[0]
                    del goal_distance[0]
        
           
                time.sleep(0.3)    #  time between looks.  still travelling during this time
                move_no+=1

# end of main robot movement loop
####################################################3

    
        except KeyboardInterrupt:
            BP.reset_all()            
         #   close_log(robotlog)


        return goal_reached

