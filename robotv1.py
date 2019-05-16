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
turn_size=510 # with good batteries global variable of motor degrees movement to turn the robot 90 degrees
#turn_size=510 # with low batteries, 485 with good batteries
power_percent=50
speed_limit=350   # in degrees per second
wait_time=2  # seconds to wait after a move
colour = ["none", "Black", "Blue", "Green", "Yellow", "Red", "White", "Brown"]


# size of table x= 4200, y=8400
# position is from centre.  robot has a radius of about 100

class robot:
    def __init__(self,x=200,y=300,r=["F","R","B","L"]):   #, m=[["L","R","-","T"],["T","-","L","R"],["R","L","T","-"],["-","T","R","L"]]):
        self.posx=x
        self.posy=y
        self.orient=r
        self.direction=0
        self.robot_radius=100
        self.goalx=2000
        self.goaly=4000
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

    def show_goal(self):
        print("goal=(",self.goalx,",",self.goaly,")")
      #  robotlog.write("goal=(",self.goalx,",",self.goaly,")")


    def turn_decision(self,gx,gy):
        print("turn decision.  current orientation=",self.orient[self.direction])
        turn=self.turn_matrix[self.direction+1]
        print("turn=",turn)
        if self.direction==0 or self.direction==2:   # forward or backwards orient
            if self.posx<=gx:
                e=1
            else:
                e=2
        else:                                   # left or right orientation
            if self.posy<=gy:
                e=3
            else:   
                e=4

        td=self.turn_matrix[self.direction+1][e]
        print("turn decision=",td," element=",e)

        if td=="L":
            self.turn_left90()
        elif td=="R":
            self.turn_right90()
        elif td=="T":
            self.turn_right90()
            self.turn_right90()
 
        return


    def speed_decision(self,distance,gyro_angle):
        print("speed decision.  distance=",distance)
        front_distance=(distance[0]+distance[1])/2  # average the two readings
        rear_distance=distance[2]
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
                if front_distance>=start_range and front_distance<=finish_range:
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
                if rear_distance>=start_range and rear_distance<=finish_range:
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
            # Gyro sensor

        col=0
        for d in distance_choices:
            if col>0:
                start_range=int(d[0:3])
                finish_range=int(d[4:7])
       #         print("gyro angle choices col=",col," d=",d," start=",start_range," finish=",finish_range)
                if gyro_angle>=start_range and gyro_angle<=finish_range:
                    break

            col+=1

     #   print("column no chosen of gyro angle choices =",col)
        if col<=self.sensor_speed_matrix_w:
            gspeed=int(self.sensor_speed_matrix[gyro_angle_row][col])  # using only front sensor
      #  speed=self.sensor_speed_matrix[rear_sensor_row][col]       # if rear sensor
        else:
            print("gyro angle out of range")
            gspeed=0
            # how do I determine if the speeds recommended are different from the different sensors at the same time?

##################################3
            




        if distance[0] < 10 or distance[1]<10 or distance[2]<10:  # front and rear (left and right count as 1) sensor
            obstacle_flag=True
        else:
            obstacle_flag=False

        
        print("speed decisions= f,r,g",fspeed,rspeed,gspeed," obstacle flag=",obstacle_flag)
       

        speed=min(fspeed,rspeed,gspeed)   # return the lowest speed recommended by the three sensors
        print("speed decision=",speed)
       # input("?")

        return(speed, obstacle_flag)





    
    def calc_dist_to_goal(self):
        return round(math.sqrt((self.goalx-self.posx)**2+(self.goaly-self.posy)**2))

    def calc_dist_to_goal2(self,gx,gy):
        return round(math.sqrt((gx-self.posx)**2+(gy-self.posy)**2))


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


    def look_around(self):
        distance=[0,0,0]   # Front left, front right, back
        try:
            distance[0]=self.readEV3_ultrasonic_sensorS1()  #left
            distance[1]=self.readNXT_ultrasonic_sensorS4()  #right
            distance[2]=self.readNXT_ultrasonic_sensorS3()  # back
            print("look around left, right, back",distance)

        except IOError as error:
            print(error)

        return(distance)



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
   

        if value>100:
            value=50

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

        if value>100:
            value=50

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
            print("gyro angle=",angle)
        except brickpi3.SensorError as error:
            print(error)

        return angle    

    def turn_left90(self):
        print("turn_left90")
        try:
            target=BP.get_motor_encoder(BP.PORT_B)
       #     print("motor B encoder value before move: ",target, "calculated pos SB: ", target+(turn_size)) 
            BP.set_motor_position(BP.PORT_B, target+(turn_size))    # set motor B's target position
   
        except IOError as error:
            print(error)

 #   time.sleep(3)

    
     
        try:
            target=BP.get_motor_encoder(BP.PORT_C)
      #      print("motor C encoder value before move: ",target, "calculated pos SB: ", target-(turn_size)) 
            BP.set_motor_position(BP.PORT_C, target-(turn_size))    # set motor C's target position 
        except IOError as error:
            print(error)


        if self.direction>0:
            self.direction-=1
        else:
            self.direction=3
                        
        time.sleep(wait_time)
        self.configure_gyro()   # reset the gyro, it gets out when turning

    #    print("motor B encoder value after move = ", BP.get_motor_encoder(BP.PORT_B))
     #   print("motor C encoder value after move = ", BP.get_motor_encoder(BP.PORT_C))


    #time.sleep(wait_time)


    def turn_right90(self):
        print("turn_right 90")
        try:
            target=BP.get_motor_encoder(BP.PORT_B)
   #         print("motor B encoder value before move: ",target, "calculated pos SB: ", target-(turn_size)) 
            BP.set_motor_position(BP.PORT_B, target-(turn_size))    # set motor B's target position
   
        except IOError as error:
            print(error)

       # time.sleep(1)

    
     
        try:
            target=BP.get_motor_encoder(BP.PORT_C)
  #          print("motor C encoder value before move: ",target, "calculated pos SB: ", target+(turn_size)) 
            BP.set_motor_position(BP.PORT_C, target+(turn_size))    # set motor C's target position 
        except IOError as error:
            print(error)

        if self.direction<3:
            self.direction+=1
        else:
            self.direction=0

        time.sleep(wait_time)
        self.configure_gyro()   # reset the gyro, it gets out when turning

#        print("motor B encoder value after move = ", BP.get_motor_encoder(BP.PORT_B))
 #       print("motor C encoder value after move = ", BP.get_motor_encoder(BP.PORT_C))

        #time.sleep(wait_time)


    def turn_angle(self, angle):
        print("turn ",angle," degrees")
        angle_move=angle*(turn_size/90)
        try:
            target=BP.get_motor_encoder(BP.PORT_B)
      #      print("motor B encoder value before move: ",target, "calculated pos SB: ", target-(angle_move)) 
            BP.set_motor_position(BP.PORT_B, target-(angle_move))    # set motor B's target position
   
        except IOError as error:
            print(error)

   # time.sleep(1)

    
     
        try:
            target=BP.get_motor_encoder(BP.PORT_C)
      #      print("motor C encoder value before move: ",target, "calculated pos SB: ", target+(angle_move)) 
            BP.set_motor_position(BP.PORT_C, target+(angle_move))    # set motor C's target position 
        except IOError as error:
            print(error)

        time.sleep(wait_time)
        self.configure_gyro()   # reset the gyro, it gets out when turning

    #    print("motor B encoder value after move = ", BP.get_motor_encoder(BP.PORT_B))
    #    print("motor C encoder value after move = ", BP.get_motor_encoder(BP.PORT_C))

    #time.sleep(wait_time)


    def goal_seek(self,gx,gy):

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
        move=0
    
  
        back_up_flag=False
        back_up_moveno=0
        obstacle_flag=False
        goal_reached=False
    
        old_dist=[]
        goal_distance=[]
        position_log=[]


       # a function to move the robot to a x,y position (gx,gy)
        # this will be called recurively later
        # returns True is goal reached
        print("goal seek to: gx=",gx," , gy=",gy)
        loop=True

        try:
            while loop:


                # firstly, update the position based on the movement of the motor encoders
                b_en=BP.get_motor_encoder(BP.PORT_B)
                b_move=b_en-b_last_move
                c_en=BP.get_motor_encoder(BP.PORT_C)
                c_move=c_en-c_last_move
                ave_move=round((b_move+c_move)/2)
             #   print("ave_move=",ave_move," b encoder=",b_en," b_move=",b_move," c encoder=",c_en,"c_move=",c_move)
                if self.orient[self.direction]=="F":
                    self.posy-=ave_move
                elif self.orient[self.direction]=="R":
                    self.posx+=ave_move
                elif self.orient[self.direction]=="B":
                    self.posy+=ave_move
                elif self.orient[self.direction]=="L":
                    self.posx-=ave_move
                else:
                    print("invalid orientation")

                b_last_move=b_en
                c_last_move=c_en


       #     ave_move=rp.calc_dist_moved(b_last_move,c_last_move)
         #   print("ave_move=",ave_move)

         #   angle=self.gyro_angle()
            #    self.show_position()

            #    loop= not self.read_touch_sensor()    

                distance_to_goal=self.calc_dist_to_goal2(gx,gy)
                print("goal=",gx,gy," dist to goal=",distance_to_goal)
            
                if distance_to_goal<self.robot_radius*2:
                    print("goal reached!")
                    loop=False
                    goal_reached=True


                if move>3 and distance_to_goal>goal_distance[3]:
                    print("getting further away. gd[3]=",goal_distance[3])
                    self.turn_decision(gx,gy)

                
                distance=self.look_around()
                turn_dir="-"

               # speed=40   # default go forward

                angle=self.gyro_angle()[0]
                speed, obstacle_flag=self.speed_decision(distance,abs(angle))
            
                #if distance[0] < 10 or distance[1]<10:  # left and right sensor
                #    obstacle_flag=True
                #  time to turn.  Turn towards the goal
                if obstacle_flag:
                    self.turn_decision(gx,gy)
                    obstacle_flag=False
               


               
             #   if abs(angle)>15:      #Robot climbing walls and fall off table
             #       print("angle >15 degrees", angle,"degrees: about to jump wall")
             #       back_up_flag=True  # reverse off
             #       back_up_moveno=move
          #      elif move>3:
              #     print("distance[0]=",distance[0],distance[1]," old dist=",old_dist[3][0],old_dist[3][1])
           #         if abs(distance[0]-old_dist[3][0])>30 or abs(distance[1]-old_dist[3][1])>30:    
           #             print("sudden change in view.  jumped the wall?")
           #             back_up_flag=True   # reverse off
           #             back_up_moveno=move

                if abs(angle)>20:
                    print("TILT!")
                    speed=0
                    loop=False
                    goal_reached=False

                    
                
            
                
                
                BP.set_motor_power(BP.PORT_B + BP.PORT_C, -speed)   # going backwards is actually going forward

                position_log.append([self.posx,self.posy,self.orient[self.direction]])    
                goal_distance.append(distance_to_goal)
                old_dist.append(distance)
            
      
            #    print("move=",move,"position=",position_log[move]," speed=",speed," goal_distance=",goal_distance," old_dist=",old_dist)
                print("move=",move,"position=",position_log[move]," speed=",speed)
               

             #   writestr=("move"+str(move))
            #    writestr=("move="+str(move)+" dist to obst="+str(d)+" speed="+str(speed)+" pos=("+str(rp.posx)+","+str(rp.posy)+")  orient="+rp.orient[rp.direction]+" goal=("+str(rp.goalx)+","+str(rp.goaly)+") goal dist="+str(distance_to_goal)+"\r")    
            #    print(writestr)
            #    print("distance to goal:",goal_distance)
              #  robotlog.write(writestr)


                if move>3:
                    del old_dist[0]
                    del goal_distance[0]
        
           
                time.sleep(0.05)
                move+=1
    
        except KeyboardInterrupt:
            BP.reset_all()            
         #   close_log(robotlog)


        return goal_reached

