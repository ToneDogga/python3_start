# Anthony Paech's Raspbery pi / Brick Pi Lego robot project in python started 25/5/19
# by anthony paech
#
#   Goal :  build a raspberry pi python coded teachable arm
#   loads a config file
#  outputs to a logfile.csv
#   has a teach mode
#  can memorise a huge number of moves, saves them to file
# uses a flirc remote
#   interchangable head
# gyro control to keep head vertical at all times
# small motor one way turns the head, the other tightens the grippers
# direct drive, no gears . use actuators
#
#!/usr/bin/env python
#
# https://www.dexterindustries.com/BrickPi/
# https://github.com/DexterInd/BrickPi3
#

from __future__ import print_function # use python 3 syntax but make it compatible with python 2
from __future__ import division       #                           ''


import sys
import time
import random
import brickpi3   # brickpi dexter library

def configure_gyro(rp):
    try:
        rp.set_sensor_type(rp.PORT_2, rp.SENSOR_TYPE.EV3_GYRO_ABS_DPS)
        print("configuring gyro")
    except brickpi3.SensorError as error:
        print(error)
            
def gyro_angle(rp):
        # BP.get_sensor retrieves a sensor value.
        # BP.PORT_1 specifies that we are looking for the value of sensor port 1.
        # BP.get_sensor returns the sensor value (what we want to display).
    angle=0
    try:
        angle=rp.get_sensor(rp.PORT_2)   # print the gyro sensor values
      #      print("gyro angle=",angle)
    except brickpi3.SensorError as error:
        print(error)

    return angle    


def calibrate_gyro(rp):
    starting_angle=0
    print("Calibrating Gyro...")
   # while not rp.read_touch_sensor():
   # time.sleep(4)

   # while c<7:
   # try:
    starting_angle=gyro_angle(rp)[0]
    #except IOError as error:
    #        print(error)
    #else: starting_angle=0
    
    #finally:
        
  #  time.sleep(0.5)
    print("starting angle=",starting_angle)
    time.sleep(4)
    
    return starting_angle


def motor_init(rp):
    rp.set_motor_limits(rp.PORT_A, 50, 100)          # optionally set a power limit (in percent) and a speed limit (in Degrees Per Second)
    rp.set_motor_limits(rp.PORT_B, 50, 100)          # optionally set a power limit (in percent) and a speed limit (in Degrees Per Second)
    rp.set_motor_limits(rp.PORT_C, 50, 100)          # optionally set a power limit (in percent) and a speed limit (in Degrees Per Second)
    rp.set_motor_limits(rp.PORT_D, 50, 100)          # optionally set a power limit (in percent) and a speed limit (in Degrees Per Second)




def reset_encoders(rp):
    print("reset encoders")
    try:
        rp.offset_motor_encoder(rp.PORT_A, rp.get_motor_encoder(rp.PORT_A)) # reset encoder
     #   print("Motor D reset encoder done")
    except IOError as error:
        print(error)

    try:
        rp.offset_motor_encoder(rp.PORT_D, rp.get_motor_encoder(rp.PORT_D)) # reset encoder
     #   print("Motor D reset encoder done")
    except IOError as error:
        print(error)

    try:
        rp.offset_motor_encoder(rp.PORT_B, rp.get_motor_encoder(rp.PORT_B)) # reset encoder
     #   print("Motor B reset encoder done")
    except IOError as error:
        print(error)

    try:
        rp.offset_motor_encoder(rp.PORT_C, rp.get_motor_encoder(rp.PORT_C)) # reset encoder
    #    print("Motor C reset encoder done")
    except IOError as error:
        print(error)

def reset_to_first_position(rp):
    print("reset to first position")



def float_motors(rp):
    
 #  rp.offset_motor_encoder(BP.PORT_A, BP.get_motor_encoder(BP.PORT_A)) # reset encoder
#            print("Motor A reset encoder done")
#        except IOError as error:
#            print(error)3

    rp.set_motor_power(rp.PORT_A, rp.MOTOR_FLOAT)
    rp.set_motor_power(rp.PORT_B, rp.MOTOR_FLOAT)
    rp.set_motor_power(rp.PORT_C, rp.MOTOR_FLOAT)
    rp.set_motor_power(rp.PORT_D, rp.MOTOR_FLOAT)
   


def config_touch_sensor(rp):
    # Configure for a touch sensor.
    # If an EV3 touch sensor is connected, it will be configured for EV3 touch, otherwise it's configured for NXT touch.
    # BP.set_sensor_type configures the BrickPi3 for a specific sensor.
    # BP.PORT_1 specifies that the sensor will be on sensor port 1.
    # BP.SENSOR_TYPE.TOUCH specifies that the sensor will be a touch sensor.
    print("config touch sensor")
    rp.set_sensor_type(rp.PORT_3, rp.SENSOR_TYPE.TOUCH)




def read_touch_sensor(rp):
         # read and display the sensor value
        # BP.get_sensor retrieves a sensor value.
        # BP.PORT_1 specifies that we are looking for the value of sensor port 1.

        # BP.get_sensor returns the sensor value (what we want to display).
    value=False
    try:
        value = rp.get_sensor(rp.PORT_3)
         #   print("Read touch sensor. value= ",value)
    except brickpi3.SensorError as error:
        print(error)   
    return value



class arm_class:
    def __init__(self):
        self.command_angle=0      # in degrees
        self.current_angle=0
        self.error_history=[]   # keep the last 10 error measurements to calculate the sum over time

        self.call_count=0   # counting the number of times it has been called to keep the last 10 errors
        
        self.command_speed=0      # motor speed in degrees per second
        self.current_speed=0

        self.totalerror=0
        self.lasterror=0
       
        
        

    def arm_teach(self,rp):

      #  start_position=[0,0,0,0]
        finish_position=[0,0,0,0]
        
       # start_position[0]=rp.get_motor_encoder(rp.PORT_A)  # shoulder
       # start_position[1]=int(rp.get_motor_encoder(rp.PORT_B))   # elbow
       # start_position[2]=int(rp.get_motor_encoder(rp.PORT_C))    # wrist
       # start_position[3]=int(rp.get_motor_encoder(rp.PORT_D))    # hand
        
       # input("ready to teach? Hit enter")
        
        main_loop=True
        while main_loop:
            float_motors(rp)
            while not read_touch_sensor(rp):
    
                pos_a=rp.get_motor_encoder(rp.PORT_A)
                pos_b=rp.get_motor_encoder(rp.PORT_B)
                pos_c=rp.get_motor_encoder(rp.PORT_C)
                pos_d=rp.get_motor_encoder(rp.PORT_D)

                print("shoulder=",pos_a," Elbow=",pos_b," Wrist=",pos_c)
                time.sleep(0.2)
             
            rp.set_motor_position(rp.PORT_A,pos_a)  # shoulder
            rp.set_motor_position(rp.PORT_B,pos_b)   # elbow
            rp.set_motor_position(rp.PORT_C,pos_c)    # wrist
            rp.set_motor_position(rp.PORT_D,pos_d)    # hand

            answer=input("position correct? (y/n)")
            if answer=="y":
                main_loop=False

        finish_position[0]=pos_a # shoulder
        finish_position[1]=pos_b   # elbow
        finish_position[2]=pos_c   # wrist
        finish_position[3]=pos_d    # hand
 
        #print("finish pos=",finish_position)
        #input("?")
            
        return finish_position
        
        
    def move_joint_abs(self,rp,joint,pos):
        print("move joint absolute",joint," pos=",pos)  # move joint absolute values
        
        if joint=="SHOULDER":
            # move PORT_A
            # turn table 60 teeth, driven by 20 teeth cog.  therefore time the angle move by 7.5

            
         #   try:
          #      current_pos = int(rp.get_motor_encoder(rp.PORT_A)) # read motor's position
           # except IOError as error:
            #    print(error)
        
           # print("current pos=",current_pos," target=",current_pos+angle*3)

            try:
                rp.set_motor_position(rp.PORT_A,pos) # read motor's position
            except IOError as error:
                print(error)

                
        elif joint=="ELBOW":
            # move PORT_B
         #   try:
         #       current_pos = rp.get_motor_encoder(rp.PORT_B) # read motor's position
         #   except IOError as error:
         #       print(error)
        
         #   print("current pos=",current_pos," target=",current-angle)

            try:
                rp.set_motor_position(rp.PORT_B,pos) # read motor's position
            except IOError as error:
                print(error)
           
            
        elif joint=="WRIST":    
            #move PORT_C
         #   try:
         #       current_pos = rp.get_motor_encoder(rp.PORT_C) # read motor's position
         #   except IOError as error:
          #      print(error)
        
           # print("current pos=",current_pos," target=",current_pos-angle)

            try:
                rp.set_motor_position(rp.PORT_C,pos) # read motor's position
            except IOError as error:
                print(error)

            
        elif joint=="HAND":
            # move PORT_D
           # try:
           #     current_pos = rp.get_motor_encoder(rp.PORT_D) # read motor's position
           # except IOError as error:
           #     print(error)
        
            #print("current pos=",current_pos," target=",current_pos-angle)

            try:
                rp.set_motor_position(rp.PORT_D,pos) # read motor's position
            except IOError as error:
                print(error)


            
        else:
            print("Invalid move_joint() call")
            return False


       # print("move completed")
        time.sleep(2)
        return True


    def move_joint_rel(self,rp,joint,angle):
        print("move joint relative",joint," angle=",angle)  # move joint absolute values
        motorname=""
        current_pos=0
        time.sleep(1)
        if joint=="SHOULDER":
            # move PORT_A
            # turn table 60 teeth, driven by 20 teeth cog.  therefore time the angle move by 7.5

            
            try:
                current_pos = int(rp.get_motor_encoder(rp.PORT_A)) # read motor's position
            except IOError as error:
                print(error)
        
            print("current pos=",current_pos," target=",current_pos+angle*3)

            try:
                rp.set_motor_position(rp.PORT_A,current_pos+angle*3) # read motor's position
            except IOError as error:
                print(error)

                
        elif joint=="ELBOW":
            # move PORT_B
            try:
                current_pos = rp.get_motor_encoder(rp.PORT_B) # read motor's position
            except IOError as error:
                print(error)
        
            print("current pos=",current_pos," target=",current_pos-angle)

            try:
                rp.set_motor_position(rp.PORT_B,current_pos-angle) # read motor's position
            except IOError as error:
                print(error)
           
            
        elif joint=="WRIST":    
            #move PORT_C
            try:
                current_pos = rp.get_motor_encoder(rp.PORT_C) # read motor's position
            except IOError as error:
                print(error)
        
            print("current pos=",current_pos," target=",current_pos-angle)

            try:
                rp.set_motor_position(rp.PORT_C,current_pos-angle) # read motor's position
            except IOError as error:
                print(error)

            
        elif joint=="HAND":
            # move PORT_D
            try:
                current_pos = rp.get_motor_encoder(rp.PORT_D) # read motor's position
            except IOError as error:
                print(error)
        
            print("current pos=",current_pos," target=",current_pos-angle)

            try:
                rp.set_motor_position(rp.PORT_D,current_pos-angle) # read motor's position
            except IOError as error:
                print(error)


            
        else:
            print("Invalid move_joint() call")
            return False


        print("move completed")
        time.sleep(2)
        return True







def main():
    
   #################################3
    #    initiate
    BP=brickpi3.BrickPi3()
    arm=arm_class()



##################################
    # config sensors
    
    print("Keep robot still for gyro config!")
    time.sleep(0.1)

    motor_init(BP)          # motor init set power and speed limits
    
    config_touch_sensor(BP)  # sensor port 3
    
    #reset_position(BP)    # port MB and port MC
    #configure_gyro(BP)    #Sensor port 2
    
   # rp.configEV3_ultrasonic_sensorS1()
   # rp.configNXT_ultrasonic_sensorS4()
    #time.sleep(2)
    
    BP.set_sensor_type(BP.PORT_1, BP.SENSOR_TYPE.EV3_INFRARED_REMOTE)

    time.sleep(2)   # 4 wait while sensors get ready

#####################################



######################################
    
    armlog=open("armlog.csv","w")
    armconfig=open("armconfig.txt","r")
    
 

#############################################
   #  key constants
    main_loop=True

    taught_position=[0,0,0,0]
    

    positions=[]    

    oldtime=time.process_time()

#############################################
    # reset encoders for position
    reset_to_first_position(BP)
    reset_encoders(BP)
    float_motors(BP)
  #  BP.set_motor_position(BP.PORT_B, 50)  # 7500 open grabber
  #  BP.set_motor_position(BP.PORT_C, 50)  # arm
   # time.sleep(5)
   # reset_encoders(BP)
   # float_motors(BP)
   
##################################################
    #  calibrate gyro
    try:
     
            timelog=time.process_time()
            # standardise the rate of processing at two every 1/1000 th of a second
            while timelog<oldtime+0.0002:   #0.0005
                timelog=time.process_time()
            oldtime=timelog
        

            teach_loop=True
            while teach_loop:
                answer=input("ready to teach? (y/n)")
                if answer=="y":
                    float_motors(BP)
                    taught_position=arm.arm_teach(BP)       
                    positions.append(taught_position)   #  keep the every 10th error in order to sum them for the integral (i)
                    #if self.call_count>=full_history_count:
                    #    del self.error_history[0]

                    print("positions=",positions)
                else:
                    teach_loop=False
                    
            
            n=0
            length=len(positions)
            #print("length=",length)
            input("hit enter to float motors")
            float_motors(BP)
            input("hit enter to start moving")
            while n<length:
                arm.move_joint_abs(BP,"SHOULDER",positions[n][0])
                arm.move_joint_abs(BP,"ELBOW",positions[n][1])
                arm.move_joint_abs(BP,"WRIST",positions[n][2])
                arm.move_joint_abs(BP,"HAND",positions[n][3])
                print("n=",n)
                time.sleep(4)
                #input("n=",n," ?")
                n+=1
                
        

            if read_touch_sensor(BP):
                print("touch sensor hit.")
                robotlog.write("touch sensor hit.\n")
                motor_speed=0
                turn=0
                main_loop=False
            
                
    
    except KeyboardInterrupt:
        BP.reset_all()        # Unconfigure the sensors, disable the motors, and restore the LED to the control of the BrickPi3 firmware.


    
    BP.reset_all()
    armlog.close()
    armconfig.close()
        
main()

