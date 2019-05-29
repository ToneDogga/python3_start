# Anthony Paech's Raspbery pi / Brick Pi Lego robot project in python started 25/5/19
# by anthony paech
#
#   Goal :  build a raspberry pi python coded two wheeled balancing robot that moves to a lego RC controller
#
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
import pygame
# from pygame.locals import *
import robot3_lib  # brickpi robot class, motor and sensor functions
import brickpi3   # brickpi dexter library



def calibrate_gyro(rp):
    starting_angle=0
    print("Set robot up ready to balance. wait 1.7 seconds.")
   # while not rp.read_touch_sensor():
    time.sleep(4)

   # while c<7:
   # try:
    starting_angle=rp.gyro_angle()[0]
    #except IOError as error:
    #        print(error)
    #else: starting_angle=0
    
    #finally:
        
  #  time.sleep(0.5)
    print("starting angle=",starting_angle)
    time.sleep(4)
    
    return starting_angle


#def float_motorA():
    
#BP.offset_motor_encoder(BP.PORT_A, BP.get_motor_encoder(BP.PORT_A)) # reset encoder
#            print("Motor A reset encoder done")
#        except IOError as error:
#            print(error)3


#        try:
#            BP.set_motor_power(BP.PORT_A, BP.MOTOR_FLOAT)
   


class PID_control:
    def __init__(self):
        self.command_angle=0      # in degrees
        self.current_angle=0
        self.error_history=[]   # keep the last 10 error measurements to calculate the sum over time

        self.call_count=0   # counting the number of times it has been called to keep the last 10 errors
        
        self.command_speed=0      # motor speed in degrees per second
        self.current_speed=0

        self.totalerror=0
        self.lasterror=0
       
        self.Kp=50          # proportional constant (1.8)
        self.Ki=10      # integral constant (6)
        self.Kd=0.009        # differential constant (0.025)

    def PID_tuning(self,loopname):   
        print(loopname," PID tuning: Kp=,",self.Kp," Ki=",self.Ki," Kd=",self.Kd)
        self.Kp=input("Kp=")
        self.Ki=input("Ki=")
        self.Kd=input("Kd=")
        #   setup adjustments of these constants here
       
        
        

    def PID_processor(self,actual,command,diff):
        #  take the actual value and return an appropoirate correction factor to bring it to the command value
        #   the diff is passes from the gyro as the rate of angle change
        integral_sum_cycles=10
        sample_rate=15   # the number of cycles per sample of error
        full_history_count=integral_sum_cycles*sample_rate

        error=command-actual
        if self.call_count%sample_rate==0:
    #        print("call=",self.call_count)
            self.error_history.append(error)   #  keep the every 10th error in order to sum them for the integral (i)
            if self.call_count>=full_history_count:
                del self.error_history[0]
        error_sum=sum(self.error_history)
        
        self.totalerror+=error
        p=self.Kp*error 
        
        i=self.Ki*error_sum    # total error for the last 10 calls

        if diff==0:
            d=self.Ki*(error-self.lasterror)
        else:    
            d=self.Ki*diff

        self.lasterror=error
        self.call_count+=1
     #   print("error history",self.error_history)
        #print("p=",p," i=",i," d=",d)
        
        return(p,i,d,self.error_history)   # return(p+i+d)
        
        

        





def main():
    
   #################################3
    #    initiate
    
    rp=robot3_lib.robot()   # instantiate a robot
    Inner_PID=PID_control()
    BP=brickpi3.BrickPi3()
  #  pygame.init()
   # Screen_Width = 80
   # Screen_Height = 60

    #Total_Display = pygame.display.set_mode((Screen_Width, Screen_Height))
    #Total_Display.fill((255,0,0))

    print("Keep robot still for gyro config!")
    time.sleep(0.1)

    rp.config_touch_sensor()  # sensor port 3
    
    rp.motorC_reset()   # port MC
    rp.motorB_reset()    # port MB
    rp.configure_gyro()    #Sensor port 2
   # rp.configEV3_ultrasonic_sensorS1()
   # rp.configNXT_ultrasonic_sensorS4()

    time.sleep(4)   # wait while sensors get ready
    robotlog=open("robotlog.txt","w")

    main_loop=True
    n=0
   # time.clock_settime()


#############################################
   #  key constants
    deadzone_minangle=-1  # plus or minus 2 degrees.  No speed adjustments in the deadzone
    deadzone_maxangle=0
    tipped_over_angle=28  # plus or minus 20 degrees.  Kill the motor and let the robot stop
    perfect_angle=0  # the angle the robot needs to be to balance.  The error angle is calculated from this
    start_angle=0   # when the inner loop starts, the gyro starts at 0 as an absolute angle.  We need to find the relative angle
    

    motor_speed=0   # until we get to turning, the motorb and motorc engine speed should be locked at the same
    motorB_speed=0
    motorC_speed=0

    p=0
    i=0
    d=0
    i_error=[]

    min_speed_limit=50
    max_speed_limit=5000
    low_power_percent=10
    normal_power_percent=20
    emergency_power_percent=30

    dead_zone=False
    floated_motor=False
    emerg_power=False

    #BP.set_motor_limits(BP.PORT_C, power_percent, max_speed_limit)          # optionally set a power limit (in percent) and a speed limit (in Degrees Per Second)
    #BP.set_motor_limits(BP.PORT_B, power_percent, max_speed_limit)          # optionally set a power limit (in percent) and a speed limit (in Degrees Per Second)
 
   
##################################################
    #  calibrate gyro

    
    start_angle=calibrate_gyro(rp)
    time.sleep(1)
    print("starting angle=",start_angle)
    robotlog.write("starting angle="+str(start_angle)+"\n")
    #input("?")

 #############################333

    print("starting NOW. start angle=",start_angle)
    tipped_over=False
    try:
        while main_loop:

#   two main loops.  and inner loop which controls the balance by adjusting the wheel speed to correct the angle error (current angle - command angle)
#   using a PID algorithm.  this keeps the robot balanced.
# the outer loop takes the RC input and compared it to the current wheel speed (current speed - command speed)
# this goes through another different PID which processes the speed error and adjusts the command angle

          
           # timelog=time.process_time()      
# get gyro angle
            try:
                angle=rp.gyro_angle()   # returns a tuple, angle and rate of change
            except IOError as error:
                print(error)
            #else: angle=old_angle

            #  time.sleep(0.02)
            absolute_angle=angle[0]
            angle_rate=angle[1]
            relative_angle=start_angle-absolute_angle
            actual_angle=relative_angle-perfect_angle
            old_angle=angle  
    
# is the robot tipped over?

            if abs(actual_angle)>tipped_over_angle or tipped_over:    
             #   print("tipped over! actual angle=",actual_angle)
          #      robotlog.write("TIPPED OVER. time="+str(timelog)+" actual angle="+str(actual_angle)+" relative angle "+str(relative_angle)+", rate:"+str(angle_rate)+" p="+str(p)+" i="+str(i)+" d="+str(d)+" PID motor speed="+str(motor_speed)+" Kp="+str(Inner_PID.Kp)+" Ki="+str(Inner_PID.Ki)+" Kd="+str(Inner_PID.Kd)+"\n")
                BP.set_motor_dps(BP.PORT_B + BP.PORT_C, 0)
                motor_speed=0
                tipped_over=True
                
                if relative_angle==perfect_angle:
                    tipped_over=False
                # Inner_PID.PID_tuning("Inner")
            else:

                # calculate correction motor speed based in a correction factor calcultaed by the inner PID function
              
                p,i,d,i_error=Inner_PID.PID_processor(relative_angle,perfect_angle,angle_rate)   #  current angle, command angle and gyro rate (as the differential element)
                motor_speed=p+i+d
                
                
           #     if motor_speed>max_speed_limit or motor_speed<-max_speed_limit:
           #         motor_speed=max_speed_limit

                         
                    

            #    print("actual angle=",actual_angle," rel angle:",relative_angle,"angle_rate=",angle_rate," p=",p," i=",i," d=",d," motor speed=",motor_speed)
                    
                if relative_angle<deadzone_minangle or relative_angle>deadzone_maxangle:
    #               print("set motor speeds here;  speed=",motor_speed)
                    dead_zone=False
                    if abs(motor_speed)<300:
                        emerg_power=False
                        BP.set_motor_limits(BP.PORT_C+BP.PORT_C, low_power_percent, max_speed_limit)
                      #  BP.set_motor_power(BP.PORT_B+BP.PORT_C,low_power_percent)
                     #   BP.set_motor_power(BP.PORT_C,low_power_percent)
                    elif abs(motor_speed)>=300 and abs(motor_speed)<2000:
                        emerg_power=False
                        BP.set_motor_limits(BP.PORT_C+BP.PORT_C, normal_power_percent, max_speed_limit)
                      #  BP.set_motor_power(BP.PORT_C+BP.PORT_B,normal_power_percent)
                        #BP.set_motor_power(BP.PORT_C,normal_power_percent)
                    elif abs(motor_speed)>=2000:
                        emerg_power=True
                        BP.set_motor_limits(BP.PORT_C+BP.PORT_C, emergency_power_percent, max_speed_limit)
                      #  BP.set_motor_power(BP.PORT_C+BP.PORT_B,emergency_power_percent)
                     #   BP.set_motor_power(BP.PORT_C,emergency_power_percent)
                      
                        
                    if random.randint(1,2)==1:
                        BP.set_motor_dps(BP.PORT_C, motor_speed)
                        BP.set_motor_dps(BP.PORT_B , -motor_speed)   # set_motor_speed
                    else:
                        BP.set_motor_dps(BP.PORT_B , -motor_speed)   # set_motor_speed
                        BP.set_motor_dps(BP.PORT_C, motor_speed)
                    
                else:
                    dead_zone=True
                    if abs(motor_speed)<min_speed_limit:
                        floated_motor=True
                        motor_speed=0
                        BP.set_motor_dps(BP.PORT_C+BP.PORT_B, 0)
                        #BP.set_motor_dps(BP.PORT_C, 0)
                        BP.set_motor_power(BP.PORT_C+BP.PORT_B, BP.MOTOR_FLOAT)
                        #BP.set_motor_power(BP.PORT_C, BP.MOTOR_FLOAT)
                    else:
                        floated_motor=False
                    
                        
                    

           
         #  print("n=",n," time=",timelog," angle=",relative_angle," rate:",angle_rate," Motor speed=",motor_speed)
            
            #if abs(actual_angle)<=tipped_over_angle and angle_rate!=0:
            robotlog.write("n="+str(n)+" actual a="+str(actual_angle)+" ra "+str(relative_angle)+" emerg p="+str(emerg_power)+" dz="+str(dead_zone)+" fm="+str(floated_motor)+" rate:"+str(angle_rate)+" p="+str(p)+" i="+str(i)+" d="+str(d)+" PID ms="+str(motor_speed)+" Kp="+str(Inner_PID.Kp)+" Ki="+str(Inner_PID.Ki)+" Kd="+str(Inner_PID.Kd)+" eh="+str(i_error)+"\n")
             

            if rp.read_touch_sensor():
                print("touch sensor hit.")
                robotlog.write("touch sensor hit.\n")
                main_loop=False
            
            n+=1    
    
    except KeyboardInterrupt:
        rp.reset_all()        # Unconfigure the sensors, disable the motors, and restore the LED to the control of the BrickPi3 firmware.


    
    rp.reset_all()
    robotlog.close()

        
main()

