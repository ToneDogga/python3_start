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
       
        
        

    def PID_processor(self,actual,command,diff,Kp,Ki,Kd,integral_sum_cycles,sample_rate):
        #  take the actual value and return an appropoirate correction factor to bring it to the command value
        #   the diff is passes from the gyro as the rate of angle change
       # integral_sum_cycles=10   # 100 seems to be the wavelength of the derivtive occilation at 2 cycles per 1/1000th of a sec
       # sample_rate=10   # the number of cycles per sample of error
        full_history_count=integral_sum_cycles*sample_rate

        error=command-actual
        if self.call_count%sample_rate==0:
    #        print("call=",self.call_count)
            self.error_history.append(error)   #  keep the every 10th error in order to sum them for the integral (i)
            if self.call_count>=full_history_count:
                del self.error_history[0]
        error_sum=sum(self.error_history)
        
        self.totalerror+=error
        p=Kp*error 
        
        i=Ki*error_sum    # total error for the last 10 calls

        if diff==0:
            d=Kd*(error-self.lasterror)
        else:    
            d=Kd*diff

        self.lasterror=error
        self.call_count+=1
     #   print("error history",self.error_history)
        #print("p=",p," i=",i," d=",d)
        
        return(p,i,d,self.error_history)   # return(p+i+d)
        
        

        





def main():
    
   #################################3
    #    initiate
    
    rp=robot3_lib.robot()   # instantiate a robot
    Inner_PID=PID_control()   # controls error on balance angle
    Outer_PID=PID_control()    # controls error on motor speed     
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
    time.sleep(2)
    
    BP.set_sensor_type(BP.PORT_1, BP.SENSOR_TYPE.EV3_INFRARED_REMOTE)

    time.sleep(4)   # wait while sensors get ready
    robotlog=open("robotlog.csv","w")

    main_loop=True
    n=0
   # time.clock_settime()


#############################################
   #  key constants
  #  deadzone_minangle=0  # plus or minus 2 degrees.  No speed adjustments in the deadzone
  #  deadzone_maxangle=0
    tipped_over_angle=24  # plus or minus 20 degrees.  Kill the motor and let the robot stop
    target_angle=1  # (-4) the angle the robot needs to be to balance.  The error angle is calculated from this
    start_target_angle=target_angle
    target_angle_temp=0

    start_angle=0   # when the inner loop starts, the gyro starts at 0 as an absolute angle.  We need to find the relative angle
    actual_angle=0  # calculated

    motor_speed=0   # until we get to turning, the motorb and motorc engine speed should be locked at the same
    command_speed=0  #speed command from remote


    # inner loop PID constants   angle error adjust wheel speed
    IKp=26        # (25)proportional constant (31)     total motor speed struggles to get > 1000 total seems to be 60p
    IKi=1.5  # (1.2)integral constant (x20) or 2.4 x18
    IKd=2    # (2.8)differential constant (6)  max rate is 160

    # outer loop PID constants  speed error, adjust the target angle
    OKp=0      # proportional constant 
    OKi=0     # integral constant(x10) 0.0005 
    OKd=0       # differential constant 
 
    p=0
    i=0
    d=0

    p2=0
    i2=0
    d2=0
    
    i_error=[]  # angle error history
    s_error=[] # speed error history

    min_speed_limit=50
    max_speed_limit=5000

    turn=0
    
   # low_power_percent=25
    normal_power_percent=25
   # emergency_power_percent=25
    oldtime=time.process_time()

    #BP.set_motor_limits(BP.PORT_C+BP.PORT_C, normal_power_percent, max_speed_limit)
    BP.set_motor_limits(BP.PORT_C+BP.PORT_C, 100, 5000)


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
   # robotlog.write("starting angle="+str(start_angle)+"\n")
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

            
            timelog=time.process_time()
            # standardise the rate of processing at two every 1/1000 th of a second
            while timelog<oldtime+0.0005:
                  timelog=time.process_time()
            oldtime=timelog
        

            
          # get gyro angle
            try:
                angle=rp.gyro_angle()   # returns a tuple, angle and rate of change
            except IOError as error:
                print(error)
            #else: angle=old_angle

            if not isinstance(angle,list):
                angle=[0,0]
                
         #   time.sleep(0.02)
            absolute_angle=angle[0]
            angle_rate=angle[1]
            actual_angle=-absolute_angle-start_target_angle
            
            old_angle=angle  
    
# is the robot tipped over?

            if abs(actual_angle)>tipped_over_angle or tipped_over:    
             #   print("tipped over! actual angle=",actual_angle)
          #      robotlog.write("TIPPED OVER. time="+str(timelog)+" actual angle="+str(actual_angle)+" relative angle "+str(relative_angle)+", rate:"+str(angle_rate)+" p="+str(p)+" i="+str(i)+" d="+str(d)+" PID motor speed="+str(motor_speed)+" Kp="+str(Inner_PID.Kp)+" Ki="+str(Inner_PID.Ki)+" Kd="+str(Inner_PID.Kd)+"\n")
                BP.set_motor_dps(BP.PORT_B + BP.PORT_C, 0)
                motor_speed=0
                tipped_over=True
                
                if round(actual_angle)==round(target_angle):
                    tipped_over=False
            
            else:

                # calculate correction motor speed based in a correction factor calcultaed by the inner PID function
              
                p,i,d,i_error=Inner_PID.PID_processor(actual_angle,target_angle,angle_rate,IKp,IKi,IKd,10,17)   # 12, 5 current angle, command angle and gyro rate (as the differential element), integral number of samples, and sample rate

                #if p>300:
                #    p=300
                #elif p<-300:
                #    p=-300
                motor_speed=p+i+d

                if n%10==0:   # check every 10 cycles
                    remote=BP.get_sensor(BP.PORT_1)
                    if not isinstance(remote[0],list):   # the [0] list is the first (top) channel
                        remote[0]=[0,0,0,0,0]                   #  buttons in order: left top,left  bottom, right top, right bottom, centre

                    if remote[0][2]==1:   #Go faster or forward
                        target_angle-=0.03
                    elif remote[0][3]==1:  #go slower or reverse
                        target_angle+=0.03
                    elif remote[0][0]==1:  # turn left
                        turn+=10
                    elif remote[0][1]==1:  # turn right
                        turn-=10
                    elif remote[0][4]==1:  # cancel turning
                        turn=0

                    p2,i2,d2,s_error=Outer_PID.PID_processor(motor_speed,command_speed,0,OKp,OKi,OKd,15,1)
                #    print("outer p=",p," i=",i," d=",d)
                    target_angle_temp=(p+i+d)+start_target_angle
                    if target_angle>tipped_over_angle:
                        target_angle=tipped_over_angle
                      
                        
                if random.randint(1,2)==1:
                    BP.set_motor_dps(BP.PORT_C, motor_speed+turn)
                    BP.set_motor_dps(BP.PORT_B ,- motor_speed+turn)   # set_motor_speed
                else:
                    BP.set_motor_dps(BP.PORT_B , -motor_speed+turn)   # set_motor_speed
                    BP.set_motor_dps(BP.PORT_C, motor_speed+turn)
                    
          #  robotlog.write(" n="+str(n)+" actual a="+str(actual_angle)+" PID  actual s="+str(motor_speed)+" ta="+str(target_angle)+" tat "+str(target_angle_temp)+" rate:"+str(angle_rate)+" cs="+str(command_speed)+" p="+str(p)+" i="+str(i)+" d="+str(d)+" ehist="+str(i_error)+" shist="+str(s_error)+"t="+str(timelog)+"\n")
            robotlog.write(str(actual_angle)+","+str(p)+","+str(i)+","+str(d)+"\n")
  

            if rp.read_touch_sensor():
                print("touch sensor hit.")
            #    robotlog.write("touch sensor hit.\n")
                motor_speed=0
                turn=0
                main_loop=False
            
            n+=1    
    
    except KeyboardInterrupt:
        rp.reset_all()        # Unconfigure the sensors, disable the motors, and restore the LED to the control of the BrickPi3 firmware.


    
    rp.reset_all()
    robotlog.close()

        
main()

