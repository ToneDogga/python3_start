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


def reset_position(rp):
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
    Balance_PID=PID_control()   # controls error on balance angle
    Speed_PID=PID_control()    # controls error on motor speed
    Position_PID=PID_control()  # controls error on position from start
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
    deadzone_minangle=0  # plus or minus 2 degrees.  No speed adjustments in the deadzone
    deadzone_maxangle=0
    tipped_over_angle=25  # plus or minus 20 degrees.  Kill the motor and let the robot stop
    target_angle=0  # (-4) the angle the robot needs to be to balance.  The error angle is calculated from this
   

    balance_bias_angle=1  # (1.55)angle robot needs to be at to stay neutral in position

    target_angle=balance_bias_angle
    start_angle=0   # when the inner loop starts, the gyro starts at 0 as an absolute angle.  We need to find the relative angle
    actual_angle=0  # calculated
    start_target_angle=target_angle

    motor_speed=0   # until we get to turning, the motorb and motorc engine speed should be locked at the same
    command_speed=0  #speed command from remote
    command_position=0  # position on the encoder we want to stay at

    # Balance loop PID constants   angle error adjust wheel speed
    BKp=8.65      #(8.65) (9)proportional constant    
    BKi=0.16  # (0.16)(0.18)integral constant ( frequency=23 x 10)=230, period of occilation is approx 150. or frequency is 8x10=80, half of occilation
    BKd=0.18    # (0.18)differential constant 

    # Speed loop PID constants  speed error, adjust the target angle every 40 cycles
    SKp=0.05      # (0.05) (0.012)proportional constant 
    SKi=0.01     # (0.01)integral constant(x10) 0.0005 
    SKd=0      # (0.0)differential constant 


    # Position loop PID constants  position error, adjust the target angle every 40 cycles
    PKp=0.02      # (0.001) proportional constant 
    PKi=0     # (0.0)integral constant 
    PKd=0      # (0.0)differential constant 

 
    p=0
    i=0
    d=0

    p2=0
    i2=0
    d2=0

    p3=0
    i3=0
    d3=0


    
    i_error=[]  # angle error history
    s_error=[] # speed error history
    speed_history=[]   # sample the speed every 10th cycle
    position_history=[]  # sample the position evey 100th cycle
    
    speed_list_length=0  # number of elements in the speed_history list
    position_list_length=0 # number of elements in the position_history list


    turn=0

    oldtime=time.process_time()

#############################################
    # reset encoders for position

    reset_position(BP)
    
    try:
        BP.set_motor_power(BP.PORT_B, BP.MOTOR_FLOAT)    # float motor B
     #   print("Motor B floated")
    except IOError as error:
        print(error)

    try:
        BP.set_motor_power(BP.PORT_C, BP.MOTOR_FLOAT)    # float motor B
     #   print("Motor C floated")
    except IOError as error:
        print(error)
 
   
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
            while timelog<oldtime+0.0005:   #0.0005
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
                BP.set_motor_power(BP.PORT_B + BP.PORT_C, 0)
                motor_speed=0
                target_angle=start_target_angle
                tipped_over=True
                
                if round(actual_angle)==round(start_target_angle):
                    tipped_over=False
            
            else:

                # calculate correction motor speed based in a correction factor calculated by the inner PID function
              
                p,i,d,i_error=Balance_PID.PID_processor(actual_angle,target_angle,angle_rate,BKp,BKi,BKd,10,8)   # 8 current angle, command angle and gyro rate (as the differential element), integral number of samples, and sample rate

             #   if i>150:   # to prevent integral windup and a runaway
             #       i=150
             #   elif i<-150:
             #       i=-150

             #   if p>40:   # to prevent a runaway
             #       p=40
             #   elif p<-40:
             #       p=-40
    
                motor_speed=p+i+d

            if n%12==0:   # check every 12 cycles
 

                    
                speed_history.append(motor_speed)   #  keep the every 10th error in order to sum them for the integral (i)
                speed_list_length+=1
                if speed_list_length>10:   # list of 20
                    speed_list_length-=1
                    del speed_history[0]
                ave_motor_speed=sum(speed_history)/10

                p2,i2,d2,s_error=Speed_PID.PID_processor(ave_motor_speed,command_speed,0,SKp,SKi,SKd,10,7)
                #    print("outer p=",p," i=",i," d=",d)
                target_angle=-((p2+i2+d2)-start_target_angle)+balance_bias_angle

 
               





            if n%60==0:   # check every 60 cycles
                position = (BP.get_motor_encoder(BP.PORT_B)-BP.get_motor_encoder(BP.PORT_C))/2
                position_history.append(position)   #  keep the every 10th error in order to sum them for the integral (i)
                position_list_length+=1
                if position_list_length>10:   # list of 20
                    position_list_length-=1
                    del position_history[0]
                ave_position=sum(position_history)/10

                p3,i3,d3,p_error=Position_PID.PID_processor(ave_position,command_position,0,PKp,PKi,PKd,10,7)

                position_adjustment_angle=p3+i3+d3
                target_angle+=position_adjustment_angle
              #  print("position=",position," position adjustment angle=",p3+i3+d3)  

                remote=BP.get_sensor(BP.PORT_1)
                if not isinstance(remote[0],list):   # the [0] list is the first (top) channel
                    remote[0]=[0,0,0,0,0]                   #  buttons in order: left top,left  bottom, right top, right bottom, centre

                turn=0
                if remote[0][2]==1:   #Go faster or forward
                    target_angle-=0.005
                    reset_position(BP)
                elif remote[0][3]==1:  #go slower or reverse
                    target_angle+=0.005
                    reset_position(BP)
                elif remote[0][0]==1:  # turn left
                    turn=5
                elif remote[0][1]==1:  # turn right
                    turn=-5
             #   elif remote[0][4]==1:  # cancel turning
             #       turn=0



                        
            if actual_angle<deadzone_minangle or actual_angle>deadzone_maxangle:
    #               print("set motor speeds here;  speed=",motor_speed)
                      
                        
                if random.randint(1,2)==1:
                    BP.set_motor_power(BP.PORT_C, motor_speed+turn)
                    BP.set_motor_power(BP.PORT_B ,- motor_speed+turn)   # set_motor_speed
                else:
                    BP.set_motor_power(BP.PORT_B , -motor_speed+turn)   # set_motor_speed
                    BP.set_motor_power(BP.PORT_C, motor_speed+turn)


            else:
                BP.set_motor_power(BP.PORT_C+BP.PORT_B, BP.MOTOR_FLOAT)


      #      print("motor speed=",motor_speed,"target angle",target_angle," ave speed=",ave_motor_speed," speed histyr=",speed_history)
                    
          #  robotlog.write(" n="+str(n)+" actual a="+str(actual_angle)+" PID  actual s="+str(motor_speed)+" ta="+str(target_angle)+" tat "+str(target_angle_temp)+" rate:"+str(angle_rate)+" cs="+str(command_speed)+" p="+str(p)+" i="+str(i)+" d="+str(d)+" ehist="+str(i_error)+" shist="+str(s_error)+"t="+str(timelog)+"\n")
            robotlog.write(str(actual_angle)+","+str(position)+","+str(position_adjustment_angle)+","+str(target_angle)+","+","+str(p)+","+str(i)+","+str(d)+"\n")
  

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

