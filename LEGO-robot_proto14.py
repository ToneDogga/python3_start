# Anthony Paech's Raspbery pi / Brick Pi Lego robot project in python started 19/4/19
# by anthony paech
#
#   Goal :  build a tracked vehicle controlled by a python code
#
#
#!/usr/bin/env python
#
# https://www.dexterindustries.com/BrickPi/
# https://github.com/DexterInd/BrickPi3
#

from __future__ import print_function # use python 3 syntax but make it compatible with python 2
from __future__ import division       #                           ''


import time    
import robotv1  # brickpi robot class, motor and sensor functions



def main():
    
    goal_count=0
   
    rp=robotv1.robot()   # instantiate a robot

    rp.read_in_sensor_speed_decision_matrix()  # read in from a CSV file
    rp.read_in_turn_decision_matrix()   # read in from a CSV file
  #  input("?")
    #rp.config_colour_sensor()
 #   rp.config_touch_sensor()
    time.sleep(2)
    
   # rp.motorC_reset()
   # rp.motorB_reset()
   # rp.configure_gyro()
    rp.configNXT_ultrasonic_sensorS3()
    rp.configNXT_ultrasonic_sensorS4()
    rp.configEV3_ultrasonic_sensorS1()
    time.sleep(5)   # wait while sensors get ready
    robotlog=open("robotlog.txt","w")

    rp.posx=500
    rp.posy=200
    rp.direction=0  # Forward down the table. Orientation = "F"
    
    time.sleep(1)

    loop=True

    try:
     #   while loop:
            succ=False
            succ=rp.goal_seek(2000,4000)     #to_visit[move])
           
            if succ:
                goal_count+=1
                print("goal number=",goal_count," reached.\n\n\n")
                


            #succ=rp.goal_seek(400,400)     #to_visit[move])


    
    except KeyboardInterrupt:
        rp.reset_all()        # Unconfigure the sensors, disable the motors, and restore the LED to the control of the BrickPi3 firmware.


    
    rp.reset_all()
    robotlog.close()

        
main()

