# Anthony Paech's Raspbery pi / Brick Pi Lego robot project in python started 19/4/19
# by anthony paech
#
#
#  EV ultra sonic testing platform
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
colour = ["none", "Black", "Blue", "Green", "Yellow", "Red", "White", "Brown"]


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
   # BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_REFLECTED)

    BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_COLOR)
    time.sleep(2)

    
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
  #  config_colour_sensor()
    print("read colour sensor")
    try:
  #      BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_REFLECTED)        # Configure for an EV3 color sensor in reflected mode.
  #      time.sleep(0.02)
  #      value1 = BP.get_sensor(BP.PORT_3)                                           # get the sensor value
  #      print("value1= ",value1,"reflected mode colour=",colour[value1])
  #      BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_AMBIENT)          # Configure for an EV3 color sensor in ambient mode.
  #      time.sleep(0.02)
   #    value2 = BP.get_sensor(BP.PORT_3)                                        # get the sensor value
#    print("value2= ",value2,"ambient mode colour=")        
    #    BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_COLOR)            # Configure for an EV3 color sensor in color mode.
        time.sleep(0.2)
        value3 = BP.get_sensor(BP.PORT_3)                                        # get the sensor value
        print("value3= ",value3,"colour colour mode colour=",colour[value3])       
    #    BP.set_sensor_type(BP.PORT_3, BP.SENSOR_TYPE.EV3_COLOR_COLOR_COMPONENTS) # Configure for an EV3 color sensor in color components mode.
    #    time.sleep(0.02)
    #    value4 = BP.get_sensor(BP.PORT_3)                                        # get the sensor value
    #    print("value4= ",value4,"colour components mode colour=",colour[value4[0]])

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
    BP.set_sensor_type(BP.PORT_4, BP.SENSOR_TYPE.EV3_ULTRASONIC_CM) # Configure for an EV3 ultrasonic sensor.



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


    
def execute_command():
    # distance=read_ultrasonic_sensor()
      #  print("ultra sonic sensor dist= ", distance)
        read_colour_sensor()
        #print("colour=",colour)
      #  touch=read_touch_sensor()
      #  print("touch=",touch)
        return 




def main():
    d=0
    config_touch_sensor()
    config_colour_sensor()
    config_ultrasonic_sensor()
    try:
        while True:
            d=execute_command()
            time.sleep(0.1)
        else:
            print("command invalid")
    
    except KeyboardInterrupt:
        BP.reset_all()            

main()
BP.reset_all()        # Unconfigure the sensors, disable the motors, and restore the LED to the control of the BrickPi3 firmware.



