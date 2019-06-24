# dualpis_server.py


from __future__ import print_function # use python 3 syntax but make it compatible with python 2
from __future__ import division       #                           ''

import math
#import pygame
#import turtle
import random
import sys
import socket   # socket module for live file transfer
import csv      # for importing the decision tables as .csv files
import time     # import the time library for the sleep function
import brickpi3 # import the BrickPi3 drivers


BP = brickpi3.BrickPi3() # Create an instance of the BrickPi3 class. BP will be the BrickPi3 object.





port = 60000                    # Reserve a port for your service.
s = socket.socket()             # Create a socket object

#host = socket.gethostname()     # Get local machine name

host = socket.gethostbyname("192.168.0.110") # Get local machine name
#host = socket.gethostbyname("localhost") # Get local machine name


print("host=",host)

s.bind((host, port))            # Bind to the port
s.listen(5)                     # Now wait for client connection.

print("Server listening....")




conn, addr = s.accept()     # Establish connection with client.
print("Got connection from", addr)
data = conn.recv(1024)
print("Server received", repr(data))
print("Server sending encoder values...")
try:
    while True:
  #  r=str(row)+"\n"
        motor_pos_str=str(BP.get_motor_encoder(BP.PORT_A)).zfill(6)
        motor_pos=motor_pos_str[0:6]
        b = motor_pos.encode('utf-8')
        conn.send(b)
        
            


    print("Done sending")
       # conn.send(b'Thank you for connecting')
    conn.close()
#break

except KeyboardInterrupt:
    BP.reset_all()

BP.reset_all()
