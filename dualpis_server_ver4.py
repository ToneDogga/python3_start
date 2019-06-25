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



def read_in_data_struct(file):

    #  column 0 is the name of the data
    # col 1 is its type (int, or string)
    # col 2 is its length
    # col 3 is the actual value


    ds=[]
    print("read in data structure")

    try:    
        with open(file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                ds.append(row)
        return ds
    
    except OSError:
        print("file ",file," not found.")
        return []

    

def print_data_struct(ds):
        print(" Print data structure ")
        print("------------")
        for elem in ds:
            print(elem)
        print("-----------")
        print("  ")
          
def interpret_ds(ds):
    #  column 0 is the name of the data
    # col 1 is its type (int, or string)
    # col 2 is its length
    # col 3 is the actual value
    size=0
    elements=0
    for row in ds:
        ds[elements][2]=int(ds[elements][2])
        size+=ds[elements][2]  
       # if ds[elements][1]=="int":
       #     ds[elements][3]=int(ds[elements][3])

        elements+=1    
    
    return size, elements



def load_data_struct_values(ds,rp):
        zero="0"
        ds[0][3]=str(zero.zfill(8))
        ds[1][3]=str(zero.zfill(8))
        ds[2][3]=str(zero.zfill(8))
        ds[3][3]=str(zero.zfill(8))
        
        try:
            a=str(rp.get_motor_encoder(BP.PORT_A)).zfill(8)
            a=a[0:8]
            ds[4][3]=a
        except IOError as error:
            print(error)

        try:
            b=str(rp.get_motor_encoder(BP.PORT_B)).zfill(8)
            b=b[0:8]
            ds[5][3]=b
        except IOError as error:
            print(error)

        try:
            c=str(rp.get_motor_encoder(BP.PORT_C)).zfill(8)
            c=c[0:8]
            ds[6][3]=c
        except IOError as error:
            print(error)

        try:
            d=str(rp.get_motor_encoder(BP.PORT_D)).zfill(8)
            d=d[0:8]
            ds[7][3]=d
        except IOError as error:
            print(error)
       
 

def convert_ds_values_to_bytes(ds):
    e=0
    b=bytearray()
    for row in ds:
        b.extend(ds[e][3].encode('utf-8'))        
        e+=1


    return b


#msg = bytearray()  # New empty byte array
# Append data to the array
#msg.extend(b"blah")
#msg.extend(b"foo") 
        #motor_pos_str=str(BP.get_motor_encoder(BP.PORT_A)).zfill(8)
        #motor_pos=motor_pos_str[0:8]
        #print("motor_pos=",motor_pos)
        #b = motor_pos.encode('utf-8')



BP=brickpi3.BrickPi3()
BP.offset_motor_encoder(BP.PORT_A,BP.get_motor_encoder(BP.PORT_A))

data_struct=read_in_data_struct("/home/pi/Python_Lego_projects/data_struct.csv")
size, elements = interpret_ds(data_struct)
print("total size=",size," no of elements=",elements)    
print_data_struct(data_struct)
#input("?")


# data_struct
    #  column 0 is the name of the data
    # col 1 is its type (int, or string)
    # col 2 is its length
    # col 3 is the actual value







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
print("Got connection with", addr)
data = conn.recv(1024)
print("Server received", repr(data))
print("Server sending encoder values...")
try:
    while True:
        load_data_struct_values(data_struct,BP)
        #print_data_struct(data_struct)
       # input("?")
        b=convert_ds_values_to_bytes(data_struct)
        
        #motor_pos_str=str(BP.get_motor_encoder(BP.PORT_A)).zfill(8)
        #motor_pos=motor_pos_str[0:8]
        #print("motor_pos=",motor_pos)
        #b = motor_pos.encode('utf-8')
        conn.send(b)
        
            



#break

except KeyboardInterrupt:
    BP.reset_all()
    print("Done sending")
       # conn.send(b'Thank you for connecting')
    conn.close()



BP.reset_all()
