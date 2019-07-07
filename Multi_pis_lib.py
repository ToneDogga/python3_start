import numpy as np
from functools import partial
import socket
import hashlib
import time

#import sys

#import logging
#import os
 
#logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
#log = logging.getLogger(__name__)

def process_data(data):
    print(data)
    print(data.decode('utf-8'))
    print(repr(data))


def read_config():
    global target
    with open('/home/pi/Python_Lego_projects/myfile.csv','r') as f:
        config=f.readline().split(',')
        print(config)
        setpoint=float(config[0])



def read_bin():
    chunk_size=1024
    with open("myfile.csv", "rb") as f:
        for chunk in iter(partial(f.read, chunk_size), b""):
            #process_data(chunk)
            write_bin(chunk)
    f.close()        
    

def write_frame(filename,frame):
    with open(filename, "ab") as f:
        f.write(frame)
    f.close()    


def frame_test():
    
        #read_config()
    #read_bin()

    #log.debug("debug!")
    to_ip="192.168.0.110"
    from_ip="192.168.0.105"

    clock_start=time.clock()

    create_frame_file("myfile.csv",from_ip,to_ip)
    
    clock_end=time.clock()

    duration_clock=clock_end-clock_start

    print("Clock: start=",clock_start," end=",clock_end)
    print("Clock: duration_clock =", duration_clock)
    print("\n")


    clock_start=time.clock()

    read_frame_file("myfile.bin")
    
    clock_end=time.clock()

    duration_clock=clock_end-clock_start

    print("Clock: start=",clock_start," end=",clock_end)
    print("Clock: duration_clock =", duration_clock)
    print("\n")




#    from_ip_bytes=socket.inet_aton(from_ip)
    
#    print(ip_bytes,"size=",len(ip_bytes))
#    ip_str=socket.inet_ntoa(ip_bytes)
#    log.debug(ip_str)
#    print("ip str=",ip_str)



"""
        byte = f.read(1)
        while byte != b"":
            # Do stuff with byte.
            print(byte)
            byte = f.read(1)
"""

def hash_msg(msg):
    return (hashlib.md5(str(msg).encode('utf-8')).digest()) #.digest()

def create_frame(byte_frame,from_ip_bytes,to_ip_bytes):
#    byte_frame=bytearray(chunk)
    byte_frame.extend(from_ip_bytes)
    byte_frame.extend(to_ip_bytes)
    #print("byte frame=",byte_frame," len=",len(byte_frame))
    hash_frame=hash_msg(byte_frame)
    #print("hash frame=",hash_frame," len=",len(hash_frame))
    byte_frame.extend(hash_frame)
    print("\nfinal frame=",byte_frame)
    print("frame length=",len(byte_frame))
    return(byte_frame)
 

def create_frame_file(filename,from_ip,to_ip):    
    # read 1000 bytes from a csv file
    # append a to_ip address 4 bytes
    # append a from_ip address 4 bytes
    # hash to contents
    # append the hash  16 bytes
    # should have a 1024 bytes frame

    # write this frame to a binary file

    # close binary file

    to_ip_bytes=socket.inet_aton(to_ip)
    from_ip_bytes=socket.inet_aton(from_ip)

    chunk_size=1000
    with open(filename, "rb") as f:
        for chunk in iter(partial(f.read, chunk_size), b""):
            chunk=bytearray(chunk)
            extra_zeros=chunk_size-len(chunk)
            if extra_zeros>0:
                chunk.extend(b'\x00' * extra_zeros)
                             
            byte_frame=create_frame(chunk,from_ip_bytes,to_ip_bytes)
            write_frame("myfile.bin",byte_frame)
    f.close()        



 


def read_frame_file(filename):
    input("?")
    # open binary file
    #
    # read 1024 chunk which is a frame
    # split of the last 16 bytes which is the hash
    # hash the remaining 1008 bytes
    # compare with the original hash in the frame
    # if different error, try again or stop
    # split the first 1000 bytes as the data
    # convert to text
    # split the to_ip address and from _ip_address
    # convert to string
    # print
    


"""            
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

# Use a compound data type for structured arrays
data = np.zeros(4, dtype={"names":("name", "age", "weight"),
                          "formats":('U10', 'i4', 'f8')})
print(data.dtype)


data["name"] = name
data["age"] = age
data["weight"] = weight
print(data)

# Get all names
print(data["name"])

# Get first row of data
print(data[0])

# Get the name from the last row
print(data[-1]["name"])

# Get names where age is under 30
print(data[data['age'] < 30]['name'])

matrix = np.zeros((3,5), dtype = int)

matrix=np.array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 2, 0]])

print(matrix)


print("data3\n")
data3=np.dtype([("name", 'S10'), ("age", 'i4'), ("weight", 'f8')])

#data3=data
data["name"] = name
#data3["age"] = age
#data3["weight"] = weight
print(data3.dtype)

# Get all names
print(data3["name"])

# Get first row of data
print(data3[0])

# Get the name from the last row
print(data3[-1]["name"])

# Get names where age is under 30
print(data3[data3['age'] < 30]['name'])
"""

#create_config()
frame_test()


"""
print("data4\n")
data4=np.dtype('S10,i4,f8')

data4=data
#data4["name"] = name
#data4["age"] = age
#data4["weight"] = weight
print(data4)

# Get all names
print(data4["name"])

# Get first row of data
print(data4[0])

# Get the name from the last row
print(data4[-1]["name"])

# Get names where age is under 30
print(data4[data4['age'] < 30]['name'])

"""

"""Character	Description	Example
'b'	Byte	np.dtype('b')
'i'	Signed integer	np.dtype('i4') == np.int32
'u'	Unsigned integer	np.dtype('u1') == np.uint8
'f'	Floating point	np.dtype('f8') == np.int64
'c'	Complex floating point	np.dtype('c16') == np.complex128
'S', 'a'	String	np.dtype('S5')
'U'	Unicode string	np.dtype('U') == np.str_
'V'	Raw data (void)	np.dtype('V') == np.void
"""





        
