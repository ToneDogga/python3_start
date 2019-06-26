
# dualpis_client.py

import socket  # Import socket module
import brickpi3
import time
import csv


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
        if ds[elements][1]=="int":
            ds[elements][3]=int(ds[elements][3])
        elements+=1
    
    return size, elements


def send_data_struct_values_to_brickpi(ds,rp):
    #print(".")

#        sdata=data.decode('utf-8')
 #       print(sdata)

   # portnames=['PORT_1','PORT_2','PORT_3','PORT_4','PORT_A','PORT_B','PORT_C','PORT_D']
     
    for field in range(4,7):    
        try:
            str(ds[field][3])
            try:
                int(ds[field][3])
                motor_pos=int(ds[field][3])
                error_flag=False
            except ValueError:
                error_flag=True
                motor_pos=0
        except ValueError:
            error=True
            motor_pos=0
        if not error_flag:   # and motor_pos!=old_motor_pos:
            #print(motor_pos)
            if field==4:
                BP.set_motor_position(BP.PORT_A,motor_pos)
            elif field==5:
                BP.set_motor_position(BP.PORT_B,motor_pos)
            elif field==6:
                BP.set_motor_position(BP.PORT_C,motor_pos)
            elif field==7:
                BP.set_motor_position(BP.PORT_D,motor_pos)



                

 


def convert_bytes_to_ds_values(b,ds):
    lenb=64
    info = [b[i:i+8].decode('utf-8') for i in range(0, lenb, 8)]
    for field in range(0,7):
        ds[field][3]=info[field]

    return ds
    

    


BP=brickpi3.BrickPi3()
BP.offset_motor_encoder(BP.PORT_A,BP.get_motor_encoder(BP.PORT_A))
BP.offset_motor_encoder(BP.PORT_B,BP.get_motor_encoder(BP.PORT_B))
BP.offset_motor_encoder(BP.PORT_C,BP.get_motor_encoder(BP.PORT_C))
BP.offset_motor_encoder(BP.PORT_D,BP.get_motor_encoder(BP.PORT_D))




data_struct=read_in_data_struct("/home/pi/Python_Lego_projects/data_struct.csv")
size, elements = interpret_ds(data_struct)
print("total size=",size," no of elements=",elements)    
#print_data_struct(data_struct)
#input("?")




s = socket.socket()             # Create a socket object
host= socket.gethostbyname("192.168.0.110")
#host = socket.gethostname()     # Get local machine name
port = 60000                    # Reserve a port for your service.

print("host=",host)
s.connect((host, port))
s.send(b'Hello server!')


motor_pos=0
BP.set_motor_position(BP.PORT_A,motor_pos)
BP.set_motor_position(BP.PORT_B,motor_pos)
BP.set_motor_position(BP.PORT_C,motor_pos)
BP.set_motor_position(BP.PORT_D,motor_pos)



#old_motor_pos=0


try:
    while True:
        b=s.recv(size)
        if not b:
            break

        convert_bytes_to_ds_values(b,data_struct)
        send_data_struct_values_to_brickpi(data_struct,BP)

except KeyboardInterrupt:
    s.close()
print("connection closed")
