
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


           sdata=data.decode('utf-8')
        print(sdata)
        try:
            str(sdata)
            try:
                int(sdata)
                motor_pos=int(sdata)
                error_flag=False
            except ValueError:
                error_flag=True
                motor_pos=0
        except ValueError:
            error=True
            motor_pos=0
        if not error_flag and motor_pos!=old_motor_pos:
            #print(motor_pos)
            BP.set_motor_position(BP.PORT_A,motor_pos)

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
       
 


def convert_bytes_to_ds_values(b,ds):
    e=0
   # b=bytearray()
   # for row in ds:
   #     b.extend(ds[e][3].encode('utf-8'))        
   #     e+=1


    return ds
    

    


BP=brickpi3.BrickPi3()
BP.offset_motor_encoder(BP.PORT_A,BP.get_motor_encoder(BP.PORT_A))

data_struct=read_in_data_struct("/home/pi/Python_Lego_projects/data_struct.csv")
size, elements = interpret_ds(data_struct)
print("total size=",size," no of elements=",elements)    
print_data_struct(data_struct)
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
old_motor_pos=0


try:
    while True:
        #print("receiving data...")
        #time.sleep(0.02)
        #data = s.recv(1024)
      #  error_flag=False
        b=s.recv(size)
        if not b:
            break

        convert_bytes_to_ds_values(b,data_struct)
        send_data_struct_values_to_brickpi(data_struct,BP)
     #   sdata=data.decode('utf-8')
     #   print(sdata)
     #   try:
     #       str(sdata)
     #       try:
     #           int(sdata)
     #           motor_pos=int(sdata)
     #           error_flag=False
     #       except ValueError:
     #           error_flag=True
     #           motor_pos=0
     #   except ValueError:
     #       error=True
     #       motor_pos=0
     #   if not error_flag and motor_pos!=old_motor_pos:
            #print(motor_pos)
     #       BP.set_motor_position(BP.PORT_A,motor_pos)

except KeyboardInterrupt:
    s.close()
print("connection closed")
