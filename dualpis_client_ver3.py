
# dualpis_client.py

import socket  # Import socket module
import brickpi3
import time


BP=brickpi3.BrickPi3()
BP.offset_motor_encoder(BP.PORT_A,BP.get_motor_encoder(BP.PORT_A))

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
        error_flag=False
        data=s.recv(8)
        if not data:
            break

        sdata=data.decode('utf-8')
        
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

except KeyboardInterrupt:
    s.close()
print("connection closed")
