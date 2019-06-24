
# dualpis_client.py

import socket  # Import socket module
import brickpi3


BP=brickpi3.BrickPi3()


s = socket.socket()             # Create a socket object
host= socket.gethostbyname("192.168.0.110")
#host = socket.gethostname()     # Get local machine name
port = 60000                    # Reserve a port for your service.

print("host=",host)
s.connect((host, port))
s.send(b'Hello server!')

#with open("received_file.txt", "wb") as f:
#    print("file opened")
sdata=""
try:
    while True:
        #print("receiving data...")
        olddata=sdata

        #data = s.recv(1024)
        data=s.recv(6)
        sdata=data.decode('utf-8')
        if sdata != olddata:
           # print(sdata)
            motor_pos=int(sdata)
           # print("motor pos=",motor_pos)
            BP.set_motor_position(BP.PORT_A,motor_pos)
        if not data:
            break
        # write data to a file
 #       f.write(data)

#f.close()
#print("Successfully got the file")
except KeyboardInterrupt:
    s.close()
print("connection closed")
