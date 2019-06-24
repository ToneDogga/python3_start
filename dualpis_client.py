
# dualpis_client.py

import socket                   # Import socket module
import time
import sys
import brickpi3
import csv

#with open('myfile.csv') as csvfile:
#       readCSV = csv.reader(csvfeile, delimiter=',')
#    for row in readCSV:
#           lines=row



#def read_file():
global target
#	with open("



s = socket.socket()             # Create a socket object
host= socket.gethostbyname("192.168.0.106")
#host=socket.gethostbyname("localhost")
#host = socket.gethostname()     # Get local machine name
port = 60000                    # Reserve a port for your service.

print("host=",host)
s.connect((host, port))
s.send(b'Hello server!')

with open("received_file.csv", "wb") as f:
    print("file opened")
    while True:
        print("receiving data...")
        data = s.recv(1024)
        print("data=%s", (data))

	
        if not data:
            break
        # write data to a file
        f.write(data)


f.close()
print("Successfully got the file")
s.close()
print("connection closed")

row=0
print("opening received file.csv")
with open("received_file.csv","r") as f:

#	config=f.readlines().split(",")
	config=f.readlines()
	print("config=",config)
#	row+=1
f.close()
