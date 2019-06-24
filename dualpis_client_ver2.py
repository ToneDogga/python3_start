
# dualpis_client.py

import socket                   # Import socket module

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
            print(sdata)
        if not data:
            break
        # write data to a file
 #       f.write(data)

#f.close()
#print("Successfully got the file")
except KeyboardInterrupt:
    s.close()
print("connection closed")
