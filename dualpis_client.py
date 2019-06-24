
# dualpis_client.py

import socket                   # Import socket module

s = socket.socket()             # Create a socket object
host= socket.gethostbyname("192.168.0.106")
#host = socket.gethostname()     # Get local machine name
port = 60000                    # Reserve a port for your service.

print("host=",host)
s.connect((host, port))
s.send(b'Hello server!')

with open("received_file.txt", "wb") as f:
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
