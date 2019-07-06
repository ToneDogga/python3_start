# chat_client.py

import sys, socket, select
import hashlib
import time



# client2.py
#!/usr/bin/env python

#import socket
#import time

#TCP_IP="192.168.0.110"
#TCP_IP = "localhost"
#TCP_PORT = 9001
#BUFFER_SIZE = 1024

#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect((TCP_IP, TCP_PORT))

def file_receive(s):
    clock_start=time.clock()

    #time_start=time.time()
    filename="received_file.csv"

    with open(filename, 'wb') as f:
        print(filename," opened")
        while True:
        #print('receiving data...')
            data = s.recv(BUFFER_SIZE)
        #print("data=%s", (data))
            if not data:
                f.close()
                print(filename," close()")
                break
            # write data to a file
            f.write(data)

    print("Successfully got the file")
    #s.close()
    #print("connection closed")

    clock_end=time.clock()
    #time_end=time.time()

    duration_clock=clock_end-clock_start

    print("Clock: start=",clock_start," end=",clock_end)
    print("Clock: duration_clock =", duration_clock)

    #duration_time=time_end-time_start

    #print("Time: start=",time_start," end=",time_end)
    #print("Time: duration_time =", duration_time)



def file_send(s):
    clock_start=time.clock()

    filename='mytext.csv'
    f = open(filename,'rb')
    while True:
        l = f.read(BUFFER_SIZE)
        while (l):
            s.send(l)   #self.sock.send(l)
            #print('Sent ',repr(l))
            l = f.read(BUFFER_SIZE)
            if not l:
                f.close()
                s.close()
                break

    clock_end=time.clock()
    #time_end=time.time()

    duration_clock=clock_end-clock_start

    print("Clock: start=",clock_start," end=",clock_end)
    print("Clock: duration_clock =", duration_clock)




def hexhash_msg(msg):
    return (hashlib.md5(str(msg).encode('utf-8')).hexdigest())

def hash_msg(msg):
    return (hashlib.md5(str(msg).encode('utf-8')).digest()) #.digest()

def chat_client():
    if(len(sys.argv) < 3) :
        print("Usage : python3 Multi_pis_chat_client.py hostname port")
        sys.exit()

    host = sys.argv[1]
    port = int(sys.argv[2])
     
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
     
    # connect to remote host
    try :
        s.connect((host, port))
    except :
        print("Unable to connect")
        sys.exit()
     
    print("Connected to remote host. You can start sending messages")
    sys.stdout.write("[Me] "); sys.stdout.flush()
     
    while True:
        socket_list = [sys.stdin, s]
         
        # Get the list sockets which are readable
        read_sockets, write_sockets, error_sockets = select.select(socket_list , [], [])
         
        for sock in read_sockets:            
            if sock == s:
                # incoming message from remote server, s
                data = sock.recv(4096).decode('utf-8')
                if not data :
                    print("\nDisconnected from chat server")
                    sys.exit()
                else :
                    print(str(sock.getpeername())," : ",data," hash=",hash_msg(data))
                    
                    sys.stdout.write(str(sock.getpeername())+" : "+data)
                    sys.stdout.write("[Me] "); sys.stdout.flush()     
            
            else :
                # user entered a message
                msg = sys.stdin.readline()
                byte_msg=bytearray(msg.encode('utf-8'))
                print("msg byte array=",byte_msg)
                byte_msg.extend(hash_msg(msg))
                print("msg : ",msg," byte_msg with hash=",byte_msg)

                
                s.send(byte_msg)
                sys.stdout.write("[Me] "); sys.stdout.flush() 

if __name__ == "__main__":

    sys.exit(chat_client())
