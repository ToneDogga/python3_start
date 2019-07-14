# Anthony Paech's Raspbery pi AI project July 2019
# by anthony paech
#
#   Goal :  build an algorithim to fit csv file data to an output using linear regression
#   and other techniques, that is run in parallel on multiple pis similatously using wireless TCP socket frame transfers

#
#!/usr/bin/env python
#



#from __future__ import print_function # use python 3 syntax but make it compatible with python 2
from __future__ import division       #                           ''

#import pandas
import numpy as np
from functools import partial
import socket
import hashlib
import time
import csv
import sys
import math
import os
import select

import threading
import queue
import pygame
from pygame.locals import *

import base64


RECV_BUFFER = 4096 




#import itertools

#import logging

#logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
#log = logging.getLogger(__name__)


#import pygame
#import turtle
#import random


# first testing
# mp.create_frame_file calls mp.pack_frame reads a big csv file and creates socketdata.bin
#  mp.read_frame_file calls mp.unpack_frame reads socketdata.bin creates an outfile.csv
# which matches the 
#
# next level testing
# a text msg is given to mp.send_msg which calls mp.pack_frame and sends it to the ip_to address
# creates a 1024 byte frame and then sends it to to_ip address using mp.send_frame
#
# mp.receive_msg calls mp.unpack_frame waits for socket communications from_ip address
# recives 1024 byte frame using mp.receive_frame
# check hash and extract data as a text list


#HOST = '' 
#HOST="192.168.0.110"
SOCKET_LIST = [] 
PORT = 9009
EXIT_COMMAND = "exit"
HOST="192.168.0.110"




import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES

class AESCipher(object):

    def __init__(self, key): 
        self.bs = 32
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, raw):
        raw = self._pad(raw)
        # iv is initialisation vector
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw))

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]







class multipi:
    def __init__(self):
        self.frame2=bytearray(b'')

 



    def chat_server_encrypted(self,e):  # e is a AES cipher, hasher is the multipi hash functions
        pygame.init()
        BLACK = (0,0,0)
        WIDTH = 100
        HEIGHT = 100
        windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

        windowSurface.fill(BLACK)
          # x = 0
    
        mymsg="" 
        main_loop=True
        me_print=False
        msg_send=False

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(10)
     
        # add server socket object to the list of readable connections
        SOCKET_LIST.append(server_socket)
 
        print("Chat server started on port " + str(PORT))
 
        while main_loop:

            # do work here
            ################################




            ###############################
            if me_print==False: 
                print('\n'+"[Me] "+mymsg,end="")
                me_print=True

            # get the list sockets which are ready to be read through select
            # 4th arg, time_out  = 0 : poll and never block
            ready_to_read,ready_to_write,in_error = select.select(SOCKET_LIST,[],[],0)


            for event in pygame.event.get():
                  if event.type == pygame.KEYDOWN: 
                        k=pygame.key.name(event.key)
                    
                        if k=="left ctrl":
                            main_loop=False
                            break
                    
                        elif k=="return":   #kd==pygame.K_RETURN:
                            #msg_send=True
                            mymsg+='\n'
                            #print("mymsg before hash=",mymsg)
                            mymsg=self.append_hash(mymsg)
                            #print("mymsg after hash=",mymsg)
                            enc=e.encrypt(mymsg)
                            #print('\n'+"server broadcasting msg=",mymsg," encrypted=",enc)
                        
                            self.broadcast(server_socket,"",e.encrypt(mymsg))
                            mymsg=""
                            me_print=False
                            break
                        elif k=="backspace" or k=="delete":
                            mymsg=mymsg[:-1]
                            k=""
                            me_print=False
                        elif k=="space":
                            k=" "

                        if me_print:    
                            mymsg=mymsg+k
                            print(k,end="")
                    
 


      
            for sock in ready_to_read:
                # a new connection request recieved
                if sock == server_socket: 
                    sockfd, addr = server_socket.accept()
                    SOCKET_LIST.append(sockfd)
                   # print("Client (%s, %s) connected" % addr)
                    msg="Client connected. [%s:%s] entered our chatting room\n" % addr
                    print(msg)
                    msg=self.append_hash(msg)   #.encode('utf-8'))
                    #msg2=e.encrypt(msg)
                    #print("msg=",msg)
                    me_print=False
                 
                    self.broadcast(server_socket, sockfd, e.encrypt(msg))
             
                # a message from a client, not a new connection
                else:
                    # process data recieved from client, 
                    try:
                        # receiving data from the socket.
                        data = sock.recv(RECV_BUFFER)

          #             data = sock.recv(RECV_BUFFER).decode('utf-8')

                        if data:
                             # there is something in the socket
                           # print("there is data")
                            inenc=base64.encodestring(data).rstrip()

                            #print("received encrypted base64=",inenc) 

                            try:
                                dec=e.decrypt(inenc)
                            except:   #TypeError
                                print("Decrypt failed.",inenc)
                                sys.exit()
                            dec,success,hash_bytes=self.unpack_hash(dec)
                      #      print("decrypted=",dec)

                            if success:
                           #     print("hash correct")
                                #print(str(sock.getpeername()),"data = ",dec)
                                msg="\r" + str(sock.getpeername()) + " : " +dec
                                #print(msg)
                                me_print=False

                            else:
                                me_print=False
                                msg="\r" + "hash incorrect hash=" + str(hash_bytes)+" : "+str(sock.getpeername()) + "] " + dec +"\n"
                            print(msg) #.rstrip())
                            msg2=self.append_hash(msg)
                            self.broadcast(server_socket, sock, e.encrypt(msg2))  
                        else:
                            print("removing Client (%s, %s) socket, connection broken\n" %addr)
                            me_print=False
                            # remove the socket that's broken    
                            if sock in SOCKET_LIST:
                                SOCKET_LIST.remove(sock)

                            # at this stage, no data means probably the connection has been broken
                            msg="Socket broken? Client (%s, %s) is offline\n" % addr
                            #print(msg)
                            msg2=self.append_hash(msg)
                           # print("msg=",msg)
                            self.broadcast(server_socket, sock, e.encrypt(msg2)) 

                    # exception 
                    except ConnectionError:   #ConnectionError   #BrokenPipeError
                   #     print("exception")
                        msg="Error exception: Client (%s, %s) is offline" % addr
                        #print(msg)
                        msg2=self.append_hash(msg)                
                        self.broadcast(server_socket, sock, e.encrypt(msg2))
                        continue

        pygame.quit()
        server_socket.close()
    
    # broadcast chat messages to all connected clients
    def broadcast (self,server_socket, sock, encrypted_msg):
        #print("broadcasting;", message)
        #print("encrypted=",encrypted_msg)
        print("\n")
        socket_count=0
        senddata = base64.decodestring(encrypted_msg)
        for socket in SOCKET_LIST:
            print("socket #",socket_count," : ",socket)
            socket_count+=1
            # send the message only to peer
            if socket != server_socket and socket != sock :
                try :
                    #socket.send(message.encode('utf-8'))
                    socket.send(senddata)
                    #socket.send(senddata.encode('utf-8'))
                except :
                    # broken socket connection
                    socket.close()
                    # broken socket, remove it
                    if socket in SOCKET_LIST:
                        SOCKET_LIST.remove(socket)
 


    def chat_client_encrypted(self,e):    #,hasher):   # e is the AES cipher
        if(len(sys.argv) < 4) :
            print("Usage : python3 encrypted_chat_client1.py hostname port passphrase")
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
            #  do work here
            ###################################




            ##########################################        
            socket_list = [sys.stdin, s]
         
            # Get the list sockets which are readable
            read_sockets, write_sockets, error_sockets = select.select(socket_list , [], [])
         
            for sock in read_sockets:            
                if sock == s:
                    # incoming message from remote server, s
                    #data = sock.recv(4096).decode('utf-8')
                    try:
                        # receiving data from the socket
                        data = sock.recv(4096)    #RECV_BUFFER)                   
                        if not data:    #==b'':
                            print("\nDisconnected from chat server")
                            sys.exit()
                        else :
#                            print(str(sock.getpeername())," : ",data," hash=",hash_msg(data))

                            #print("received bytes=",receivedata)
                           # data=data.encode('utf-8')
        

                            inenc=base64.encodestring(data).rstrip()
                           # print("received encrypted base64=",inenc) 
                            try:
                                indec=e.decrypt(inenc)
                            except:
                                print("Decrypt failed.",inenc)
                                sys.exit()
 
                           # print("decrypted=",indec)
                        
                    
                            dec2,success,hash_bytes2=self.unpack_hash(indec)
                           # print("decrypt=",dec2," hash bytes",hash_bytes2)

                            if not success:
                                sys.stdout.write("Hash incorrect: hash="+str(hash_bytes2)+" : "+str(sock.getpeername())+" : "+dec2+"\n")

                                  #  sys.stdout.write(str(sock.getpeername())+" : "+data)
                            sys.stdout.write(str(sock.getpeername())+" : "+dec2)
                            sys.stdout.write("[Me] "); sys.stdout.flush()     

                
                          # exception 
                    except ConnectionError:   #ConnectionError   #BrokenPipeError
                       # print("exception")
                        msg="Error exception: Client (%s, %s) is offline" % addr
                        print(msg)
                        msg2=self.append_hash(msg)
                        self.broadcast(server_socket, sock, e.encrypt(msg2))
                        continue


                else :
                    # user entered a message
                    msg = sys.stdin.readline()
                   # byte_msg=bytearray(msg.encode('utf-8'))
                   # print("msg byte array=",byte_msg)
                
                    msg2=self.append_hash(msg)    #.encode('utf-8'))
 #                   byte_msg.extend(hash_msg(msg))
  #                 print("msg : ",msg," byte_msg with hash=",byte_msg)

                    enc=e.encrypt(msg2)
                  # print("encrypted=",enc)
                
                    senddata = base64.decodestring(enc)
                             
                    #print("Encrypted bytes=",senddata)
                
                    #s.send(byte_msg)
                    try:
                        s.send(senddata)
                    except:
                        # broken socket connection
                        print("Broken socket connection.")
                        socket.close()
                        # broken socket, remove it
                        if socket in SOCKET_LIST:
                            SOCKET_LIST.remove(socket)

                
                    sys.stdout.write("[Me] "); sys.stdout.flush() 












    def append_hash(self,msg):  #,from_ip_bytes,to_ip_bytes):
        msg_byte=bytearray(msg.rstrip(),'utf-8')   #.encode('utf-8')).digest()
  #     hash_frame=(hashlib.md5(msg.encode('utf-8'))).hexdigest()  #.digest
        hash_frame=(hashlib.sha256(msg.encode('utf-8'))).hexdigest()  #.digest
        return(msg+hash_frame)

 
    def unpack_hash(self,byte_frame):
        length=len(byte_frame)
    
        if length>64:
            hash_bytes=byte_frame[-64:]
        
            msg=str(byte_frame[:length-64])   #,'utf-8')
            msg_bytes=byte_frame[:length-64]   #,'utf-8')

  #         check_hash=(hashlib.md5(msg.encode('utf-8'))).hexdigest()  #.digest

            check_hash=(hashlib.sha256(msg.encode('utf-8'))).hexdigest()  #.digest
    
            if hash_bytes!=check_hash:
                print("hash incorrect: actual hash=",hash_bytes," check=",check_hash)
                return(msg,False,hash_bytes)  #"hash incorrect")

            else:
                #print("hash correct  Frame_no:",frame_count)
                return(msg,True,hash_bytes)
        else:
            print("byte frame error.  len=",length,"<=64.  byte_frame=",byte_frame)
            return("",False,b'')




    # Python program to find SHA256 hash string of a file
    def hash_a_file(self):
 
        filename = input("Enter the input file name: ")
        sha256_hash = hashlib.sha256()
        with open(filename,"rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096),b""):
                sha256_hash.update(byte_block)
        print(sha256_hash.hexdigest())






    def send_msg(self,msg,from_ip,to_ip):    
        # 
         # convert to bytes
         # and pack into a frame
         #
        to_ip_bytes=socket.inet_aton(to_ip)
        from_ip_bytes=socket.inet_aton(from_ip)

        chunk_size=1000
           
        for chunk in iter(partial(msg, chunk_size), b""):
            chunk=bytearray(chunk)
            extra_fill=chunk_size-len(chunk)
            if extra_fill>0:
                chunk.extend(b' ' * extra_fill)   #b'\x00'
                             
            byte_frame=self.pack_frame(chunk,from_ip_bytes,to_ip_bytes)
            self.broadcast_frame(byte_frame,from_ip,to_ip)
        

    def pack_frame(self,byte_frame,from_ip_bytes,to_ip_bytes):
    #    byte_frame=bytearray(chunk)
        byte_frame.extend(from_ip_bytes)
        byte_frame.extend(to_ip_bytes)
        #print("byte frame=",byte_frame," len=",len(byte_frame))
        hash_frame=bytearray(hashlib.md5(byte_frame).digest())  #.digest

        #hash_frame=hash_msg(byte_frame)
        #print("hash frame=",hash_frame," len=",len(hash_frame))
        byte_frame.extend(hash_frame)
       # print("\nfinal frame=",byte_frame)
       # print("frame length=",len(byte_frame))
        return(byte_frame)



###########################################################




    def chat_server(self,host,port):
        pygame.init()
        BLACK = (0,0,0)
        WIDTH = 100
        HEIGHT = 100
        windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

        windowSurface.fill(BLACK)
   # x = 0
        
        msg="" 
        main_loop=True
        me_print=False
        msg_send=False

        #server_host_ip=socket.inet_aton(server_host)

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(10)
 
        # add server socket object to the list of readable connections
        SOCKET_LIST.append(server_socket)
 
        print("Chat server started on port " + str(port))
 
        while main_loop:

            if me_print==False: 
                print('\n'+"[Me] "+msg,end="")
                me_print=True


            # get the list sockets which are ready to be read through select
            # 4th arg, time_out  = 0 : poll and never block
            ready_to_read,ready_to_write,in_error = select.select(SOCKET_LIST,[],[],0)

            for event in pygame.event.get():
                  if event.type == pygame.KEYDOWN: 
                        k=pygame.key.name(event.key)
                        
                        if k=="left ctrl":
                            main_loop=False
                            break
                    
                        elif k=="return":   #kd==pygame.K_RETURN:
                            #msg_send=True
                            msg+='\n'
                            #print('\n'+"server broadcasting msg=",msg) 
                            self.broadcast(server_socket,"",msg)
                            msg=""
                            me_print=False
                            break
                        elif k=="backspace" or k=="delete":
                            msg=msg[:-1]
                            k=""
                            me_print=False
                        elif k=="space":
                            k=" "

                        if me_print:    
                            msg=msg+k
                            print(k,end="")
                    
                        
      
            for sock in ready_to_read:
                # a new connection request recieved
                if sock == server_socket: 
                    sockfd, addr = server_socket.accept()
                    SOCKET_LIST.append(sockfd)
                    print('\n'+"Client (%s, %s) connected" % addr)
                    me_print=False
                 
                    self.broadcast(server_socket, sockfd, "[%s:%s] entered our chatting room\n" % addr)
             
                # a message from a client, not a new connection
                else:
                
                    

                    
                    # process data recieved from client, 
                    try:
                        # receiving data from the socket.
                        raw_data = bytearray(sock.recv(RECV_BUFFER))
                    

                        #raw_data.split(b'\n')
                        #data_split=raw_data[0]
                        #hash_md5=raw_data[1]
                        #print("data_split:",data_split," hash",hash_md5)
                        data=raw_data.decode('utf-8')
                        if data:
                               # there is something in the socket
                           # print("there is data")
                            print('\n'+str(sock.getpeername()),":",data)
                            me_print=False
                            self.broadcast(server_socket, sock, "\r" + "[" + str(sock.getpeername()) + "] " + data)  
                        else:
                            print('\n'+"removing Client (%s, %s) socket, connection broken" %addr)
                            me_print=False
                            # remove the socket that's broken    
                            if sock in SOCKET_LIST:
                                SOCKET_LIST.remove(sock)

                            # at this stage, no data means probably the connection has been broken
                            self.broadcast(server_socket, sock, "Socket broken? Client (%s, %s) is offline\n" % addr) 

                    # exception 
                    except ConnectionError:   #ConnectionError   #BrokenPipeError
                       # print("exception")
                        self.broadcast(server_socket, sock, "Error exception: Client (%s, %s) is offline\n" % addr)
                        continue

    


        pygame.quit()
        server_socket.close()
    

    def frame_server(self,host,port):
    #    pygame.init()
    #    BLACK = (0,0,0)
    #    WIDTH = 100
    #    HEIGHT = 100
    #    windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

     #   windowSurface.fill(BLACK)
   # x = 0
        
        msg="" 
        main_loop=True
        me_print=False
        msg_send=False

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(10)
 
        # add server socket object to the list of readable connections
        SOCKET_LIST.append(server_socket)
 
        print("Frame server started on port " + str(port))
        print("listening for frames from frame clients")
 
        while main_loop:

        #    if me_print==False: 
        #        print('\n'+"[Me] "+msg,end="")
        #        me_print=True


            # get the list sockets which are ready to be read through select
            # 4th arg, time_out  = 0 : poll and never block
            ready_to_read,ready_to_write,in_error = select.select(SOCKET_LIST,[],[],0)
                        
      
            for sock in ready_to_read:
                # a new connection request recieved
                if sock == server_socket: 
                    sockfd, addr = server_socket.accept()
                    SOCKET_LIST.append(sockfd)
                    print('\n'+"Client (%s, %s) connected" % addr)
                    me_print=False
                 
                    self.broadcast_frame(server_socket, sockfd, "[%s:%s] entered our chatting room\n" % addr)
             
                # a message from a client, not a new connection
                else:
                
                    

                    
                    # process data recieved from client, 
                    try:
                        # receiving data from the socket.
                        raw_data = bytearray(sock.recv(RECV_BUFFER))
                        data=raw_data[:1000].decode('utf-8')
                       
                        if data:   #data?
                               # there is something in the socket
                            data=self.unpack_frame(raw_data)   
                            print("there is data")
                            print('\n'+str(sock.getpeername()),":",data)
                            me_print=False
                            self.send_frame(server_socket, sock, "\r" + "[" + str(sock.getpeername()) + "] " + data)  
                        else:
                            print('\n'+"removing Client (%s, %s) socket, connection broken" %addr)
                            me_print=False
                            # remove the socket that's broken    
                            if sock in SOCKET_LIST:
                                SOCKET_LIST.remove(sock)

                            # at this stage, no data means probably the connection has been broken
                            self.broadcast_frame(server_socket, sock, "Socket broken? Client (%s, %s) is offline\n" % addr) 

                    # exception 
                    except ConnectionError:   #ConnectionError   #BrokenPipeError
                       # print("exception")
                        self.broadcast_frame(server_socket, sock, "Error exception: Client (%s, %s) is offline\n" % addr)
                        continue

    


      #  pygame.quit()
        server_socket.close()
    



    
    # broadcast chat messages to all connected clients
    def broadcast (self,server_socket, sock, message):
        #print("broadcasting;", message)
        for socket in SOCKET_LIST:
            # send the message only to peer
            if socket != server_socket and socket != sock :
                try :
                     socket.send(message.encode('utf-8'))
                except :
                    # broken socket connection
                    socket.close()
                    # broken socket, remove it
                    if socket in SOCKET_LIST:
                        SOCKET_LIST.remove(socket)


    
    # broadcast frames messages to all connected clients
    def broadcast_frame(self,server_socket, sock, message):
        print("broadcasting frame;", message)
        for socket in SOCKET_LIST:
            # send the message only to peer
            if socket != server_socket and socket != sock :
                try :
                
                    
                    socket.send(self.pack_frame(message,HOST,sock),server_socket,socket)
                    
                except :
                    # broken socket connection
                    socket.close()
                    # broken socket, remove it
                    if socket in SOCKET_LIST:
                        SOCKET_LIST.remove(socket)



    # broadcast frames messages to all connected clients
    def send_frame(self,server_socket, sock, message):
        print("send frame: message=",message," server_socket=",HOST," sock=",sock)
        # send the message only to peer
        if socket != server_socket and socket != sock :
            try :
                
                socket.send(self.pack_frame(message,HOST,sock),HOST,sock)
               
                #socket.send(message.encode('utf-8'))
                    
            except :
                # broken socket connection
                socket.close()
                # broken socket, remove it
                if socket in SOCKET_LIST:
                    SOCKET_LIST.remove(socket)
  

    def chat_client(self,host,port):
       # if(len(sys.argv) < 3) :
       #     print("Usage : python3 Multi_pis_chat_client.py hostname port")
       #     sys.exit()

    #host = sys.argv[1]
    #port = int(sys.argv[2])
     
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)


       # host_ip=socket.inet_aton(host)
        # connect to remote host
        try :
            s.connect((host, port))
        except :
            print("Unable to connect")
            sys.exit()
     
        print("Connected to remote host. You can start sending messages")
        #sys.stdout.write("[Me] "); sys.stdout.flush()
     
        while True:
            socket_list = [sys.stdin, s]
         
            # Get the list sockets which are readable
            read_sockets, write_sockets, error_sockets = select.select(socket_list , [], [])
         
            for sock in read_sockets:            
                if sock == s:
                    # incoming message from remote server, s
                    data = sock.recv(1024).decode('utf-8')
                    if not data :
                        print("\nDisconnected from chat server")
                        sys.exit()
                    else :
                        print(str(sock.getpeername())," : ",data)
                    
                        sys.stdout.write(str(sock.getpeername())+" : "+data)
                        sys.stdout.write("[Me] "); sys.stdout.flush()     
            
                else :
                    # user entered a message
                    msg = sys.stdin.readline()
                    byte_msg=bytearray(msg.encode('utf-8'))
                  #  print("msg byte array=",byte_msg)
                  #  byte_msg.extend(hash_msg(msg))
                  #  print("msg : ",msg," byte_msg with hash=",byte_msg)

                
                    s.send(byte_msg)
                    sys.stdout.write("[Me] "); sys.stdout.flush() 


 

    def chat_client2(self,host,port):
       # if(len(sys.argv) < 3) :
       #     print("Usage : python3 Multi_pis_chat_client.py hostname port")
       #     sys.exit()

    #host = sys.argv[1]
    #port = int(sys.argv[2])
       # host_ip=socket.inet_aton(host)
     
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
        #print("[Me] ",'\t')
        
     
        while True:
            socket_list = [sys.stdin, s]
            #socket_list = [str(input()), s]
            #print("socket list;",socket_list)
             
            # Get the list sockets which are readable
            read_sockets, write_sockets, error_sockets = select.select(socket_list , [], [])
         
            for sock in read_sockets:            
                if sock == s:
                    # incoming message from remote server, s
                    data = sock.recv(1024).decode('utf-8')
                    if not data :
                        print("\nDisconnected from chat server")
                        sys.exit()
                    else :
                        #print(str(sock.getpeername())," : ",data)
                    
                        #sys.stdout.write(str(sock.getpeername())+" : "+data)
                        sys.stdout.write("[Me] "); sys.stdout.flush()
                        print(str(sock.getpeername()),":",data)
            
                else :
                    # user entered a message
                    msg = sys.stdin.readline()
                    byte_msg=bytearray(msg.encode('utf-8'))
                  #  print("msg byte array=",byte_msg)
                  #  byte_msg.extend(hash_msg(msg))
                  #  print("msg : ",msg," byte_msg with hash=",byte_msg)

                
                    s.send(byte_msg)
                    #sys.stdout.write("[Me] "); sys.stdout.flush() 
                    print("[Me] ",'\t')




    def frame_client(self,host):
       # if(len(sys.argv) < 3) :
       #     print("Usage : python3 Multi_pis_chat_client.py hostname port")
       #     sys.exit()

    #host = sys.argv[1]
    #port = int(sys.argv[2])

        msg_list=[]
     
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
     
        # connect to remote host
        try :
            s.connect((host, PORT))
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
                    #data = sock.recv(1024).decode('utf-8')

                    raw_data = sock.recv(1024)
                    data=bytearray(raw_data[:1000])
                    
                    if not data :
                        print("\nDisconnected from frame server")
                        sys.exit()
                    else :
                        print(str(sock.getpeername())," : ",data)

                        msg_list=self.unpack_frame(data) 
                    
                        sys.stdout.write(str(sock.getpeername())+" : "+data)
                        sys.stdout.write("[Me] "); sys.stdout.flush()     
            
                else :
                    # user entered a message
                    msg = sys.stdin.readline()
                    byte_msg=bytearray(msg.encode('utf-8'))
                  #  print("msg byte array=",byte_msg)
                  #  byte_msg.extend(hash_msg(msg))
                  #  print("msg : ",msg," byte_msg with hash=",byte_msg)

                
                    s.send(byte_msg)
                    sys.stdout.write("[Me] "); sys.stdout.flush() 








    def receive_msg(self):
        chunk_size=1000
        frame_size=1024
        frame_count=0
        msg_list=[]
        
        #with open(outfile, "ab") as f:
           
       # frame_count+=1
        #    print("len frame=",len(frame))
        msg_list=self.unpack_frame(frame)
        #    print("string frame=",string_frame) 
        #    g.write("%s\n" % string_frame[0])
                
        return(msg_list)



    def unpack_frame(self,frame):
          # split of the last 16 bytes which is the hash
        # hash the remaining 1008 bytes
        # compare with the original hash in the frame
        # if different error, try again or stop
        # split the first 1000 bytes as the data
        # convert to text
        # split the to_ip address and from _ip_address
        # convert to string

       # print("frame length passed=",len(frame))
        rest_of_frame=frame[:1008]
    
        msg_bytes=frame[:1000]

        ip_bytes=rest_of_frame[-8:]  
        to_ip_bytes=ip_bytes[-4:]
        from_ip_bytes=ip_bytes[:4]
    #    print("lengths msg=",len(msg_bytes)," ip=", len(ip_bytes))
        msg=str(msg_bytes.decode('utf-8'))
        from_ip=socket.inet_ntoa(from_ip_bytes)
        to_ip=socket.inet_ntoa(to_ip_bytes)
     #   print("msg=",msg,"\n\n from=",from_ip,"\n\n to=",to_ip,"\n\n")

        hash_bytes=bytes(frame[-16:])
     #   print(hash_bytes," len=",len(hash_bytes))
   
        check_hash=hashlib.md5(rest_of_frame).digest()  #.digest
     #   print(check_hash," len=",len(check_hash))

    
        if hash_bytes!=check_hash:
            print("hash incorrect  Frame_no:",frame_count," actual hash=",hash_bytes," check=",check_hash)
            return(["hash incorrect","192.168.0.from","192.168.0.to"])

        else:
            #print("hash correct  Frame_no:",frame_count)
            return([msg,from_ip,to_ip])
 

    
    






    def split_a_file_in_2(self,infile):

        #infile = open("input","r")

        with open(infile,'r') as f:
            linecount= sum(1 for row in f)

        splitpoint=linecount/2

        f.close()

        infilename=os.path.splitext(infile)[0]

        f = open(infile,"r")
        outfile1 = open(infilename+"001.csv","w")
        outfile2 = open(infilename+"002.csv","w")

        print("linecount=",linecount , "splitpoint=",splitpoint)

        linecount=0

        for line in f:
            linecount=linecount+1
            if ( linecount <= splitpoint ):
                outfile1.write(line)
            else:
                outfile2.write(line)

        f.close()
        outfile1.close()
        outfile2.close()


    def split_a_file_in_4(self,infile):

        #infile = open("input","r")

        with open(infile,'r') as f:
            linecount= sum(1 for row in f)

        splitpoint1=linecount/4
        splitpoint2=linecount/2
        splitpoint3=linecount*3/4

        f.close()

        infilename=os.path.splitext(infile)[0]

        f = open(infile,"r")
        outfile1 = open(infilename+"001.csv","w")
        outfile2 = open(infilename+"002.csv","w")
        outfile3 = open(infilename+"003.csv","w")
        outfile4 = open(infilename+"004.csv","w")


        print("linecount=",linecount , "splitpoints=",splitpoint1,splitpoint2,splitpoint3)

        linecount=0

        for line in f:
            linecount=linecount+1
            if ( linecount <= splitpoint1 ):
                outfile1.write(line)
            elif (linecount <=splitpoint2):
                outfile2.write(line)
            elif (linecount <=splitpoint3):
                outfile3.write(line)
            else:
                outfile4.write(line)

        f.close()
        outfile1.close()
        outfile2.close()
        outfile3.close()
        outfile4.close()


    
    def count_file_rows(self,filename):
        with open(filename,'r') as f:
            return sum(1 for row in f)

   

    def join2files(self,in1,in2,out):
        os.system("cat "+in1+" "+in2+" > "+out)

    def join4files(self,in1,in2,in3,in4,out):
        os.system("cat "+in1+" "+in2+" "+in3+" "+in4+" > "+out)




    def calculate(self,filename,formula):
        # takes a csv file of numbers
        # and a formula string using the code f[0], f[1], f[2] .... for the field names
        # an calculates for each row.  Appending the formula and the answer on the end of each row delimited by ",' 

       # safe_dict = dict((k, getattr(math, k)) for k in safe_list)

        #add = lambda x, y: x + y
        #subtract= lambda x,y: x-y
        #multiply =lambda x,y: x*y
        #divide=lambda x,y :x/y

    
        try:
            #out=open("new"+filename,'w') 
            with open(filename, 'r') as csvfile:

                first_line = csvfile.readline()
                #your_data = csvfile.readlines()

                ncol = first_line.count(',') + 1 

                print("file:",filename," has ",ncol," columns")
                print("row count=",count_file_rows(filename))
                print("formula=",formula)
                reader=csv.reader(csvfile,delimiter=',')
                csvfile.seek(0)
            
                f=[0.0]*(ncol+1)
                rowcount=0
                totalvalue=0.0
            
                for row in reader:
                    for field_count in range(0,ncol,1):
                        #print(field_count,row[field_count])
                        try:
                            f[field_count]=float(row[field_count])
                        except TypeError:
                            f[field_count]=0.0
                        #print("first line : f[",field_count,"]=",f[field_count])
                   
                    #print(row[0],"=",eval(formula)) #,{},{}))   #{'__builtins__':None},{}))
                    f[ncol]=eval(formula)
                    totalvalue+=f[ncol]
                    rowcount+=1
                    #print("row:",row[0],"f=",f)
                   # out.write(str(f)+"\n")
                     

        except Exception as e:
            print("error=",e)
            #print(sys.exc_type)

        csvfile.close()
       # out.close()
        if rowcount!=0:
            return(totalvalue/rowcount)   # this is the avarage fitness of the data sample
        else:
           print("file empty")
           return(0)


    def create_frame_file(self,filenamein,filenameout,from_ip,to_ip):    
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
        g=open(filenameout,"wb")
        with open(filenamein, "rb") as f:
            for chunk in iter(partial(f.read, chunk_size), b""):
                chunk=bytearray(chunk)
                extra_fill=chunk_size-len(chunk)
                if extra_fill>0:
                    chunk.extend(b' ' * extra_fill)   #b'\x00'
                             
                byte_frame=self.pack_frame(chunk,from_ip_bytes,to_ip_bytes)
            #write_frame("myfile.bin",byte_frame)
        
                g.write(byte_frame)
        f.close()
        g.close()
        

    def read_frame_file(self,infile,outfile):
        chunk_size=1000
        frame_size=1024
        frame_count=0

        g=open(outfile,"w")
        with open(infile, "rb") as f:
            for chunk in iter(partial(f.read, frame_size), b""):
                frame=bytearray(chunk)
                frame_count+=1
        #    print("len frame=",len(frame))
                string_frame=self.read_frame(frame,frame_count)
        #    print("string frame=",string_frame) 
        #    g.write("%s\n" % string_frame[0])
                g.write(string_frame[0])

        g.close()
        f.close()        
        # open binary file
        #
        # read 1024 chunk which is a frame
        # print



    def file_receive(self,s):
        clock_start=time.clock()
        #time_start=time.time()
        filename="receive_file.csv"

        with open(filename, 'wb') as f:
            print(filename," opened")
            while True:
            #print('receiving data...')
                data = s.recv(BUFFER_SIZE)
            #print("data=%s", (data))
                if not data:
                    f.close()
                    print(filename," file close()")
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



    def file_send(self,s):
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


    def valid_ip(self,ip_addr):
        try:
            socket.inet_aton(ip_addr)
            return True
        except socket.error:
            return False


"""
def split_file(file_path, chunk=4000):

    p = subprocess.Popen(['split', '-a', '2', '-l', str(chunk), file_path,
                          os.path.dirname(file_path) + '/'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()
    # Remove the original file if required
    #try:
    #    os.remove(file_path)
    #except OSError:
    #    pass
    #return True
"""
"""
#splitting files
import tempfile
from itertools import groupby, count

temp_dir = tempfile.mkdtemp()

def tempfile_split(filename, temp_dir, chunk=4000000):
    with open(filename, 'r') as datafile:
        groups = groupby(datafile, key=lambda k, line=count(): next(line) // chunk)
        for k, group in groups:
            output_name = os.path.normpath(os.path.join(temp_dir + os.sep, "tempfile_%s.tmp" % k))
            for line in group:
                with open(output_name, 'a') as outfile:
                outfile.write(''.join(group))

"""
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





        
