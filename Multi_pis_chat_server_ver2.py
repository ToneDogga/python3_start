# chat_server.py
 
import sys
import socket
import select
import threading
import queue
import time
import hashlib
import pygame
from pygame.locals import *



#HOST = '' 
HOST="192.168.0.110"
SOCKET_LIST = []
RECV_BUFFER = 4096 
PORT = 9009

EXIT_COMMAND = "exit"




def hexhash_msg(msg):  #128 bit hash
    return (hashlib.md5(str(msg).encode("utf-8")).hexdigest()) #For sha  hash use hashlib.sha256


def hash_msg(msg):   # 128 bit hash
    return (hashlib.md5(str(msg).encode('utf-8')).digest())


def chat_server():
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

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(10)
 
    # add server socket object to the list of readable connections
    SOCKET_LIST.append(server_socket)
 
    print("Chat server started on port " + str(PORT))
 
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
                        broadcast(server_socket,"",msg)
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
                 
                broadcast(server_socket, sockfd, "[%s:%s] entered our chatting room\n" % addr)
             
            # a message from a client, not a new connection
            else:
                
                    

                    
                # process data recieved from client, 
                try:
                    # receiving data from the socket.
                    raw_data = bytearray(sock.recv(RECV_BUFFER))
                    

                    raw_data.split(b'\n')
                    data_split=raw_data[0]
                    hash_md5=raw_data[1]
                    print("data_split:",data_split," hash",hash_md5)
                    data=raw_data.decode('utf-8')
                    if data:
                        # there is something in the socket
                       # print("there is data")
                        print('\n'+str(sock.getpeername()),":",data," hash:",hash_msg(data)," hexhash:",hexhash_msg(data))
                        me_print=False
                        broadcast(server_socket, sock, "\r" + "[" + str(sock.getpeername()) + "] " + data)  
                    else:
                        print('\n'+"removing Client (%s, %s) socket, connection broken" %addr)
                        me_print=False
                        # remove the socket that's broken    
                        if sock in SOCKET_LIST:
                            SOCKET_LIST.remove(sock)

                        # at this stage, no data means probably the connection has been broken
                        broadcast(server_socket, sock, "Socket broken? Client (%s, %s) is offline\n" % addr) 

                # exception 
                except ConnectionError:   #ConnectionError   #BrokenPipeError
                   # print("exception")
                    broadcast(server_socket, sock, "Error exception: Client (%s, %s) is offline\n" % addr)
                    continue

    


    pygame.quit()
    server_socket.close()
    
# broadcast chat messages to all connected clients
def broadcast (server_socket, sock, message):
    print("broadcasting;", message," hash=",hash_msg(message)," hexhash=",hexhash_msg(message))
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






    


 
#if __name__ == "__main__":
sys.exit(chat_server())
   



