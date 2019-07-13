import base64
import hashlib
import multipiv2

# chat_server.py
 
import sys, socket, select
import pygame
from pygame.locals import *

#HOST = '' 
HOST="192.168.0.110"
SOCKET_LIST = []
RECV_BUFFER = 4096 
PORT = 9009


def chat_server_encrypted(e,hasher):  # e is a AES cipher, hasher is the multipi hash functions
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
                        mymsg=hasher.append_hash(mymsg)
                        #print("mymsg after hash=",mymsg)
                        enc=e.encrypt(mymsg)
                        #print('\n'+"server broadcasting msg=",mymsg," encrypted=",enc)
                        
                        broadcast(server_socket,"",e.encrypt(mymsg))
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
                msg="Client connected. [%s:%s] entered our chatting room" % addr
                print(msg)
                msg=hasher.append_hash(msg)   #.encode('utf-8'))
                #print("msg=",msg)
                me_print=False
                 
                broadcast(server_socket, sockfd, e.encrypt(msg))
             
            # a message from a client, not a new connection
            else:
                # process data recieved from client, 
                try:
                    # receiving data from the socket.
                    data = sock.recv(RECV_BUFFER)

          #          data = sock.recv(RECV_BUFFER).decode('utf-8')

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
                        dec,success=hasher.unpack_hash(dec)
                      #  print("decrypted=",dec)

                        if success:
                           # print("hash correct")
                            #print(str(sock.getpeername()),"data = ",dec)
                            msg="\r" + str(sock.getpeername()) + dec
                            #print(msg)
                            me_print=False

                        else:
                            me_print=False
                            msg="\r" + "hash incorrect [" + str(sock.getpeername()) + "] " + dec
                        print(msg.rstrip())
                        broadcast(server_socket, sock, e.encrypt(msg))  
                    else:
                        print("removing Client (%s, %s) socket, connection broken" %addr)
                        me_print=False
                        # remove the socket that's broken    
                        if sock in SOCKET_LIST:
                            SOCKET_LIST.remove(sock)

                        # at this stage, no data means probably the connection has been broken
                        msg="Socket broken? Client (%s, %s) is offline" % addr
                        print(msg)
                        msg=hasher.append_hash(msg)
                       # print("msg=",msg)
                        broadcast(server_socket, sock, e.encrypt(msg)) 

                # exception 
                except ConnectionError:   #ConnectionError   #BrokenPipeError
                   # print("exception")
                    msg="Error exception: Client (%s, %s) is offline" % addr
                    print(msg)
                    msg=hasher.append_hash(msg)                
                    broadcast(server_socket, sock, e.encrypt(msg))
                    continue

    pygame.quit()
    server_socket.close()
    
# broadcast chat messages to all connected clients
def broadcast (server_socket, sock, encrypted_msg):
    #print("broadcasting;", message)
    #print("encrypted=",encrypted_msg)
    senddata = base64.decodestring(encrypted_msg)
    for socket in SOCKET_LIST:
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
 


#    sys.exit(chat_server())

print("Encrypted chat server v1")
pp=input("Passphrase?")
key=str(hashlib.md5(pp.encode('utf-8')).digest())
#print("sumkey=",key)

e=multipiv2.AESCipher(key)
hasher=multipiv2.multipi()
chat_server_encrypted(e,hasher)
