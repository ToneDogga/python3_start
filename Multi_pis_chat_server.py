# chat_server.py
 
import sys, socket, select

#HOST = '' 
HOST="192.168.0.110"
SOCKET_LIST = []
RECV_BUFFER = 4096 
PORT = 9009


def file_receive(s):
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




def chat_server():

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(10)
 
    # add server socket object to the list of readable connections
    SOCKET_LIST.append(server_socket)
 
    print("Chat server started on port " + str(PORT))
 
    while 1:

        # get the list sockets which are ready to be read through select
        # 4th arg, time_out  = 0 : poll and never block
        ready_to_read,ready_to_write,in_error = select.select(SOCKET_LIST,[],[],0)
      
        for sock in ready_to_read:
            # a new connection request recieved
            if sock == server_socket: 
                sockfd, addr = server_socket.accept()
                SOCKET_LIST.append(sockfd)
                print("Client (%s, %s) connected" % addr)
                 
                broadcast(server_socket, sockfd, "[%s:%s] entered our chatting room\n" % addr)
             
            # a message from a client, not a new connection
            else:
                # process data recieved from client, 
                try:
                    # receiving data from the socket.
                    data = sock.recv(RECV_BUFFER).decode('utf-8')
                   
                    if data:
                        # there is something in the socket
                       # print("there is data")
                        print(str(sock.getpeername()),"data = ",data)
                        broadcast(server_socket, sock, "\r" + "[" + str(sock.getpeername()) + "] " + data)  
                    else:
                        print("removing Client (%s, %s) socket, connection broken" %addr)
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

    server_socket.close()
    
# broadcast chat messages to all connected clients
def broadcast (server_socket, sock, message):
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
 
if __name__ == "__main__":

    sys.exit(chat_server())


         
