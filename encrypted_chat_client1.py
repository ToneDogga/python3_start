# chat_client.py

import sys, socket, select
import hashlib
import base64
import multipiv2

RECV_BUFFER = 4096 


def chat_client_encrypted(e,hasher):   # e is the AES cipher
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
        
        socket_list = [sys.stdin, s]
         
        # Get the list sockets which are readable
        read_sockets, write_sockets, error_sockets = select.select(socket_list , [], [])
         
        for sock in read_sockets:            
            if sock == s:
                # incoming message from remote server, s
                #data = sock.recv(4096).decode('utf-8')
                try:
                    # receiving data from the socket
                    data = sock.recv(RECV_BUFFER)                   
                    if not data:    #==b'':
                        print("\nDisconnected from chat server")
                        sys.exit()
                    else :
#                        print(str(sock.getpeername())," : ",data," hash=",hash_msg(data))

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
                        
                    
                        dec2,success,hash_bytes2=hasher.unpack_hash(indec)
                       # print("decrypt=",dec2," hash bytes",hash_bytes2)

                       # if success:
                              #  sys.stdout.write(str(sock.getpeername())+" : "+data)
                        sys.stdout.write(str(sock.getpeername())+" : "+dec2)
                      #  else:
                      #      sys.stdout.write("Hash incorrect: hash="+str(hash_bytes2)+" : "+str(sock.getpeername())+" : "+dec2+"\n")
                        sys.stdout.write("[Me] "); sys.stdout.flush()     

                
                      # exception 
                except ConnectionError:   #ConnectionError   #BrokenPipeError
                   # print("exception")
                    msg="Error exception: Client (%s, %s) is offline" % addr
                    print(msg)
                    broadcast(server_socket, sock, e.encrypt(msg))
                    continue


            else :
                # user entered a message
                msg = sys.stdin.readline()
               # byte_msg=bytearray(msg.encode('utf-8'))
               # print("msg byte array=",byte_msg)
                
                msg2=hasher.append_hash(msg)    #.encode('utf-8'))
 #               byte_msg.extend(hash_msg(msg))
  #              print("msg : ",msg," byte_msg with hash=",byte_msg)

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


print("Encrypted chat client v1")
#pp=input("Passphase?")
pp = sys.argv[3]   # passphrase
key=str(hashlib.md5(pp.encode('utf-8')).digest())
#print("sumkey=",key)

e=multipiv2.AESCipher(key)
hasher=multipiv2.multipi()
chat_client_encrypted(e,hasher)



