# chat_client.py

import sys, socket, select
import hashlib
import base64
import multipiv2




def chat_client_encrypted(e):   # e is the AES cipher
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
                data = sock.recv(4096)
              #  if not data:
              #      try:
              #          ddata=data.decode('utf-8')
              #      except:
              #          ddata=False
                    
                if data==b'':
                    print("\nDisconnected from chat server")
                    sys.exit()
                else :
#                    print(str(sock.getpeername())," : ",data," hash=",hash_msg(data))

                    #print("received bytes=",receivedata)
                   # data=data.encode('utf-8')
                    inenc=base64.encodestring(data).rstrip()


                   # print("received encrypted base64=",inenc) 
                    try:
                        dec=e.decrypt(inenc)
                    except:
                        print("Decrypt failed.",inenc)
                        sys.exit()
 
                    #print("decrypted=",dec)
                    
                    dec,success=hasher.unpack_hash(dec)

                    if success:
                  #  sys.stdout.write(str(sock.getpeername())+" : "+data)
                        sys.stdout.write(str(sock.getpeername())+" : "+dec)
                    else:
                        sys.stdout.write("Hash incorrect:"+str(sock.getpeername())+" : "+dec+"\n")
                    sys.stdout.write("[Me] "); sys.stdout.flush()     
            
            else :
                # user entered a message
                msg = sys.stdin.readline()
               # byte_msg=bytearray(msg.encode('utf-8'))
               # print("msg byte array=",byte_msg)
                
                msg=hasher.append_hash(msg)    #.encode('utf-8'))
 #               byte_msg.extend(hash_msg(msg))
  #              print("msg : ",msg," byte_msg with hash=",byte_msg)

                enc=e.encrypt(msg)
              # print("encrypted=",enc)

                senddata = base64.decodestring(enc)

                #print("Encrypted bytes=",senddata)
                
                #s.send(byte_msg)

                s.send(senddata)
                
                sys.stdout.write("[Me] "); sys.stdout.flush() 


print("Encrypted chat client v1")
#pp=input("Passphase?")
pp = sys.argv[3]   # passphrase
key=str(hashlib.md5(pp.encode('utf-8')).digest())
#print("sumkey=",key)

e=multipiv2.AESCipher(key)
hasher=multipiv2.multipi()
chat_client_encrypted(e)
#message=input("message to encrypt:")
#enc=e.encrypt(message)
#print("encrypted=",enc)

#senddata = base64.decodestring(enc)

#print("Encrypted bytes=",senddata)


#receivedata=senddata

#print("received bytes=",receivedata)

#inenc=base64.encodestring(receivedata).rstrip()


#print("received base64=",inenc) 

#dec=e.decrypt(inenc)
#print("decrypted=",dec)



