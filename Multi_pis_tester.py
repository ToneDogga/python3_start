from __future__ import print_function # use python 3 syntax but make it compatible with python 2
from __future__ import division       #                           ''

import multipiv2
import time

to_ip="192.168.0.110"
from_ip="192.168.0.105"


# first testing
# mp.create_frame_file calls mp.create_frame reads a big csv file and creates socketdata.bin
#  mp.read_frame_file calls mp.read_frame reads socketdata.bin creates an outfile.csv
# which matches the 
#
# next level testing
# a text msg is given to mp.send_msg which calls mp.pack_frame and sends it to the ip_to address
# creates a 1024 byte frame and then sends it to to_ip address using mp.send_frame
#
# mp.receive_msg calls mp.unpack_frame waits for socket communications from_ip address
# recives 1024 byte frame using mp.receive_frame
# check hash and extract data as a text list


HOST = '' 
HOST="192.168.0.106"
SOCKET_LIST = []
RECV_BUFFER = 1024  #4096 
PORT = 9009



mp = multipiv2.multipi() # Create an instance of the multi class. mp will be the multipi object.

def main():


    server=input("Is this a multipi server? (y/n)")
    if server=="y":
        print("Multipi Server IP=",HOST," [",PORT,"]")         
        mp.chat_server2(HOST)
    else:
        valid=False
        while not valid:
            host_ip=input("This is a Multipi client. Enter Multipi Server IP address:")
            
            valid=mp.valid_ip(host_ip)
            #socket.inet_aton(host_ip)
            if valid:
                print("Valid address")
            else:
                print("Invalid address")
              

        print("Multi pi host ip address=",host_ip)
        mp.chat_client(host_ip)


    """
    #read_bin()
    print("in rows=",mp.count_file_rows("tuninglog1.csv"))
    #log.debug("debug!")

    print("read a csv file and create a binaryfile")
    clock_start=time.clock()

    mp.create_frame_file("tuninglog1.csv","socketdata.bin",from_ip,to_ip)
    
    clock_end=time.clock()

    duration_clock=clock_end-clock_start

    print("Clock: start=",clock_start," end=",clock_end)
    print("Clock: duration_clock =", duration_clock)
    print("\n")


    

    print("read socketdata.bin")


    clock_start=time.clock()

    mp.read_frame_file("socketdata.bin","testout.csv")
    
    clock_end=time.clock()

    duration_clock=clock_end-clock_start

    print("Clock: start=",clock_start," end=",clock_end)
    print("Clock: duration_clock =", duration_clock)
    print("\n")

    print("output back to out.csv")

    print("testout rows=",mp.count_file_rows("testout.csv"))


    print("split a big csv file in 4")


    clock_start=time.clock()

  #  split_a_file_in_2("tuninglog1.csv")
    mp.split_a_file_in_4("tuninglog1.csv")
  
    
    clock_end=time.clock()

    duration_clock=clock_end-clock_start

    print("Clock: start=",clock_start," end=",clock_end)
    print("Clock: duration_clock =", duration_clock)
    print("\n")

    print("output back to ...00n.csv")



    print("join4 csv files")


    clock_start=time.clock()

  #  join2files("tuninglog1001.csv","tuninglog1002.csv","testout.csv")
    mp.join4files("tuninglog1001.csv","tuninglog1002.csv","tuninglog1003.csv","tuninglog1004.csv","testout2.csv")
    
    clock_end=time.clock()

    duration_clock=clock_end-clock_start

    print("Clock: start=",clock_start," end=",clock_end)
    print("Clock: duration_clock =", duration_clock)
    print("\n")



    outfile=open("formula_fitness.csv",'w') 
    infile=open("formulas.txt", 'r')
    linecount=0
    
    #formula="f[0]+f[4]*f[3]+f[2]**2"

    for formula in infile:
        linecount=linecount+1
        print("formula[",linecount,"]=",formula) 


        clock_start=time.clock()

        fitness=mp.calculate("shop sales test 2019.csv",formula)  # creates a new csv file call with "new" added to the front of the name
# tuninglog1.csv

        clock_end=time.clock()

        duration_clock=clock_end-clock_start


        outfile.write(str(linecount)+" , "+formula.rstrip()+" ,"+str(fitness)+" , "+str(duration_clock)+"\n")
   # print("newfile row count=",count_file_rows("newtuninglog1.csv"))
        print("fitness=",fitness) 

 #   print("Clock: start=",clock_start," end=",clock_end)
        print("Clock: duration_clock =", duration_clock)
        print("\n")

    infile.close()
    outfile.close()
"""
    

#if __name__ == "__main__":

#    sys.exit(chat_server())


main()
