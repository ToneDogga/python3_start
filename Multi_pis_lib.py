from __future__ import division

import numpy as np
from functools import partial
import socket
import hashlib
import time
import csv
import sys
import math
import os
#import itertools

#import sys

#import logging
#import os
 
#logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
#log = logging.getLogger(__name__)






def process_data(data):
    print(data)
    print(data.decode('utf-8'))
    print(repr(data))


def read_config():
    global target
    with open('/home/pi/Python_Lego_projects/myfile.csv','r') as f:
        config=f.readline().split(',')
        print(config)
        setpoint=float(config[0])

def count_file_rows(filename):
    with open(filename,'r') as f:
        return sum(1 for row in f)

def read_bin():
    chunk_size=1024
    with open("myfile.csv", "rb") as f:
        for chunk in iter(partial(f.read, chunk_size), b""):
            #process_data(chunk)
            write_bin(chunk)
    f.close()        
    

#def write_frame(filename,frame):
#    with open(filename, "ab") as f:
#        f.write(frame)
#    f.close()    



    

#    from_ip_bytes=socket.inet_aton(from_ip)
    
#    print(ip_bytes,"size=",len(ip_bytes))
#    ip_str=socket.inet_ntoa(ip_bytes)
#    log.debug(ip_str)
#    print("ip str=",ip_str)


def join2files(in1,in2,out):
    os.system("cat "+in1+" "+in2+" > "+out)

def join4files(in1,in2,in3,in4,out):
    os.system("cat "+in1+" "+in2+" "+in3+" "+in4+" > "+out)


# joining csv files
#  cat file001.csv file002.csv > merged.csv

"""
        byte = f.read(1)
        while byte != b"":
            # Do stuff with byte.
            print(byte)
            byte = f.read(1)
"""

#def hash_msg(msg):
#    return (hashlib.md5(str(msg).encode('utf-8')).digest()) #.digest()

def create_frame(byte_frame,from_ip_bytes,to_ip_bytes):
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



 

def create_frame_file(filenamein,filenameout,from_ip,to_ip):    
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
                             
            byte_frame=create_frame(chunk,from_ip_bytes,to_ip_bytes)
            #write_frame("myfile.bin",byte_frame)
        
            g.write(byte_frame)
    f.close()
    g.close()



def read_frame(frame,frame_count):
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
 


def read_frame_file(infile,outfile):
    chunk_size=1000
    frame_size=1024
    frame_count=0

    g=open(outfile,"w")
    with open(infile, "rb") as f:
        for chunk in iter(partial(f.read, frame_size), b""):
            frame=bytearray(chunk)
            frame_count+=1
        #    print("len frame=",len(frame))
            string_frame=read_frame(frame,frame_count)
        #    print("string frame=",string_frame) 
            #with open("out.csv","a") as g:
                #cr = csv.writer(g,delimiter=",",lineterminator="\n")
                #cr.writerow(string_frame)
                #g.write(string_frame)
            #for item in string_frame:
         #       print("item=",item)
                #g.write("%s\n" % item)
        #    g.write("%s\n" % string_frame[0])
            g.write(string_frame[0])

    g.close()
    f.close()        
    # open binary file
    #
    # read 1024 chunk which is a frame
    # print
    


def split_a_file_in_2(infile):

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


def split_a_file_in_4(infile):

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
    




def calculate(filename,formula):
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

def frame_test():
    """
        #read_config()
    #read_bin()
    #print("in rows=",count_file_rows("myfile.csv"))
    #log.debug("debug!")
    to_ip="192.168.0.110"
    from_ip="192.168.0.105"

   # print("Read myfile.csv")
    clock_start=time.clock()

    create_frame_file("tuninglog1.csv","socketdata.bin",from_ip,to_ip)
    
    clock_end=time.clock()

    duration_clock=clock_end-clock_start

    print("Clock: start=",clock_start," end=",clock_end)
    print("Clock: duration_clock =", duration_clock)
    print("\n")


    

    print("read socketdata.bin")


    clock_start=time.clock()

    read_frame_file("socketdata.bin","testout.csv")
    
    clock_end=time.clock()

    duration_clock=clock_end-clock_start

    print("Clock: start=",clock_start," end=",clock_end)
    print("Clock: duration_clock =", duration_clock)
    print("\n")

    print("output back to out.csv")

    print("testout rows=",count_file_rows("testout.csv"))


    print("split a big csv file in 4")


    clock_start=time.clock()

  #  split_a_file_in_2("tuninglog1.csv")
    split_a_file_in_4("tuninglog1.csv")
  
    
    clock_end=time.clock()

    duration_clock=clock_end-clock_start

    print("Clock: start=",clock_start," end=",clock_end)
    print("Clock: duration_clock =", duration_clock)
    print("\n")

    print("output back to ...00n.csv")



    print("join4 csv files")


    clock_start=time.clock()

  #  join2files("tuninglog1001.csv","tuninglog1002.csv","testout.csv")
    join4files("tuninglog1001.csv","tuninglog1002.csv","tuninglog1003.csv","tuninglog1004.csv","testout2.csv")
    
    clock_end=time.clock()

    duration_clock=clock_end-clock_start

    print("Clock: start=",clock_start," end=",clock_end)
    print("Clock: duration_clock =", duration_clock)
    print("\n")
    """

    outfile=open("formula_fitness.csv",'w') 
    infile=open("formulas.txt", 'r')
    linecount=0
    
    #formula="f[0]+f[4]*f[3]+f[2]**2"

    for formula in infile:
        linecount=linecount+1
        print("formula[",linecount,"]=",formula) 


        clock_start=time.clock()

        fitness=calculate("shop sales test 2019.csv",formula)  # creates a new csv file call with "new" added to the front of the name
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

    

#create_config()
frame_test()



        
