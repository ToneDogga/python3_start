from __future__ import print_function
from __future__ import division

from multiprocessing import Pool, TimeoutError, Process, freeze_support, Queue, Lock, Manager

#from  multiprocessing import Pool, Queue
from os import getpid
from time import sleep
from random import random
import itertools
import platform
import math
import sys
import multiprocessing
import time
import os
import hashlib
from pathlib import Path

EOL='\n'
FILENAME="testfile1.csv"


def generate_payoff_environment_1d_mp(astart_val,asize_of_env):   
    rowno=0
    print("payoff calc function started:",os.getpid())

    clock_start=time.process_time()

    f=open(FILENAME,"a")
    for a in range(astart_val,astart_val+asize_of_env):
        payoff=-10*math.sin(a/44)*12*math.cos(a/33)*1000/(a+1)    
        f.write(str(rowno)+","+str(a)+","+str(payoff)+EOL)
        rowno+=1
    f.close()

    clock_end=time.process_time()
    duration_clock=clock_end-clock_start

    print("payoff pid:[",os.getpid(),"] finished chunk",rowno,"rows in:",duration_clock,"secs.")
    return   #(payoff)    #(os.getpid(),payoff)


        
# Python program to find SHA256 hash string of a file
def hash_a_file(FILENAME):
    #filename = input("Enter the input file name: ")
    sha256_hash = hashlib.sha256()
    with open(FILENAME,"rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    return(sha256_hash.hexdigest())



def count_file_rows(FILENAME):
    with open(FILENAME,'r') as f:
        return sum(1 for row in f)



# Warning from Python documentation:
# Functionality within this package requires that the __main__ module be
# importable by the children. This means that some examples, such as the
# multiprocessing.Pool examples will not work in the interactive interpreter.
#if __name__ == '__main__':

def main():
    if(len(sys.argv) < 2) :
        print("Usage : python3 mp_simple1.py rownumbers")
        sys.exit()

    row_numbers = int(sys.argv[1])

    freeze_support()

    windows=platform.system().lower()[:7]
    print("platform=",windows)
    if windows=="windows":
        EOL="\r\n"
    else:
        EOL='\n'

    cpus = multiprocessing.cpu_count()
 #   cpus = cpu_count()
    print("cpus=",cpus)

 #   my_file = Path("/home/pi/Python_Lego_projects/testfile1.csv")
    my_file = Path("/home/pi/Python_Lego_projects/testfile1.csv")

    #my_file=FILENAME  #basename(my_file)
    if my_file.is_file():
  #      print(FILENAME," #=",hash_a_file(FILENAME))
        os.remove(FILENAME)
    f=open(FILENAME,"w")
    f.close()
    #    print(FILENAME,":removed")    
  
    #clock_start=time.process_time()


    chunk=int(row_numbers/cpus)
    remainder=row_numbers%cpus
    print("rownumbers=",row_numbers," chunk size=",chunk," remainder=",remainder)
    with Pool(processes=cpus) as pool:  # processes=cpus
        if remainder!=0:
            multiple_results = [pool.apply_async(generate_payoff_environment_1d_mp,(0,remainder,))]
        else:
            multiple_results=[]
        for i in range(0,cpus):
            multiple_results.append(pool.apply_async(generate_payoff_environment_1d_mp,(i*chunk+remainder,chunk,)))

       # print("mr=",len(multiple_results))
        for res in multiple_results:
            result=res.get(timeout=None)       
          #  res.wait()

    # Waits a bit for the child processes to do some work
    # because when the parent exits, childs are terminated.
 
    print("Generate payoff results finished")
    pool.close()
    pool.join()




   # clock_end=time.process_time()
   # duration_clock=clock_end-clock_start
   # print("gen payoff rows=",row_numbers," Time:", duration_clock," secs.")



  #  sleep(5)
    print("count file rows=",count_file_rows(FILENAME))
    print(FILENAME," #=",hash_a_file(FILENAME))      

    if my_file.is_file():
  #      print(FILENAME," #=",hash_a_file(FILENAME))
        os.remove(FILENAME)
    f=open(FILENAME,"w")
    f.close()


    clock_start=time.process_time()

    generate_payoff_environment_1d_mp(0,row_numbers)

    clock_end=time.process_time()
    duration_clock=clock_end-clock_start
    print("gen payoff 1 processor rows=",row_numbers," Time:", duration_clock," secs.")

    print("count file rows=",count_file_rows(FILENAME))
    print(FILENAME," #=",hash_a_file(FILENAME))      


main()
