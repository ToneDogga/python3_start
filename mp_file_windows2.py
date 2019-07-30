from __future__ import print_function
from __future__ import division

#from multiprocessing.dummy import Pool, TimeoutError, Process, Lock, freeze_support, Queue, Lock, Manager
#from multiprocessing import TimeoutError, Pool, Process, Lock, freeze_support, Queue, Lock, Manager
from multiprocessing.dummy import Pool, Process, Lock, freeze_support, Queue, Lock, Manager

#from  multiprocessing import Pool, Queue
from os import getpid
from time import sleep
from random import random
import itertools
import platform
import math
import sys
import time
import os
import hashlib
from pathlib import Path

EOL='\n'
FILENAME="testfile1.csv"
import multiprocessing 
import time

fn = 'c:/temp/temp.txt'


def listener(q):    #,l_lock):
    '''listens for messages on the q, writes to file. '''

   # l_lock.acquire()
    f=open(FILENAME,"a")   #,buffering=0)
    #f.flush()
    while True:
        if q.full():
            print("q full")
        m = q.get()    #timeout=20)
        if m == 'kill':
            #f.write('killed')
            break
 
        f.write(m)
        f.flush()
       
    f.close()
   # l_lock.release()




def generate_payoff_environment_1d_mp(astart_val,asize_of_env,q):    #l_lock  
    rowno=0
    print("payoff calc function started:",os.getpid())

    clock_start=time.process_time()

  #  l_lock.acquire()

    #f=open(FILENAME,"a+")   #,buffering=0)
    #f.flush()
    for a in range(astart_val,astart_val+asize_of_env):
        payoff=-10*math.sin(a/44)*12*math.cos(a/33)*1000/(a+1)
        pline=(str(rowno)+","+str(a)+","+str(payoff)+"\n")
        q.put(pline)
      #  if q.full():
      #      print("q full")
        rowno+=1
    #f.flush()
    #f.close()

 #   l_lock.release()
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

if __name__ == '__main__':

    freeze_support()

    if(len(sys.argv) < 2) :
        print("Usage : python3 mp_simple1.py rownumbers")
        sys.exit()

    row_numbers = int(sys.argv[1])


    manager = multiprocessing.Manager()
    q = manager.Queue()    
   # pool = mp.Pool(mp.cpu_count() + 2)

    #put listener to work first
   # watcher = pool.apply_async(listener, (q,))


    windows=platform.system().lower()[:7]
    print("platform=",windows)
    if windows=="windows":
        EOL="\r\n"
    else:
        EOL='\n'

    cpus = multiprocessing.cpu_count()
 #   cpus = cpu_count()
    print("cpus=",cpus)
    cpus=cpus-2    # allow one for existing processes and one for watcher

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

    #lock=Lock()

    chunk=int(row_numbers/cpus)
    remainder=row_numbers%cpus
    print("rownumbers=",row_numbers," chunk size=",chunk," remainder=",remainder)
    with Pool(processes=cpus) as pool:  # processes=cpus
        watcher = pool.apply_async(listener, (q, ))
        
        if remainder!=0:
            multiple_results = [pool.apply_async(generate_payoff_environment_1d_mp,(0,remainder,q, ))]
        else:
            multiple_results=[]
        for i in range(0,cpus):
            multiple_results.append(pool.apply_async(generate_payoff_environment_1d_mp,(i*chunk+remainder,chunk,q, )))

       # print("mr=",len(multiple_results))
        for res in multiple_results:
            result=res.get(timeout=None)       
            res.wait()

    # Waits a bit for the child processes to do some work
    # because when the parent exits, childs are terminated.
 
    print("Generate payoff results finished")
    q.put("kill")
    pool.close()
    pool.join()




   # clock_end=time.process_time()
   # duration_clock=clock_end-clock_start
   # print("gen payoff rows=",row_numbers," Time:", duration_clock," secs.")



   # sleep(5)
    print("count file rows=",count_file_rows(FILENAME))
    print(FILENAME," #=",hash_a_file(FILENAME))      

"""
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
"""

#main()
