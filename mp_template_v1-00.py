from __future__ import print_function
from __future__ import division

from multiprocessing import Pool, Process, Lock, freeze_support, Queue, Lock, Manager
from os import getpid
from time import sleep
#from random import random
#import itertools
import platform
import math
import sys
import time
import os
import hashlib
#from pathlib import Path

#EOL='\n'
FILENAME="testfile1.csv"
import multiprocessing 
from timeit import default_timer as timer

fn = 'c:/temp/temp.txt'


def listener(q):    #,l_lock):
    '''listens for messages on the q, writes to file. '''
    print("Queue listener started on:",os.getpid())
   # l_lock.acquire()
    f=open(FILENAME,"w")   #,buffering=8096)
    #f.flush()
    while True:
       # if q.full():
       #     print("q full")
       #     break
        m = q.get()    #timeout=20)  get_nowait
        if m == "kill":
            print("trying to kill listener process.  Flushing q.  qsize=",q.qsize())
            #f.write("killed\n")
            while not q.empty():
                m=q.get()
                f.write(m)            
                f.flush()
            break
        else: 
            f.write(m)  #+":"+q.qsize()+"\n")
          #  f.write(str(q.qsize())+'\n')
            f.flush()
       
  #  q.close()
    f.close()
    print("Queue listener closed on:",os.getpid())
   # l_lock.release()
    return(True)

def add_to_queue(msg,q):
    temp_buffer=[]
      #  If queue is full, put the message in a temporary buffer.
      #  If the queue is not full, adding the message to the queue.
      #  If the buffer is not empty and that the message queue is not full,
      #  putting back messages from the buffer to the queue.
  #  print(q.qsize())    
    if q.full():
        temp_buffer.append(msg)
    else:
        q.put(msg)
     #   q.put(str(q.qsize())+'\n')
        if len(temp_buffer) > 0:
            add_to_queue(temp_buffer.pop())
    #history.append(q.qsize)
    return   #(history)

def generate_payoff_environment_1d_mp(astart_val,asize_of_env,q,linewrite,lwremainder):    #l_lock  
    rowno=0
    print("payoff calc function started:",os.getpid())

    clock_start=timer()  #.process_time()

    pline=""
    for a in range(astart_val,astart_val+asize_of_env):
        payoff=-10*math.sin(a/44)*12*math.cos(a/33)*1000/(a+1)
        pline=pline+(str(rowno)+","+str(a)+","+str(payoff)+"\n")
        if rowno%linewrite==0:
            add_to_queue(pline,q)
            pline=""     
        rowno+=1
        
    if pline:
        add_to_queue(pline,q)

 
    clock_end=timer()  #.process_time()
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
    freeze_support()

    if(len(sys.argv) < 2) :
        print("Usage : python3 mp_simple1.py rownumbers")
        sys.exit()

    row_numbers = int(sys.argv[1])

    if len(sys.argv)<3:
        linewrite=10000
    else:    
        linewrite=int(sys.argv[2])
        if linewrite<=0:
            linewrite=10000
    
        
    manager = multiprocessing.Manager()
    q = manager.Queue()    
   
    cpus = multiprocessing.cpu_count()
   
    windows=platform.system().lower()[:7]
    print("platform=",windows)
    if windows=="windows":
        EOL="\r\n"
    else:
        EOL='\n'

    print("cpus=",cpus)

    clock_start=timer() #time.process_time()

    chunk=int(row_numbers/(cpus-1))   # less one cpu for listener
    remainder=row_numbers%(cpus-1)

    lwremainder=row_numbers%linewrite
    
    print("rownumbers=",row_numbers," chunk size=",chunk," remainder=",remainder)
    print("lines per write=",linewrite," remainder=",lwremainder)

    with Pool(processes=cpus) as pool:  # processes=cpus-1
         #put listener to work first
        watcher = pool.apply_async(listener, args=(q, ))


        if remainder!=0:
            multiple_results = [pool.apply_async(generate_payoff_environment_1d_mp,args=(0,remainder,q,linewrite,lwremainder ))]
        else:
            multiple_results=[]
        for i in range(0,cpus-1):
            multiple_results.append(pool.apply_async(generate_payoff_environment_1d_mp,args=(i*chunk+remainder,chunk,q,linewrite,lwremainder )))
        for res in multiple_results:
            result=res.get(timeout=None)       
            res.wait()

        print("Generate payoff results finished")

        print("killing listener")
        q.put("kill")
        result=watcher.get(timeout=None) 
        watcher.wait()

        
    print("try to close pool")
    pool.close()
    print("pool closed.  trying to join() pool")
    pool.join()
    print("pool join() complete")



    clock_end=timer()   #time.process_time()
    duration_clock=clock_end-clock_start
    print("gen payoff file windows multi processor rows=",row_numbers," Time:", duration_clock," secs.")

   
    print("count file rows=",count_file_rows(FILENAME))
    print(FILENAME," #=",hash_a_file(FILENAME))      



 


if __name__ == '__main__':
    main()
