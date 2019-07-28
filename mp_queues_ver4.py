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
MAX_WORKERS=10
FILENAME="testfile1.csv"




#global PROC_COUNT
#PROC_COUNT=0

class Testing_mp(object):
    def __init__(self):
        
       # Initiates a queue, a pool and a temporary buffer, used only
       # when the queue is full.



       # self.manager = multiprocessing.Manager()
       # self.qin = self.manager.Queue()
       # self.qout = self.manager.Queue()

       # self.qin = Queue()  # first queue write the lists into this
       # self.qout = Queue()   # second queue take from the first queue and then write to file
       # self.q3 = Queue()   # third queue take from the second queue and write to file
        self.pool = Pool(processes=MAX_WORKERS, initializer=self.worker_main,)   #initargs=([],),)
      #  self.temp_bufferin = []
      #  self.temp_bufferout = []
        #self.temp_buffer3 = []
        #self.proc=[]   # counts the child processes running

    def add_to_queuein(self, msg,qin):

        temp_bufferin=[]
      #  If queue is full, put the message in a temporary buffer.
      #  If the queue is not full, adding the message to the queue.
      #  If the buffer is not empty and that the message queue is not full,
      #  putting back messages from the buffer to the queue.
        
        if qin.full():
            temp_bufferin.append(msg)
        else:
            qin.put(msg)
            if len(temp_bufferin) > 0:
                add_to_queue(temp_bufferin.pop())

    def write_to_queuein(self,item,qin):
        
       # This function writes some messages to the queue.


        self.add_to_queuein(item,qin)
     #   print("qin size=",qin.qsize())   #.pop())
        
        
      #  for i in range(10):
      #      self.add_to_queue("First item for loop %d" % i)
            # Not really needed, just to show that some elements can be added
            # to the queue whenever you want!
      #      sleep(random()*2)
       #     self.add_to_queue("Second item for loop %d" % i)
            # Not really needed, just to show that some elements can be added
            # to the queue whenever you want!
        #    sleep(random()*2)


    def add_to_queueout(self, msg,qout):

        temp_bufferout=[]
      #  If queue is full, put the message in a temporary buffer.
      #  If the queue is not full, adding the message to the queue.
      #  If the buffer is not empty and that the message queue is not full,
      #  putting back messages from the buffer to the queue.
        
        if qout.full():
            temp_bufferout.append(msg)
        else:
            qout.put(msg)
            if len(temp_bufferout) > 0:
                add_to_queue(temp_bufferout.pop())

    def write_to_queueout(self,item,qout):
        
       # This function writes some messages to the queue.


        self.add_to_queueout(item,qout)

        


    def worker_main(self):

        
        #Waits indefinitely for an item to be written in the queue.
        #Finishes when the parent process terminates.
        
        print("Process {0} started".format(getpid()))

     #   proc.append(os.getpid())
     #   print("proc=",proc)
       # self.proc_count+=1
        while True:
            # If queuein is not empty, pop the next element and do the work.
            # If queuein is empty, wait indefinitly until an element get in the queue.
           
           #try:
           #     print("try to find qin")
            item = qin.get(block=True, timeout=None)
            #    print("after get")
            
           # except Queue.Empty:
           #     item = None
           #     pass
             #   print("qin empty")

           # else:                
              #  print("item found in qin=",item)
                # If `False`, the program is not blocked. `Queue.Empty` is thrown if 
                # the queue is empty
               # print("{0} retrieved:".format(getpid()))
            self.write_to_queueout(item,qout)
          #      print("queueout len=",qout.qsize())
            if create_payoff_file():
                print("create payoff file completed. pid=",os.getpid())   #," proc[]=",proc)




def generate_payoff_environment_1d_mp(astart_val,asize_of_env):   
    rowno=0
    payoff=[]
    for a in range(astart_val,astart_val+asize_of_env):
        payoff.append((a,-10*math.sin(a/44)*12*math.cos(a/33)*1000/(a+1)))
        rowno+=1     
    return(payoff)    #(os.getpid(),payoff)


#def generate_payoff_environment_2d_mp(bstart_val,bsize_of_env):   
#    rowno=0
#    payoff=[]
#    for b in range(bstart_val,bstart_val+bsize_of_env):
#        generate_payoff_environment_1d_mp(b,bstart_val+bsize_of_env)
#        rowno+=1     
#    return(payoff)    #(os.getpid(),payoff)






def generate_payoff_environment_5d_file(linewidth,astart_val,asize_of_env,bstart_val,bsize_of_env,cstart_val,csize_of_env,dstart_val,dsize_of_env,estart_val,esize_of_env,filename):   
        rowno=0
    
        total_rows=asize_of_env*bsize_of_env*csize_of_env*dsize_of_env*esize_of_env    
  #  with open(filename,"w") as f:
        for a in range(astart_val,astart_val+asize_of_env):
            print("\rProgress: [%d%%] " % (rowno/total_rows*100),end='\r', flush=True)
            for b in range(bstart_val,bstart_val+bsize_of_env):
                for c in range(cstart_val,cstart_val+csize_of_env):
                    for d in range(dstart_val,dstart_val+dsize_of_env):
                        for e in range(estart_val,estart_val+esize_of_env):
                            payoff=100*math.sin(a/44)*120*math.cos(b/33)-193*math.tan(c/55)+78*math.sin(d/11)-98*math.cos(e/17)
#                            f.write(str(rowno)+","+str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(e)+","+str(payoff)+"\n")
                      #      w=str(rowno)+","+str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(e)+","+str(payoff)
                            padding=linewidth-len(w)-1
                            w=w+" "*padding
                            rowno+=1
                        #    f.write(w+"\n")
       # f.close()
        print("")
        return(rowno)


#def init(L):
#    global lock
#    lock = L

def create_payoff_file():
   # try:
    payoff_list = qout.get(block=True, timeout=None)  
        # If `False`, the program is not blocked. `Queue.Empty` is thrown if 
        # the queue is empty
    
    #except Queue.Empty:
     #   print("qout empty, pid=",os.getpid())
    #    payoff_list = None
    #   return(False)

    #else:
     #   print(payoff_list) 
    if payoff_list:
            rowno=0
            end=len(payoff_list)
       #     print("create payoff file:",os.getpid()," payoff len=",end)
            f=open(FILENAME,"a")       
            while rowno<end: 
                f.write(str(rowno)+","+str(payoff_list[rowno][0])+","+str(payoff_list[rowno][1])+EOL)
                rowno+=1
            f.close()
            return(True)   #os.gedpid())
    else:
            print("payoff empty")
            return(False)


# Python program to find SHA256 hash string of a file
def hash_a_file(filename):
    #filename = input("Enter the input file name: ")
    sha256_hash = hashlib.sha256()
    with open(filename,"rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    return(sha256_hash.hexdigest())



def count_file_rows(filename):
    with open(filename,'r') as f:
        return sum(1 for row in f)


def multip_test(row_numbers,cpus,qin,qout):
    #cpu=4
   # print("cpus=",cpus)

    #q = Queue()
    #    self.pool = Pool(processes=MAX_WORKERS, initializer=self.worker_main,)
    #temp_buffer = []


    
    # qin is the results of generate payoff environent
    # qin is got by worker_main and put to qout where it is then written to file


    #pool = mp.Pool(processes=4)
    #results = [pool.apply(cube, args=(x,)) for x in range(1,7)]
    #print(results)
    

    chunk=int(row_numbers/cpus)
    remainder=row_numbers%cpus
    print("rownumbers=",row_numbers," chunk size=",chunk," remainder=",remainder)
    with Pool(processes=cpus) as pool:  # processes=cpus
        #for i in range(0,cpus):
        #    r.append(pool.apply_async(generate_payoff_environment_1d_mp,(i*chunk,chunk,)))
        #r.append(pool.apply_async(generate_payoff_environment_1d_mp,(chunk*cpus,remainder,)))    
      #  multiple_results = [pool.starmap(generate_payoff_environment_1d_mp,(i*chunk,chunk,)) for i in range(0,cpus)]
      #  multiple_results.append(pool.starmap(generate_payoff_environment_1d_mp,(chunk*cpus,remainder,)))


        multiple_results = [pool.apply_async(generate_payoff_environment_1d_mp,(i*chunk,chunk,)) for i in range(0,cpus)]
        multiple_results.append(pool.apply_async(generate_payoff_environment_1d_mp,(chunk*cpus,remainder,)))
        for res in multiple_results:
            result=res.get(timeout=None)
 #           result=res.get(timeout=None)
       #     mp_class.write_to_queue2(res)
            
            mp_class.write_to_queuein(result,qin)
            
            res.wait()


   #print(pool._pool)    #.return_code
    #mp_class.q1.put(None)    

    #ret2=list(itertools.chain(*ret))
   # return     #(ret2)

   # multiple_results[0].wait()
   # multiple_results[1].wait()
   # multiple_results[2].wait()
   # multiple_results[3].wait()
   # multiple_results[4].wait()
 
    
    print("Generate payoff results finished")
    
    pool.close()
    pool.join()
   # multiple_results.join()
  #  r.wait()
    #mp_class.q2.put(None)
    
    return   #(p)    







# Warning from Python documentation:
# Functionality within this package requires that the __main__ module be
# importable by the children. This means that some examples, such as the
# multiprocessing.Pool examples will not work in the interactive interpreter.
if __name__ == '__main__':
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

    manager = multiprocessing.Manager()
    qin = manager.Queue()
    qout = manager.Queue()
    
    mp_class = Testing_mp()
  #  mp_class.write_to_queue()
    # Waits a bit for the child processes to do some work
    # because when the parent exits, childs are terminated.
 #   sleep(5)


     
    ret=[]
    clock_start=time.process_time()



    my_file = Path("/home/pi/Python_Lego_projects/testfile1.csv")
    #my_file = Path("/home/pi/Python_Lego_projects/testfile1.csv")

  #  my_file="testfile1.csv"  #basename(my_file)
    if my_file.is_file():
        print(FILENAME," #=",hash_a_file(FILENAME))
        os.remove(FILENAME)  
        print(FILENAME,":removed")    
    


        
    multip_test(123,MAX_WORKERS,qin,qout)  # put the payoff lists into the queue
    # the worker main function will see them in the queue and create the csv file


    while not qout.empty():
        print("\rqout len=",qout.qsize(),end='\r', flush=True)
    
    print("\nqout empty")


    #while not mp_class.q3.empty():
    #    print("\rq3 len=",mp_class.q3.qsize(),end='\r', flush=True)

    #print("\n\nq3 empty")


    


   # print("1count file rows=",count_file_rows(FILENAME))
   # print(FILENAME," #=",hash_a_file(FILENAME))      


   # clock_end=time.process_time()
   # duration_clock=clock_end-clock_start
   # print("multip + filegen in one. - Clock: duration_clock =", duration_clock)

    #while PROC_COUNT!=cpus+1:
    #    print("PROC_COUNT=",PROC_COUNT)



    time.sleep(1)
    
    print("count file rows=",count_file_rows(FILENAME))
    print(FILENAME," #=",hash_a_file(FILENAME))      


###########################################
