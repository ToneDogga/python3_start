from __future__ import print_function
from __future__ import division

from multiprocessing import Pool, TimeoutError, Process, freeze_support, Queue
import multiprocessing
import sys
import itertools
import os
import time
import math
import platform
from pathlib import Path
#from os.path import basename

#from multiprocessing import Process, Queue


global EOL
EOL='\n'

#def test(start,fin): 
#    return((os.getpid(),list(range(start,fin))))

def f(q):
    q.put([42, None, 'hello'])


def generate_payoff_environment_1d_mp(xstart_val,xsize_of_env,filename):   
    rowno=0
    f=open(filename,"a+")     
    for x in range(xstart_val,xstart_val+xsize_of_env):
        payoff = -10*math.sin(x/44)*12*math.cos(x/33)*1000/(x+1)
        q.put(str(rowno)+","+str(x)+","+str(payoff)+EOL)
     #   f.write(str(rowno)+","+str(x)+","+str(payoff)+EOL)
        f.write(q.get()) 
        rowno+=1
    f.close()
    return(os.getpid())

def generate_payoff_environment_1d(xstart_val,xsize_of_env):   
    payoff = [(x,-10*math.sin(x/44)*12*math.cos(x/33)*1000/(x+1)) for x in range(xstart_val,xstart_val+xsize_of_env)]
    return(os.getpid(),payoff)    # could return os.getpid() for the process id


def create_payoff_file(payoff_list,filename):
    rowno=0
    end=len(payoff_list)
    f=open(filename,"w")
#        f.writelines("%s\n" %  for line in payoff_list)       
    while rowno<end: 
        f.write(str(rowno)+","+str(payoff_list[rowno][0])+","+str(payoff_list[rowno][1])+"\n")
        rowno+=1
    f.close()   


def count_file_rows(filename):
    with open(filename,'r') as f:
        return sum(1 for row in f)


def generate_payoff_environment_1d_file(xstart_val,xsize_of_env,filename):   
    rowno=0
    with open(filename,"w") as f:
        for x in range(xstart_val,xstart_val+xsize_of_env):
            payoff=-10*math.sin(x/44)*12*math.cos(x/33)*1000/(x+1)
      
            f.write(str(rowno)+","+str(x)+","+str(payoff)+"\n")
            rowno+=1
    f.close()   
    return(rowno)


def singlep_test(row_numbers):
    #ret=[]
    #for i in range(1):
    result=generate_payoff_environment_1d(0,row_numbers)
    print("pid=",result[0])
    return(result[1])

def multip_test(row_numbers,cpus,filename):
    #cpu=4
    print("cpus=",cpus)
    
    ret=[]
    chunk=int(row_numbers/cpus)
    remainder=row_numbers%cpus
    print("rownumbers=",row_numbers," chunk size=",chunk," remainder=",remainder)
    with Pool(processes=cpus) as pool:
        multiple_results = [pool.apply_async(generate_payoff_environment_1d_mp,(i*chunk,chunk,filename,)) for i in range(0,cpus)]
        multiple_results.append(pool.apply_async(generate_payoff_environment_1d_mp,(chunk*cpus,remainder,filename,)))
        for res in multiple_results:
            result=res.get(timeout=5)
           # ret.append(result[1])
            print("pid=",result)
    pool.close()               

    ret2=list(itertools.chain(*ret))
    return(ret2)


if __name__ == '__main__':
    freeze_support()
 #   Process(target=f).start()

    q = Queue()
    #p = Process(target=f, args=(q,))
 ##   p.start()
  #  print(q.get())    # prints "[42, None, 'hello']"
#    p.join()

    windows=platform.system().lower()[:7]
    print(windows)
    if windows=="windows":
        EOL="r\n"
    else:
        EOL='\n'

    cpus = multiprocessing.cpu_count()
    
    ret=[]
    clock_start=time.process_time()



    #my_file = Path("/home/pi/Python_Lego_projects/testfile1.csv")
    my_file="testfile1.csv"  #basename(my_file)
   # if my_file.is_file():
   # os.remove(my_file)



        
    multip_test(90023,cpus,my_file)  # 

    
    clock_end=time.process_time()
    duration_clock=clock_end-clock_start
    print("\n\nmultip + filegen in one. - Clock: duration_clock =", duration_clock)



    print("count file rows=",count_file_rows(my_file))
          

"""
    clock_start=time.process_time()

    generate_payoff_environment_1d_file(0,190023,"testfile2.csv")

    clock_end=time.process_time()
    duration_clock=clock_end-clock_start
    print("\n\nsingle gen file - Clock: duration_clock =", duration_clock)

    



    clock_start=time.process_time()

    ret=singlep_test(90023)

    clock_end=time.process_time()
    duration_clock=clock_end-clock_start
    print("\n\nsingle test - Clock: duration_clock =", duration_clock)

    print("\n",len(ret))





    ret=multip_test(190023)
   # print(ret)
    create_payoff_file(ret,"testfile.csv")

    clock_end=time.process_time()
    duration_clock=clock_end-clock_start
    print("\n\nmultip test +create file - Clock: duration_clock =", duration_clock)

    print("\n",len(ret))



    clock_start=time.process_time()


    """

          
