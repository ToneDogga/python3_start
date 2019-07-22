"""
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    p = Pool(4)
    print(p.map(f, [1, 2, 3]))


    



import multiprocessing as mp

pool=mp.Pool(4)
jobs=[]

def process(string):
    print(string)

if __name__ == '__main__':
    with open("payoff_3d.csv","r") as f:
        for line in f:
            jobs.append(pool.apply_async(process,(line)))

    for job in jobs:
        job.get()

    pool.close()


from multiprocessing import Process, Lock

def f(l, i):
    l.acquire()
    print ('hello world', i)
    l.release()

if __name__ == '__main__':
    lock = Lock()

    for num in range(10):
        Process(target=f, args=(lock, num)).start()


"""



from multiprocessing import Process
import os

def info(title):
    print (title)
    print ('module name:', __name__)
    if hasattr(os, 'getppid'):  # only available on Unix
        print ("parent process:", os.getppid())
    print ("process id:", os.getpid())

def f(name):
    info('function f')
    print ('hello', name)

if __name__ == '__main__':
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()        
