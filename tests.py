# tests learning python
#

import math
import time
import brickpi3
import pygame
import random
import turtle



#if __name__ == "__main__":
#    main()


def Hello(name2):
    print("hello",name2)

def circ(diameter):
    turtle.circle(diameter)
    turtle.getscreen()._root.mainloop()
    


def add(a,b):
    result=a + b
    return result


def main():
    print("hello what is your name?")
    name=input()
    if name=="anthony":
        print("hello")
    else:
        print("who are you?")
    
    print("input a rate and a distance")
    rate=float(input("Rate: "))
    distance=float(input("Distance: "))
    print("time:", (distance/rate))

# Hello()

    a=input("a:")
    b=input("b:")



    Hello(name)
    print("add=",add(a,b))

    word=input("please enter a four letter word")
    word_len=len(word)
    if word_len==4:
        print("4 ltters")
    elif word_len==3:
        print("3 letters")
    else:
        print("not 3 or 4 letters")


    print(math.sqrt(45))


    x=1
    while x<10:
        print(x)
        x=x+1

    words=["cat","dog","frog","mouse"]
    for word in words:
        print(word)

    for x in range(1,200,2):
        print(x)

    for i in range(1,10):
        print(random.randint(1,15))
        
    print(math.pi)
    circ(50)
    
    
main()





