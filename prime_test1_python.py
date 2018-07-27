#def main():
    
#    for n in range(1,100):
#        print("n")
        
#from random import randint
#lower = 3
#upper = 1600

#import math
import random 

class DiceAnimal:
    dice1=0
    dice2=0
    total=0

    def rolldice(self):
        self.dice1=random.randint(1,6)
        self.dice2=random.randint(1,6)
        self.total=self.dice1+self.dice2


cat=DiceAnimal()
lobster=DiceAnimal()

cat.rolldice()
lobster.rolldice()

print("the cat rolled a", cat.dice1,"and a",cat.dice2)
print("the lobster rolled a", lobster.dice1,"and a",lobster.dice2)
if cat.total > lobster.total:
    print("the cat wins!")
elif lobster.total > cat.total:
    print("the lobster wins!")
else:
    print("its a tie!")
            

# uncomment the following lines to take input from the user
lower = int(input("Enter lower range: "))
upper = int(input("Enter upper range: "))
#old=lower
#print("Prime numbers between",lower,"and",upper,"are:")
#for num in range(lower,upper + 1):
#   for i in range(2,num):
#      if (num % i) == 0:
#         break
#      else:
#         print("%d is a prime number \n" %(num)),
      

chess_board=[8,8]

   
prime_numbers = [2]

fobj = open("games.txt")
for line in fobj:
    print(line.rstrip())
fobj.close()


def iterative_factorial(n):
    result = 1
    for i in range(2,n+1):
        result *= i
    return result

def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n-1)



def prime_gen(lower, upper):
    for i in range(lower, upper, 2):
        if not any(i % j == 0 for j in range(2, i)):
            prime_numbers.append(i)


def prime_list(num):
    list_of_prime = (2, )
    current_num = 2
    is_prime = True
    while len(list_of_prime) != num:
        current_num += 1
        if current_num % 2 != 0:
            for i in list_of_prime:
                if current_num % i == 0:
                    is_prime = False
                    break
            if is_prime == True:
                list_of_prime += (current_num, )
        #To reset the status
        is_prime = True
    return list_of_prime


import time
def measureTime(fn):
    start = time.clock()
    fn()
    end = time.clock()
    #return value in millisecond
    return (end - start)*1000

print('Prime gen List:', measureTime(lambda: prime_gen(lower,upper)), 'ms')
print('Prime List:', measureTime(lambda: prime_list(upper)), 'ms')
            
#prime_gen(upper)
print(prime_numbers)
n = int(input("Enter factorial n: "))
print('factorial n:', factorial(n))
print('iterative factorial:', measureTime(lambda: iterative_factorial(n)), 'ms')
print('recursive factorial:', measureTime(lambda: factorial(n)), 'ms')
