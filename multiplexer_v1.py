Python 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 19:29:22) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 27, in <module>
    muiltiplexer()
NameError: name 'muiltiplexer' is not defined
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
addresses= [0, 0]
inputs= [0, 0, 0, 0]
Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 147, in <module>
    screen['generation'][0]=53   #[1,2,3]
NameError: name 'screen' is not defined
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
before addresses= [0, 0]
before data= [0, 0, 0, 0]
Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 46, in <module>
    multiplexer()
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 38, in multiplexer
    data=generate_data(data)
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 14, in generate_data
    data[i]=random.randint(0,1)
NameError: name 'random' is not defined
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
before addresses= [0, 0]
before data= [0, 0, 0, 0]
after addresses= [0, 0]
signal= [0, 0]
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
before addresses= [0, 0]
before data= [0, 0, 0, 0]
0
0
0
0
after addresses= [1, 0]
signal= [1, 0]
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
before addresses= [0, 0]
before data= [0, 0, 0, 0]
after addresses= [1, 0]
signal= [1, 0]
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
before addresses= [0, 0]
before data= [0, 0, 0, 0]
signal= [1, 0]
data= [1, 0, 0, 0]
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
before addresses= [0, 0]
before data= [0, 0, 0, 0]
signal= [0, 0]
data= [0, 0, 0, 0]
Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 55, in <module>
    multiplexer()
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 47, in multiplexer
    output_address=int("".join(signal),2)
TypeError: sequence item 0: expected str instance, int found
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
before addresses= [0, 0]
before data= [0, 0, 0, 0]
signal= [0, 0]
data= [0, 0, 0, 0]
Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 55, in <module>
    multiplexer()
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 50, in multiplexer
    print("output address=".output_address)
AttributeError: 'str' object has no attribute 'output_address'
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
before addresses= [0, 0]
before data= [0, 0, 0, 0]
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
before addresses= [0, 0]
before data= [0, 0, 0, 0]
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
before addresses= [0, 0]
before data= [0, 0, 0, 0]
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
?
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
?
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
?
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
?
signal= [1, 0]
data= [0, 1, 0, 0]
output address= 2
output= 0
?
signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0
?
signal= [1, 0]
data= [1, 1, 0, 0]
output address= 2
output= 0
?
signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0
?
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
?
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0
?
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0
?
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
?
signal= [1, 1]
data= [1, 1, 0, 0]
output address= 3
output= 0
?
signal= [1, 0]
data= [1, 1, 0, 0]
output address= 2
output= 0
?
signal= [1, 1]
data= [0, 1, 0, 0]
output address= 3
output= 0
?
signal= [1, 0]
data= [0, 1, 0, 0]
output address= 2
output= 0
?
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
?
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
?
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
?
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
?
signal= [0, 0]
data= [1, 1, 0, 0]
output address= 0
output= 1
?
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
?
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
?
signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0
?
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
?
Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 56, in <module>
    multiplexer()
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 54, in multiplexer
    input("?")
KeyboardInterrupt
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
signal= [0, 0]
data= [1, 1, 0, 0]
output address= 0
output= 1
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0
signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [1, 1]
data= [0, 1, 0, 0]
output address= 3
output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
signal= [0, 0]
data= [0, 1, 0, 0]
output address= 0
output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [0, 0]
data= [1, 1, 0, 0]
output address= 0
output= 1
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0
signal= [1, 1]
data= [1, 1, 0, 0]
output address= 3
output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [0, 0]
data= [1, 1, 0, 0]
output address= 0
output= 1
signal= [0, 0]
data= [1, 1, 0, 0]
output address= 0
output= 1
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [1, 1]
data= [0, 1, 0, 0]
output address= 3
output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [0, 1, 0, 0]
output address= 2
output= 0
signal= [1, 0]
data= [1, 1, 0, 0]
output address= 2
output= 0
Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 60, in <module>
    multiplexer(addresses,data)
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 40, in multiplexer
    print("output=",output)
KeyboardInterrupt
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0

output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2

output= 0
signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3

output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2

output= 0
signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3

output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2

output= 0
signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3

output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0

output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0

output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0

output= 1
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2

output= 0
signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3

output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0

output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2

output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0

output= 1
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0

output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0

output= 1
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2

output= 0
signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3

output= 0
signal= [1, 1]
data= [0, 1, 0, 0]
output address= 3

output= 0
signal= [1, 0]
data= [1, 1, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2

output= 0
signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3

output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [1, 1, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [0, 1, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [0, 1, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2

output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0

output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2

output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0

output= 0
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0

output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0

output= 1
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2

output= 0
signal= [0, 0]
data= [1, 1, 0, 0]
output address= 0

output= 1
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2

output= 0
signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2

output= 0
signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3

output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0

output= 1
signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2

output= 0
signal= [1, 1]
data= [0, 1, 0, 0]
output address= 3

output= 0
signal= [1, 0]
data= [0, 1, 0, 0]
output address= 2

output= 0
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0

output= 1
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0

output= 1
signal= Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 61, in <module>
    multiplexer(addresses,data)
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 33, in multiplexer
    print("signal=",signal)
KeyboardInterrupt
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0

signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [1, 1, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0

signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0

signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [1, 1, 0, 0]
output address= 3
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [1, 1, 0, 0]
output address= 3
output= 0

signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 61, in <module>
    multiplexer(addresses,data)
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 39, in multiplexer
    print("output address=",output_address)
KeyboardInterrupt
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1

signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1

signal= [1, 0]
data= [0, 1, 0, 0]
output address= 2
output= 0

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 63, in <module>
    multiplexer(addresses,data)
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 41, in multiplexer
    print("output address=",output_address)
KeyboardInterrupt
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1

signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0

signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0

signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [0, 0]
data= [0, 1, 0, 0]
output address= 0
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0

signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [0, 0, 0, 0]
output address= 3
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [1, 0, 0, 0]
output address= 3
output= 0

signal= [1, 1]
data= [0, 1, 0, 0]
output address= 3
output= 0

signal= [1, 0]
data= [1, 1, 0, 0]
output address= 2
output= 0

signal= [1, 1]
data= [1, 1, 0, 0]
output address= 3
output= 0

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1

signal= [0, 0]
data= [0, 1, 0, 0]
output address= 0
output= 0

signal= [1, 0]
data= [1, 1, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [1, 1, 0, 0]
output address= 2
output= 0

signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [0, 0]
data= [1, 1, 0, 0]
output address= 0
output= 1

signal= [0, 0]
data= [0, 0, 0, 0]
output address= 0
output= 0

signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1

signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1

signal= [0, 0]
data= [1, 0, 0, 0]
output address= 0
output= 1
Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 63, in <module>
    multiplexer(addresses,data)
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 42, in multiplexer
    print("output=",output)
KeyboardInterrupt
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 63, in <module>
    multiplexer(addresses,data)
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 32, in multiplexer
    data=generate_data(data)
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 14, in generate_data
    for i in len(data):
TypeError: 'int' object is not iterable
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
signal= [0, 1]
data= [1, 1, 0, 0]
output address= 1
output= 1

signal= [0, 1]
data= [0, 1, 1, 0]
output address= 1
output= 1

signal= [1, 1]
data= [0, 1, 0, 0]
output address= 3
output= 0

signal= [1, 1]
data= [0, 1, 1, 1]
output address= 3
output= 1

signal= [0, 1]
data= [0, 0, 1, 0]
output address= 1
output= 0

signal= [0, 0]
data= [0, 0, 1, 1]
output address= 0
output= 0

signal= [1, 0]
data= [0, 0, 1, 1]
output address= 2
output= 1

signal= [1, 1]
data= [0, 1, 0, 0]
output address= 3
output= 0

signal= [0, 0]
data= [1, 0, 1, 0]
output address= 0
output= 1

signal= [1, 1]
data= [0, 0, 1, 0]
output address= 3
output= 0

signal= [1, 1]
data= [0, 0, 0, 1]
output address= 3
output= 1

signal= [1, 1]
data= [1, 0, 0, 1]
output address= 3
output= 1

signal= [0, 1]
data= [0, 0, 0, 0]
output address= 1
output= 0

signal= [0, 0]
data= [0, 0, 1, 1]
output address= 0
output= 0

signal= [1, 0]
data= [1, 0, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [0, 1, 0, 1]
output address= 2
output= 0

signal= [1, 1]
data= [0, 1, 1, 1]
output address= 3
output= 1

signal= [1, 0]
data= [0, 0, 0, 0]
output address= 2
output= 0

signal= [1, 0]
data= [0, 0, 1, 0]
output address= 2
output= 1

signal= [1, 1]
data= [0, 1, 0, 0]
output address= 3
output= 0

signal= [1, 1]
data= [1, 1, 0, 1]
output address= 3
output= 1

signal= [0, 1]
data= [0, 1, 0, 0]
output address= 1
output= 1

signal= [0, 0]
data= [1, 1, 1, 0]
output address= 0
output= 1
Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 63, in <module>
    multiplexer(addresses,data)
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 42, in multiplexer
    print("output=",output)
KeyboardInterrupt
>>> 
 RESTART: C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py 
Address bits= 2  Data bits= 4  Total number of inputs= 6
signal= [0, 0]
output= 1

signal= [0, 0]
output= 0

signal= [1, 0]
output= 0

signal= [0, 0]
output= 1

signal= [1, 1]
output= 0

signal= [1, 1]
output= 0

signal= [0, 0]
output= 0

signal= [1, 0]
output= 0

signal= [1, 0]
output= 0

signal= [1, 1]
output= 0

signal= [0, 1]
output= 1

signal= [1, 0]
output= 1

signal= [0, 0]
output= 1

signal= [1, 0]
output= 1

signal= [0, 1]
output= 0

signal= [1, 1]
output= 0

signal= [1, 0]
output= 0

signal= [0, 0]
output= 1

signal= [0, 1]
output= 1

signal= [0, 0]
output= 1

signal= [1, 1]
output= 0

signal= [0, 1]
output= 1

signal= [1, 0]
output= 0

signal= [0, 0]
output= 0

signal= [1, 0]
output= 0

signal= [0, 1]
output= 1

signal= [1, 0]
output= 1

signal= [1, 0]
output= 0

signal= [1, 1]
output= 1

signal= [0, 1]
output= 0

signal= [0, 0]
output= 1

signal= [0, 0]
output= 0

signal= [0, 1]
output= 1

signal= [1, 1]
output= 0

signal= [0, 0]
output= 0

signal= [0, 0]
output= 0

signal= [1, 1]
output= 1

signal= [0, 1]
output= 1

signal= [0, 0]
output= 1

signal= [1, 1]
output= 0

signal= [1, 0]
output= 1

signal= [1, 1]
output= 0

signal= [1, 0]
output= 0

signal= [0, 1]
output= 0

Traceback (most recent call last):
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 63, in <module>
    multiplexer(addresses,data)
  File "C:/Users/Anthony Paech 2016/AppData/Local/Programs/Python/Python37-32/multiplexer6_v1-00.py", line 43, in multiplexer
    print("")
KeyboardInterrupt
>>> 
