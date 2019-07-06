import numpy as np

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
