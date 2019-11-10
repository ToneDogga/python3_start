import numpy as np
array=np.array([0,0,0,0,1,2,3,4])
list_str=[]
end=len(array)
list_str.append(str(array[i].tolist()) for i in range(0,end-1))
print list_str


print(map(str, array.tolist()))
