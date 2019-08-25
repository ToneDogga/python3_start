import pickle
import numpy as np


path_screen_type = np.dtype(dict(
    names=['generation', 'epoch_length', 'bestgeneration', 'best_distance','stop'],
    formats=[np.int32, np.int32, np.int32, np.float64, (np.int32, (2,2))],
    offsets=[0, 8, 16, 16, 16]
))

path_screen=np.zeros((3,),dtype=path_screen_type)
print(path_screen)
input("?")
path_screen['epoch_length']=[1,2,3]
print(path_screen)
input("?")
path_screen['stop'][1]=[5,6]
print(path_screen)
input("?")
path_screen['stop'][1][1]=8
print(path_screen)
input("?")
path_screen['stop'][1][1][1]=7
print(path_screen)
input("?")

print(path_screen_type.names)
print(path_screen_type.fields)
input("?")


x_as_bytes = pickle.dumps(path_screen)
print(x_as_bytes)
print(type(x_as_bytes))

y = pickle.loads(x_as_bytes)
print(y)
