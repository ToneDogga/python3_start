from sklearn import preprocessing
import numpy as np

lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])

print("1",lb.classes_)

print("2",lb.transform([1, 4, 2, 6]))

print("2a",lb.fit_transform(['yes', 'no', 'no', 'yes']))


lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))

print("3",lb.classes_)


print("4",lb.transform([0, 1, 2, 1]))

