from sklearn.naive_bayes import GaussianNB
import numpy as np
import random as rd
X = np.array([[-1, -1],[-2, -1],[-3, -2],[1, 1],[2, 1],[3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)
x1=rd.random()
y1=rd.random()
#print("%f \n %f "x1, y1)
x1 = np.array([[rd.randint(-4, 4), rd.randint(-4, 4)],[rd.randint(-4, 4), rd.randint(-4, 4)], \
     [rd.randint(-4, 4), rd.randint(-4, 4)],[rd.randint(-4, 4), rd.randint(-4, 4)], \
     [rd.randint(-4, 4), rd.randint(-4, 4)],[rd.randint(-4, 4), rd.randint(-4, 4)]])
pred = clf.predict(x1)
print(pred)
