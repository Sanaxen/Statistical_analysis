
# coding: utf-8

# In[2]:

from numpy.random import *
import numpy as np 

size=3000
a = 3
b = -2
c = 2

np.random.seed(2017)
x = np.random.uniform(size=size)

#np.random.seed(1028)
y = a * x + b*1 + beta(15,5, size=size)

#np.random.seed(1028)
z = c * y +  beta(15,5, size=size)

u = np.random.normal(10,2, size=size)

import csv

f = open('x.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(x)
f.close()

f = open('y.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(y)
f.close()

f = open('z.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(z)
f.close()

f = open('w.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(u)
f.close()


# In[ ]:



