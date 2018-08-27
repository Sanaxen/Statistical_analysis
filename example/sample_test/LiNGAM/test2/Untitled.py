
# coding: utf-8

# In[2]:




# In[14]:

import numpy as np
import csv

size = 2000
csvflie=open("aaa.csv", 'w', encoding="utf_8")
wcsv = csv.writer(csvflie)

size = 2000
np.random.seed(2017)
x = np.random.uniform(size=size)
wcsv.writerow(x)

np.random.seed(1028)
y = 2*x + np.random.uniform(size=size)
wcsv.writerow(y)

np.random.seed(1029)
z = 3*x + 3*y + np.random.uniform(size=size);
#z = 0*x + 0*y + np.random.uniform(size=size);
wcsv.writerow(z)

np.random.seed(1026)
u = 4*x + 4*y + 4*z + np.random.uniform(size=size);
#u = 0*x + 0*y + 0*z + np.random.uniform(size=size);
wcsv.writerow(u)

np.random.seed(1025)
v = 5*x + 5*y + 5*z + 5*u + np.random.uniform(size=size);
#v = 5*x + 5*y + 5*z + 5*u + np.random.uniform(size=size);
wcsv.writerow(v)

np.random.seed(1024)
w = 6*x + 6*y + 6*z + 6*u + 6*v + np.random.uniform(size=size);
#w = 6*x + 0*y + 0*z + 0*u + 0*v + np.random.uniform(size=size);
wcsv.writerow(w)

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

f = open('u.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(u)
f.close()

f = open('v.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(v)
f.close()

f = open('w.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(w)
f.close()


