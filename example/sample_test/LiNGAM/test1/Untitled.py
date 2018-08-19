
# coding: utf-8

# In[14]:

import numpy as np
import csv

size = 1000
np.random.seed(123)
x = np.random.uniform(size=size)

np.random.seed(456)
y = 3*x+ np.random.uniform(size=size)

np.random.seed(789)
z =  5*y+ np.random.uniform(size=size)

np.random.seed(135)
w = 2*x + 7*z+np.random.uniform(size=size)

xs = np.array([x,y,z,w]).T

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
writer.writerow(w)
f.close()


# In[15]:

xs.T


# In[16]:

from sklearn.decomposition import FastICA

n_samples, n_features = xs.shape
#ica = FastICA(random_state=1234, max_iter=1000).fit(xs)
winit=[
[0.001251, 0.563585, 0.193304, 0.808741],
[0.585009, 0.479873, 0.350291, 0.895962],
[0.822840, 0.746605, 0.174108, 0.858943],
[0.710501, 0.513535, 0.303995, 0.014985],    
]
ica = FastICA(w_init=winit, max_iter=1000).fit(xs)
print(ica.mixing_)
print()
W_ica = np.linalg.pinv(ica.mixing_)
print(W_ica)


# In[17]:

ica.mixing_


# In[18]:

W_ica_ = 1 / np.abs(W_ica) # もちろんこの前に成分が0でないことはチェックする
print(W_ica_)


# In[ ]:



