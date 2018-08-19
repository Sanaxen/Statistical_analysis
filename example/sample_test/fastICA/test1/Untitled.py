
# coding: utf-8

# In[44]:

import numpy as np
import matplotlib.pyplot as pl
from sklearn.decomposition import FastICA


# In[45]:

np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 10, n_samples)
s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
S = np.c_[s1, s2]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1], [0.5, 2]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations


# In[ ]:




# In[46]:

#pl.figure()
#pl.subplot(3, 1, 1)
#pl.plot(S)
#pl.title('True Sources')
#pl.subplot(3, 1, 2)
#pl.plot(X)
#pl.title('Observations (mixed signal)')
#pl.show()


# In[47]:

import csv
f = open('mix1.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(X[0:,0])
f.close()

f = open('mix2.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(X[0:,1])
f.close()

