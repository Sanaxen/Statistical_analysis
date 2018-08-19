
# coding: utf-8

# In[2]:

import numpy as np
import scipy.io.wavfile as wf

rate1, data1 = wf.read('loop1.wav')
rate2, data2 = wf.read('strings.wav')
rate3, data3 = wf.read('fanfare.wav')
if rate1 != rate2 or rate2 != rate3:
    raise ValueError('Sampling_rate_Error')


# In[3]:

mix_1 = data1 * 0.6 + data2 * 0.3 + data3 * 0.1
mix_2 = data1 * 0.3 + data2 * 0.2 + data3 * 0.5
mix_3 = data1 * 0.1 + data2 * 0.5 + data3 * 0.4
y = [mix_1, mix_2, mix_3]
y = [(y_i * 32767 / max(np.absolute(y_i))).astype(np.int16) for y_i in np.asarray(y)]

wf.write('mix_1.wav', rate1, y[0])
wf.write('mix_2.wav', rate2, y[1])
wf.write('mix_3.wav', rate3, y[2])


# In[31]:

zip(mix_1)


# In[38]:

import csv

f = open('mix_1.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(mix_1)
f.close()

f = open('mix_2.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(mix_2)
f.close()

f = open('mix_3.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(mix_3)
f.close()


# In[42]:

import numpy as np
data1 = np.loadtxt('mix_1.csv', delimiter=',', dtype='float')
data2 = np.loadtxt('mix_2.csv', delimiter=',', dtype='float')
data3 = np.loadtxt('mix_3.csv', delimiter=',', dtype='float')


# In[43]:

y = [data1, data2, data3]
y = [(y_i * 32767 / max(np.absolute(y_i))).astype(np.int16) for y_i in np.asarray(y)]

wf.write('data1.wav', rate1, y[0])
wf.write('data2.wav', rate2, y[1])
wf.write('data3.wav', rate3, y[2])


# In[ ]:



