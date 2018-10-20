
# coding: utf-8

# In[2]:

import numpy as np
import numpy as np
import scipy.io.wavfile as wf

rate1, data1 = wf.read('loop1.wav')
rate2, data2 = wf.read('strings.wav')
rate3, data3 = wf.read('fanfare.wav')
if rate1 != rate2 or rate2 != rate3:
    raise ValueError('Sampling_rate_Error')

data1 = np.loadtxt('output0.csv', delimiter=',', dtype='float')
data2 = np.loadtxt('output1.csv', delimiter=',', dtype='float')
data3 = np.loadtxt('output2.csv', delimiter=',', dtype='float')

y = [data1, data2, data3]
y = [(y_i * 32767 / max(np.absolute(y_i))).astype(np.int16) for y_i in np.asarray(y)]

wf.write('data1.wav', rate1, y[0])
wf.write('data2.wav', rate2, y[1])
wf.write('data3.wav', rate3, y[2])


# In[ ]:



