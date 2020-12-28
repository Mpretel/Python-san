
#%%
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import pywt
from Annot import set_sw_kc_annot

raw = mne.io.read_raw_brainvision('D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02.vhdr', preload=True)
info = raw.info
Data, sfreq, chan = raw._data,raw.info['sfreq'], raw.info['ch_names']

raw.drop_channels(chan[2:len(chan)])
raw.drop_channels('C3_1')

raw2,anot = set_sw_kc_annot(raw,'D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02_2020July23_23-25.txt')

t_idx = anot.onset
t_dur = anot.duration

signal = raw.get_data()


signal2 = signal[0]


swaves = []
k = 0
for i in t_idx:
    swaves.append(signal2[int(np.round((i-t_dur[k])*200)):int(np.round((i+2*t_dur[k])*200))])
    k = k + 1

# plt.plot(signal)
# plt.show()

# raw2.plot()
# plt.show()


# for i in range(0,20):
#     plt.figure()
#     plt.plot(swaves[i])
 

sw_test = swaves[1]
plt.plot(sw_test)

wavelist = pywt.wavelist(kind='discrete')
scales = np.arange(1,31)
coef, freqs = pywt.cwt(sw_test,scales,'cgau2')
plt.figure(figsize=(12,4))
plt.imshow(abs(coef),extent=[0,200,30,1],interpolation='bilinear', cmap='jet', aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
plt.gca().invert_yaxis()
plt.yticks(np.arange(1,31,1))
plt.xticks(np.arange(0,201,10))
plt.show()  


# %%
raw2.filter(0.5,4,method='iir')
# %%
raw.plot()
plt.show()

# %%
raw2.plot()
plt.show()
# %%
