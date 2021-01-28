# %%
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import pywt
from Annot import set_sw_kc_annot
from scipy import signal
from scipy.interpolate import make_interp_spline, BSpline
from skimage.restoration import denoise_wavelet

#%%

raw = mne.io.read_raw_brainvision('D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02.vhdr', preload=True)
info = raw.info
Data, sfreq, chan, fs = raw._data,raw.info['sfreq'], raw.info['ch_names'], raw.info['sfreq']

raw.drop_channels(chan[2:len(chan)])
raw.drop_channels('C3_1')

raw2,anot = set_sw_kc_annot(raw,'D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02_2020July23_23-25.txt')


t_idx2 = anot.onset
t_dur2 = anot.duration

sig_raw = raw.get_data()
sig_raw_cp = raw.copy()
sig_raw_cp.filter(0.2,4,method='iir')
sig_filt = sig_raw_cp.get_data()


X = sig_raw[0]
Y = sig_filt[0]

#%%
condition = np.logical_and(t_dur2>0.5,t_dur2<0.9)
a = np.where(condition,t_dur2,0)
b = t_dur2[condition]



#%%
swaves = []
k = 0
for i in t_idx:
    swaves.append(X[int(np.round((i-t_dur[k])*200)):int(np.round((i+2*t_dur[k])*200))])
    k = k + 1


swaves_filt = []
k = 0
for i in t_idx:
    swaves_filt.append(Y[int(np.round((i-t_dur[k])*200)):int(np.round((i+2*t_dur[k])*200))])
    k = k + 1

no_swaves = []
for i in range(0,20):
    no_swaves.append(X[ 200*i + 200*400 : 200*(i+2) + 200*400])

no_swaves_filt = []
for i in range(0,20):
    no_swaves_filt.append(Y[ 200*i + 200*400 : 200*(i+2) + 200*400])

#%%
sw_idx = 0
plt.figure(figsize=[10,5])
plt.plot(swaves[sw_idx])
plt.plot(swaves_filt[sw_idx])
plt.plot(no_swaves[sw_idx])
plt.plot(no_swaves_filt[sw_idx])
plt.show()


f1, Pxx_den1 = signal.welch(swaves[sw_idx], fs, nperseg=1024)
f2, Pxx_den2 = signal.welch(swaves_filt[sw_idx], fs, nperseg=1024)
f3, Pxx_den3 = signal.welch(no_swaves[sw_idx], fs, nperseg=1024)
f4, Pxx_den4 = signal.welch(no_swaves_filt[sw_idx], fs, nperseg=1024)

plt.figure()
plt.semilogy(f1, Pxx_den1)
plt.semilogy(f2, Pxx_den2)
plt.semilogy(f3, Pxx_den3)
plt.semilogy(f4, Pxx_den4)

plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')

plt.ylim([1e-15, 1e-8])
plt.xlim([0, 20])

plt.show()

#%%
# scales = np.arange(1,4)
# coefs = pywt.wavedec(swaves[sw_idx], 'db10',level=5)

# plt.stem(coefs[0])
# plt.show()
#cA5, cD5, cD4, cD3, cD2, cD1 = coefs

#sw_wl_filt = pywt.waverec(cA5,'db10')
# sw_wl_filt = denoise_wavelet(swaves[sw_idx],method='BayesShrink',mode='hard',wavelet_levels=2,wavelet='db2',rescale_sigma='True')
# plt.figure(figsize=[10,3])
# plt.plot(swaves[sw_idx])

# plt.plot(sw_wl_filt)
# plt.show()

#%%

# plt.figure(figsize=[10,3])
# xnew = np.linspace(0, len(cA5), 355)
# hola = make_interp_spline(np.arange(len(cA5)), cA5, k=3)  # type: BSpline
# smooth = hola(xnew)
# plt.plot(xnew,smooth)
# plt.show()
# plt.figure(figsize=[10,3])
# plt.plot(swaves[sw_idx])
# plt.show()

# plt.subplot(6,1,1)
# plt.plot(cA5)
# plt.subplot(6,1,2)
# plt.plot(cD5)
# plt.subplot(6,1,3)
# plt.plot(cD4)
# plt.subplot(6,1,4)
# plt.plot(cD3)
# plt.subplot(6,1,5)
# plt.plot(cD2)
# plt.subplot(6,1,6)
# plt.plot(cD1)
# plt.show()


#%%

wavelist_cont = pywt.wavelist(kind='continuous')
wavelist_disc = pywt.wavelist(kind='discrete')

# scales = np.arange(1,31)
# coef, freqs = pywt.cwt(sw_test,scales,'shan')
# plt.figure(figsize=(12,4))
# plt.imshow(abs(coef),extent=[0,len(sw_test)-1,30,1],interpolation='bilinear', cmap='jet', aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
# plt.gca().invert_yaxis()
# plt.yticks(np.arange(1,31,1))
# plt.xticks(np.arange(0,len(sw_test)-1,10))
# plt.show()  

# sw_test = swaves_filt[1]
# plt.figure(figsize=(12,4))
# plt.xticks(np.arange(0,len(sw_test)-1,10))
# plt.plot(sw_test)


# scales = np.arange(1,31)
# coef, freqs = pywt.cwt(sw_test,scales,'shan')
# plt.figure(figsize=(12,4))
# plt.imshow(abs(coef),extent=[0,len(sw_test)-1,30,1],interpolation='bilinear', cmap='jet', aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
# plt.gca().invert_yaxis()
# plt.yticks(np.arange(1,31,1))
# plt.xticks(np.arange(0,len(sw_test)-1,10))
# plt.show()  




# raw2.filter(0.5,4,method='iir')

# raw.plot()
# plt.show()


# raw2.plot()
# plt.show()


# %%
# file = open('dump2.txt', 'w')
# file.write(str(swaves_filt[0]))
# file.close()
# #%%
# file = open('dump3.txt', 'w')
# file.write(str(no_swaves[0]))
# file.close()
# # %%
# file = open('dump4.txt', 'w')
# file.write(str(no_swaves_filt[0]))
# file.close()
# %%
import time
 

# Wait for 5 seconds
for i in range(len(swaves)):
    # a.append(len(swaves[i]))
    plt.psd(swaves[i],Fs=200,NFFT=512)
    plt.show()
    f1, Pxx_den1 = signal.welch(swaves[i], fs, nperseg=1024)
    plt.semilogy(f1, Pxx_den1)
    plt.show()
    time.sleep(1)
# %%
