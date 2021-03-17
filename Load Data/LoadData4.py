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
import time

#%%

raw = mne.io.read_raw_brainvision('D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02.vhdr', preload=True)
info = raw.info
Data, sfreq, chan, fs = raw._data,raw.info['sfreq'], raw.info['ch_names'], raw.info['sfreq']

raw.drop_channels(chan[2:len(chan)])
raw.drop_channels('C3_1')
# raw.filter(0.2,30,method='iir')

raw2,anot = set_sw_kc_annot(raw,'D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02_2020July23_23-25.txt')


t_idx = anot.onset
t_dur = anot.duration
sig_raw = raw.get_data()

X = sig_raw[0]

#%%
condition = np.logical_and(t_dur>0.5,t_dur<0.9)
tmp_dur = t_dur[condition]
tmp_idx = t_idx[condition]
# np.where(condition,t_dur,0)
# b = t_dur[condition]

# Se pueden usar 77 para entrenar, 26 para validación y 26 para test



#%%
swaves = []
k = 0
for i in tmp_idx:
    # swaves.append(X[int(np.round((i-tmp_dur[k])*200)):int(np.round((i+2*tmp_dur[k])*200))])
    # swaves1.append(X[int(np.round(i*200)):int(np.round((i+tmp_dur[k])*200))])
    tmp_muestras = fs*(1-tmp_dur[k])
    if tmp_muestras+tmp_dur[k]*fs < fs:
        tmp_muestras = tmp_muestras + (tmp_dur[k]-tmp_muestras)
    swaves.append(X[int(np.round(i*200-(tmp_muestras/2))):int(np.round((i+tmp_dur[k])*200+(tmp_muestras/2)))])
    k = k + 1


no_swaves = []
# for i in range(0,130):
#     # no_swaves.append(X[ 200*i + 200*400 : 200*(i+2) + 200*400])
#     no_swaves.append(X[ 200*i + 200*400 : 200*(i+1) + 200*400])

offset = 126
tmp_fs = 200
for i in range(0,66):
    no_swaves.append(X[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
offset = 335
for i in range(0,12):
    no_swaves.append(X[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
offset = 15940
for i in range(0,20):
    no_swaves.append(X[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
offset = 3195
for i in range(0,22):
    no_swaves.append(X[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
offset = 370
for i in range(0,9):
    no_swaves.append(X[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])

print(len(no_swaves))
#%%
# Xlin = np.linspace(0, 200,200)
# Ylin = np.ones(200)*75e-6
# Ylin2 = np.ones(200)*(-75e-6)
# plt.plot(Xlin,Ylin,linestyle='solid',color='red',linewidth=2)
# plt.show()
#%%
# for i in range(0,129):
#     print(i)
#     plt.plot(no_swaves[i])
#     plt.plot(Xlin,Ylin,linestyle='solid',color='red',linewidth=2)
#     plt.plot(Xlin,Ylin2,linestyle='solid',color='red',linewidth=2)
#     plt.show()

#%%

def plot_scalogram(coefs,sig2plot):
    plt.figure(figsize=(12,4))
    #plt.imshow(abs(coefs),extent=[0,len(sig2plot)-1,120,0],interpolation='bilinear', cmap='jet', aspect='auto',vmax=abs(coefs).max(),vmin=-abs(coefs).max())
    plt.imshow(abs(coefs),extent=[0,len(sig2plot)-1,120,0],interpolation='bilinear', cmap='jet', aspect='auto',vmax=255,vmin=0)
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(0,121,10))
    plt.xticks(np.arange(0,len(sig2plot)-1,100))
    plt.show()
#%%
wavelist_cont = pywt.wavelist(kind='continuous')
wavelist_disc = pywt.wavelist(kind='discrete')


x_train = []

# for i in range(len(no_swaves)):
for i in range(2):
    # print("NOISE 2 SEG")
    # scales = np.arange(1,121,0.1)
    # coef, freqs = pywt.cwt(no_swaves[i],scales,'morl')
    # coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    # plot_scalogram(coef,no_swaves[i])
    # plt.figure(figsize=(12,4))
    # plt.imshow(abs(coef),extent=[0,len(no_swaves[i])-1,120,0],interpolation='bilinear', cmap='jet', aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
    # plt.gca().invert_yaxis()
    # plt.yticks(np.arange(0,121,10))
    # plt.xticks(np.arange(0,len(no_swaves[i])-1,100))
    # plt.show()   
    print(i)
    print("NOISE")
    scales = np.arange(1,121,0.1)
    coef, freqs = pywt.cwt(no_swaves[i],scales,'morl',sampling_period=1/200)
    coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    plot_scalogram(coef,no_swaves[i])
    # plt.figure(figsize=(12,4))
    # plt.imshow(abs(coef),extent=[0,len(no_swaves1[i])-1,120,0],interpolation='bilinear', cmap='jet', aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
    # plt.gca().invert_yaxis()
    # plt.yticks(np.arange(0,121,10))
    # plt.xticks(np.arange(0,len(no_swaves1[i])-1,100))
    # plt.show()        
    
    # print("SLOW WAVE 2 SEG")
    # scales = np.arange(1,121,0.1)
    # coef, freqs = pywt.cwt(swaves1[i],scales,'morl')
    # coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    # plot_scalogram(coef,no_swaves[i])
    # plt.figure(figsize=(12,4))
    # plt.imshow(abs(coef),extent=[0,len(swaves1)-1,30,1],interpolation='bilinear', cmap='jet', aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
    # plt.gca().invert_yaxis()
    # plt.yticks(np.arange(0,121,10))
    # plt.xticks(np.arange(0,len(swaves1)-1,10))
    # plt.show() 

    print("SLOW WAVE")
    scales = np.arange(1,121,0.1)
    coef, freqs = pywt.cwt(swaves[i],scales,'morl',sampling_period=1/200)
    coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    plot_scalogram(coef,no_swaves[i])
    # plt.figure(figsize=(12,4))
    # plt.imshow(abs(coef),extent=[0,len(swaves[i])-1,30,1],interpolation='bilinear', cmap='jet', aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
    # plt.gca().invert_yaxis()
    # plt.yticks(np.arange(0,121,10))
    # plt.xticks(np.arange(0,len(swaves[i])-1,10))
    # plt.show() 
    
    time.sleep(2)

#%%

#Estoy probando este cell

i = 6


plt.figure(figsize=[10,3])
plt.plot(no_swaves[k])
plt.show()

scales = np.arange(1,201,0.1)
coef, freqs = pywt.cwt(no_swaves[i],scales,'morl')
coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
plt.figure(figsize=(12,4))
#plt.imshow(abs(coef),extent=[0,len(no_swaves[i])-1,30,1], interpolation='bilinear',cmap='jet', aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
plt.imshow(coef,extent=[0,len(swaves[i])-1,30,1],interpolation='bilinear', cmap='jet', aspect='auto',vmax=coef.max(),vmin=coef.min())
plt.gca().invert_yaxis()
plt.yticks(np.arange(1,31,1))
plt.xticks(np.arange(0,len(no_swaves[i])-1,10))
plt.show()

tmp = coef[0::200]
plt.imshow(abs(tmp),extent=[0,len(no_swaves[i])-1,30,1],interpolation='bilinear',cmap='jet', aspect='auto',vmax=abs(tmp).max(),vmin=-abs(tmp).max())
plt.show()

plt.figure(figsize=[10,3])
plt.plot(swaves[k])
plt.show()

#scales = np.arange(1,121,1)
coef, freqs = pywt.cwt(swaves[i],scales,'morl',sampling_period=1/200)
coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
plt.figure(figsize=(12,4))
#plt.imshow(abs(coef),extent=[0,len(swaves[i])-1,30,1],interpolation='bilinear', cmap='jet', aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
plt.imshow(coef,extent=[0,len(swaves[i])-1,30,1],interpolation='bilinear', cmap='jet', aspect='auto',vmax=coef.max(),vmin=coef.min())
plt.gca().invert_yaxis()
plt.yticks(np.arange(1,31,1))
plt.xticks(np.arange(0,len(swaves[i])-1,10))
plt.show()

tmp = coef[0::200]
#plt.imshow(abs(tmp),extent=[0,len(no_swaves[i])-1,30,1],interpolation='bilinear', cmap='jet', aspect='auto',vmax=abs(tmp).max(),vmin=-abs(tmp).max())
#plt.show()

tmp = ((tmp - tmp.min()) * (1/(tmp.max() - tmp.min()) * 255)).astype('uint8')
plt.imshow(abs(tmp),extent=[0,len(no_swaves[i])-1,30,1],interpolation='bilinear', cmap='jet', aspect='auto')
plt.show()


#%%
scales = np.arange(1,200,1)
#scales = np.arange(1,21,1)

coef, freqs = pywt.cwt(swaves[0],scales,'morl',sampling_period=1/200)
plt.figure(figsize=(12,4))
plt.imshow(abs(coef),extent=[0,len(swaves[0])-1,20,1],interpolation='bilinear', cmap='jet', aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
plt.gca().invert_yaxis()
plt.yticks(np.arange(1,21,1))
plt.xticks(np.arange(0,len(swaves[0])-1,10))
plt.show()

coef, freqs = pywt.cwt(no_swaves[0],scales,'morl',sampling_period=1/200)
plt.figure(figsize=(12,4))
plt.imshow(abs(coef),extent=[0,len(no_swaves[0])-1,20,1],interpolation='bilinear', cmap='jet', aspect='auto',vmax=abs(coef).max(),vmin=-abs(coef).max())
plt.gca().invert_yaxis()
plt.yticks(np.arange(1,21,1))
plt.xticks(np.arange(0,len(no_swaves[0])-1,10))
plt.show() 


#%%
# sw_idx = 0
# plt.figure(figsize=[10,5])
# plt.plot(swaves[sw_idx])
# plt.plot(swaves_filt[sw_idx])
# plt.plot(no_swaves[sw_idx])
# plt.plot(no_swaves_filt[sw_idx])
# plt.show()


# f1, Pxx_den1 = signal.welch(swaves[sw_idx], fs, nperseg=1024)
# f2, Pxx_den2 = signal.welch(swaves_filt[sw_idx], fs, nperseg=1024)
# f3, Pxx_den3 = signal.welch(no_swaves[sw_idx], fs, nperseg=1024)
# f4, Pxx_den4 = signal.welch(no_swaves_filt[sw_idx], fs, nperseg=1024)

# plt.figure()
# plt.semilogy(f1, Pxx_den1)
# plt.semilogy(f2, Pxx_den2)
# plt.semilogy(f3, Pxx_den3)
# plt.semilogy(f4, Pxx_den4)

# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')

# plt.ylim([1e-15, 1e-8])
# plt.xlim([0, 20])

# plt.show()

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
