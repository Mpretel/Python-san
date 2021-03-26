# %%
############## LIBRERÍAS ##############
import matplotlib.pyplot as plt
import numpy as np 
import mne
import time
from funs import *
from sklearn.decomposition import FastICA, PCA

import simpleaudio as sa

import sounddevice as sd
import soundfile as sf


#  %%
# filename = 'D:\Disco de 500\Work\Doctorado\Repository\Python-san\Python-san\Conectividad\Audios\snare-violin.wav'
filename = 'D:\Disco de 500\Work\Doctorado\Repository\Python-san\Python-san\Conectividad\Audios\warning.wav'
# filename = 'D:\Disco de 500\Work\Doctorado\Repository\Python-san\Python-san\Conectividad\Audios\\techno.wav'
# Extract data and sampling rate from file
data, fs = sf.read(filename, dtype='float32')  
# data = data[:,1]
# sd.play(data, fs)
# status = sd.wait()  # Wait until file is done playing

# %%

ica = FastICA(n_components=2)
S = ica.fit_transform(data)  # Reconstruct signals
A = ica.mixing_  # Get estimated mixing matrix

# pca = PCA(n_components=2)
# H = pca.fit_transform(data)  # Reconstruct signals based on orthogonal components

# %%
ica = FastICA(n_components=2)
# S = np.array([s2,s2])
S = ica.fit_transform(S)  # Reconstruct signals
A = ica.mixing_  # Get estimated mixing matrix
# %%
s1 = S[:,0]
maxmax = np.max(s1)
minmin = np.min(s1)

# s1 = ((s1 - minmin) * (1/(maxmax - minmin)))
s1 = (s1 * (1/(maxmax - minmin)))
s2 = S[:,1]
maxmax = np.max(s2)
minmin = np.min(s2)
# s2 = ((s2 - minmin) * (1/(maxmax - minmin)))
s2 = (s2 * (1/(maxmax - minmin)))

# %%
# sd.play(data, fs)
# status = sd.wait()  # Wait until file is done playing
sd.play(s1, fs)
status = sd.wait()  # Wait until file is done playing
sd.play(s2, fs)
status = sd.wait()  # Wait until file is done playing


#%%
############## MAIN ##############
myprint()

raw = mne.io.read_raw_brainvision('D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Conectividad/Data/S1.vhdr', preload=True)
Data, sfreq, chan, fs = raw._data,raw.info['sfreq'], raw.info['ch_names'], raw.info['sfreq']
raw.plot()
# %%

# From this point, we would like to do the opposite and recover the original sources, undoing the mixing process.
ica = FastICA(n_components=10)
S_ = ica.fit_transform(Data)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(Data, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(Data)  # Reconstruct signals based on orthogonal components

#%%
# #############################################################################
# Plot results
plt.figure(10)
for i in range(10):


plt.title('ICA 1')
plt.subplot(3,1,1)
plt.plot(S_[:,0], color='red')
plt.title('ICA 2')
plt.subplot(3,1,2)
plt.plot(S_[:,1], color='steelblue')
plt.title('ICA 3')
plt.subplot(3,1,3)
plt.plot(S_[:,2], color='orange')
