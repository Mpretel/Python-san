# %%
############## LIBRERÍAS ##############
import matplotlib.pyplot as plt
import numpy as np 
import mne
import time
from funs import *
from sklearn.decomposition import FastICA, PCA

#%%
############## MAIN ##############
myprint()

raw = mne.io.read_raw_brainvision('D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Conectividad/Data/S1.vhdr', preload=True)
Data, sfreq, chan, fs = raw._data,raw.info['sfreq'], raw.info['ch_names'], raw.info['sfreq']
raw.plot()
# %%
tmp = Data[:,0:int(20*fs)]
# %%
ica = FastICA(n_components=10)
S = ica.fit_transform(tmp.T)  # Reconstruct signals
A = ica.mixing_  # Get estimated mixing matrix

# %%
S = np.ndarray(S)

# %%
plt.figure()
for i in range(10):
    plt.subplot(10,1,i+1)
    plt.plot(tmp.T[:,i])
plt.show()

# %%
plt.figure()
for i in range(10):
    plt.subplot(10,1,i+1)
    plt.plot(S[:,i])
plt.show()

# %%
raw2 = raw.copy()

