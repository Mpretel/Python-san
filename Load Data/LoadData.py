import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from Annot import set_sw_kc_annot


raw = mne.io.read_raw_brainvision('D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02.vhdr', preload=True)
info = raw.info
Data, sfreq, chan = raw._data,raw.info['sfreq'], raw.info['ch_names']

raw.drop_channels(chan[2:len(chan)])
raw.drop_channels('C3_1')

raw2,anot = set_sw_kc_annot(raw,'Load Data/Data/ExpS68-S02_2020July23_23-25.txt')
raw2.plot()
plt.show()