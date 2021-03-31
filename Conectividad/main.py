# %%
"""
=========================================================================
Compute source space connectivity and visualize it using a circular graph
=========================================================================

This example computes the all-to-all connectivity between 68 regions in
source space based on dSPM inverse solutions and a FreeSurfer cortical
parcellation. The connectivity is visualized using a circular graph which
is ordered based on the locations of the regions in the axial plane.

Links: 
https://mne.tools/stable/generated/mne.connectivity.spectral_connectivity.html

"""
# Authors: Matías Pretel <mpretel@itba.edu.ar>

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle, plot_sensors_connectivity



from pyvistaqt import BackgroundPlotter

print(__doc__)

# %%
###############################################################################
# Load our data
# -------------
#
raw = mne.io.read_raw_brainvision('D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Conectividad/Data/S1.vhdr', preload=True)
Data, sfreq, chan, fs = raw._data,raw.info['sfreq'], raw.info['ch_names'], raw.info['sfreq']
events2 = mne.events_from_annotations(raw)
events2 = events2[0]
events = events2

# %%
# Add a bad channel
raw.info['bads'] += ['EOG1_1','EOG2_1','EMG1_1','EMG2_1']

# Pick MEG channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

# %%
# Define epochs for left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0))
# epochs.plot()

# %%

# Compute connectivity for band containing the evoked response.
# We exclude the baseline period
# Delta 0 - 4
# Theta 4 - 7
# Alfa 7 - 14
# Beta 14 -21
# Gamma >30

# fmin, fmax = 3., 9.
fmin, fmax = 4., 7.
sfreq = raw.info['sfreq']  # the sampling frequency
tmin = 0.0  # exclude the baseline period
method = 'ciplv'
# method = 'pli'
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    epochs, method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
    faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)



# ******** PLOT ******** #
label_names = ['C3', 'C4', 'F3', 'F4', 'P3', 'P4'] # Orden original
node_order = ['F3', 'C3', 'P3', 'P4', 'C4', 'F4']
node_angles = circular_layout(label_names, node_order, start_pos=90,
    group_sep=20,group_boundaries=[0, len(label_names) / 2])

plot_connectivity_circle(con[:, :, 0],label_names,node_angles=node_angles,
    title=f'All-to-All Connectivity ({method})')

# %%
