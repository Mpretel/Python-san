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

from funs import *

from pyvistaqt import BackgroundPlotter

print(__doc__)

# %%
###############################################################################
# Load our data
# -------------
#

exp2_CR = conn_exp(1, 11, 'Data/EXPERIMENT 2/COMPLETE REMINDER 40 MIN')
exp2_IR = conn_exp(12, 11, 'Data/EXPERIMENT 2/INCOMPLETE REMINDER 40 MIN')
exp2_NR = conn_exp(23, 11, 'Data/EXPERIMENT 2/NO REMINDER 40 MIN')

# %%
offset = 1
N = 11
idx = 0
conmat = np.empty((N,6,6))
for i in range(offset,offset+N):
    raw = mne.io.read_raw_brainvision(f'Data/EXPERIMENT 2/COMPLETE REMINDER 40 MIN/S{i}.vhdr', preload=True)
    # raw = mne.io.read_raw_brainvision(f'Data/EXPERIMENT 2/INCOMPLETE REMINDER 40 MIN/S{i}.vhdr', preload=True)
    # raw = mne.io.read_raw_brainvision(f'Data/EXPERIMENT 2/NO REMINDER 40 MIN/S{i}.vhdr', preload=True)
    Data, sfreq, chan, fs = raw._data,raw.info['sfreq'], raw.info['ch_names'], raw.info['sfreq']
    events = mne.events_from_annotations(raw)[0]
    raw.info['bads'] += ['EOG1_1','EOG2_1','EMG1_1','EMG2_1']
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    event_id, tmin, tmax = 1, -1, 2
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, baseline=(-1, 0))

    # Compute connectivity for band containing the evoked response.
    # We exclude the baseline period
    # Delta 0 - 4
    # Theta 4 - 7
    # Alfa 7 - 14
    # Beta 14 -21
    # Gamma >30

    # fmin, fmax = 0, 4
    fmin, fmax = 4, 7
    # fmin, fmax = 7, 14
    # fmin, fmax = 14, 21

    sfreq = raw.info['sfreq']  # the sampling frequency
    tmin = 0.0  # exclude the baseline period
    method = 'pli'
    # method = 'pli'
    con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        epochs, method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
        faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

    conmat[i-offset,:,:] = con[:,:,0]
# %%
    # ******** PLOT ******** #
    # label_names = ['C3', 'C4', 'F3', 'F4', 'P3', 'P4'] # Orden original
    # node_order = ['F3', 'C3', 'P3', 'P4', 'C4', 'F4']
    # node_angles = circular_layout(label_names, node_order, start_pos=90,
    #     group_sep=20,group_boundaries=[0, len(label_names) / 2])

    # plot_connectivity_circle(con[:, :, 0],label_names,node_angles=node_angles,
    #     title=f'All-to-All Connectivity ({method})')

# %%

# %%
from scipy import stats

media = stats.describe(conmat).mean
var = np.sqrt(stats.describe(conmat).variance)

# plt.imshow(con[:,:,0],cmap='Reds')
plt.imshow(media,cmap='Reds')
plt.colorbar()
label_names = ['C3', 'C4', 'F3', 'F4', 'P3', 'P4']
# x = ['P0', 'P1', 'P2', 'P4']
# y = ['C0', 'C1', 'C2', 'C4']
# plt.imshow(grid, interpolation='none')
plt.xticks(range(len(label_names)), label_names, fontsize=12)
plt.yticks(range(len(label_names)), label_names, fontsize=12)

plt.show()
plt.imshow(var,cmap='Reds')
plt.colorbar()
plt.xticks(range(len(label_names)), label_names, fontsize=12)
plt.yticks(range(len(label_names)), label_names, fontsize=12)
plt.show()
# %%
bp_labels = ['C4C3','F3C3','F4C3','P3C3','P4C3',
                'F3C4','F4C4','P3C4','P4C4',
                'F4F3','P3F3','P4F3',
                'P3F4','P4P3',
                'P4P3']
full_data = [conmat[:,1,0], conmat[:,2,0], conmat[:,3,0], conmat[:,4,0], conmat[:,5,0],
        conmat[:,2,1], conmat[:,3,1],conmat[:,4,1],conmat[:,5,1],
        conmat[:,3,2], conmat[:,4,2],conmat[:,5,2],
        conmat[:,4,3], conmat[:,5,3],
        conmat[:,5,4]]

plt.boxplot(full_data,labels=bp_labels)
plt.show()
# %%
