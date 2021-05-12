# %%
import numpy as np
import mne
from mne import io
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle, plot_sensors_connectivity

# %%
def conn_exp(offset, N, path='Data/EXPERIMENT 2/COMPLETE REMINDER 40 MIN'):
    # offset = 1
    # N = 11
    idx = 0
    conmat = np.empty((N,6,6))
    i = 1
    for i in range(offset,offset+N):
        raw = mne.io.read_raw_brainvision(path+f'/S{i}.vhdr', preload=True)
        # raw = mne.io.read_raw_brainvision(f'Data/EXPERIMENT 2/COMPLETE REMINDER 40 MIN/S{i}.vhdr', preload=True)
        # raw = mne.io.read_raw_brainvision(f'Data/EXPERIMENT 2/INCOMPLETE REMINDER 40 MIN/S{i}.vhdr', preload=True)
        # raw = mne.io.read_raw_brainvision(f'Data/EXPERIMENT 2/NO REMINDER 40 MIN/S{i}.vhdr', preload=True)
        Data, sfreq, chan, fs = raw._data,raw.info['sfreq'], raw.info['ch_names'], raw.info['sfreq']
        events = mne.events_from_annotations(raw)[0]
        raw.info['bads'] += ['EOG1_1','EOG2_1','EMG1_1','EMG2_1']
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        # event_id, tmin, tmax = 1, -1, 2
        # epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, baseline=(-1, 0))
        event_id, tmin, tmax = 2, -4, 2
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, baseline=(-4, -3))

        # epochs.plot()


        # Compute connectivity for band containing the evoked response.
        # We exclude the baseline period
        
        Delta = [0, 4]
        Theta = [4, 7]
        Spindle = [11, 15]
        Alfa = [7, 14]
        Beta = [14, 21]
        Gamma = [30, 80]
        
        fmin, fmax = Theta
        
        sfreq = raw.info['sfreq']  # the sampling frequency
        tmin = 0.0  # exclude the baseline period
        # method = 'ciplv'
        # method = 'coh'
        # method = 'imcoh'
        # method = 'plv'
        method = 'pli'

        # Define wavelet frequencies and number of cycles
        # cwt_freqs = np.arange(fmin, fmax, 2)
        # cwt_n_cycles = cwt_freqs / 7.
        # con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        #     epochs, method=method, mode='cwt_morlet',
        #     cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles,
        #     sfreq=sfreq, fmin=fmin, fmax=fmax,
        #     faverage=False, tmin=tmin, mt_adaptive=False, n_jobs=1)

        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            epochs, method=method, mode='multitaper',
            sfreq=sfreq, fmin=fmin, fmax=fmax,
            faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

        conmat[idx,:,:] = con[:,:,0]
        idx = idx + 1

       # ******** PLOT ******** #
    # label_names = ['C3', 'C4', 'F3', 'F4', 'P3', 'P4'] # Orden original
    # node_order = ['F3', 'C3', 'P3', 'P4', 'C4', 'F4']
    # node_angles = circular_layout(label_names, node_order, start_pos=90,
    #     group_sep=20,group_boundaries=[0, len(label_names) / 2])

    # plot_connectivity_circle(con[:, :, 0],label_names,node_angles=node_angles,
    #     interactive=True,
    #     title=f'All-to-All Connectivity ({method})')
        
    return epochs,conmat

# %%
