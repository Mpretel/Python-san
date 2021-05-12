# %%
# %matplotlib qt
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

# matplotlib qt

###############################################################################
# Load our data
# -------------
#
# mne.channels.make_1020_channel_selections(info["ch_names"])

epoch1, exp2_CR = conn_exp(1, 11, path='Data/EXPERIMENT 2/COMPLETE REMINDER 40 MIN')
epoch2, exp2_IR = conn_exp(12, 11, path='Data/EXPERIMENT 2/INCOMPLETE REMINDER 40 MIN')
epoch3, exp2_NR = conn_exp(23, 11, path='Data/EXPERIMENT 2/NO REMINDER 40 MIN')

bp_labels = ['C4C3','F3C3','F4C3','P3C3','P4C3',
                'F3C4','F4C4','P3C4','P4C4',
                'F4F3','P3F3','P4F3',
                'P3F4','P4F4',
                'P4P3']


data_cr = [exp2_CR[:,1,0], exp2_CR[:,2,0], exp2_CR[:,3,0], exp2_CR[:,4,0], exp2_CR[:,5,0],
        exp2_CR[:,2,1], exp2_CR[:,3,1], exp2_CR[:,4,1], exp2_CR[:,5,1],
        exp2_CR[:,3,2], exp2_CR[:,4,2], exp2_CR[:,5,2],
        exp2_CR[:,4,3], exp2_CR[:,5,3],
        exp2_CR[:,5,4]]
data_ir = [exp2_IR[:,1,0], exp2_IR[:,2,0], exp2_IR[:,3,0], exp2_IR[:,4,0], exp2_IR[:,5,0],
        exp2_IR[:,2,1], exp2_IR[:,3,1], exp2_IR[:,4,1], exp2_IR[:,5,1],
        exp2_IR[:,3,2], exp2_IR[:,4,2], exp2_IR[:,5,2],
        exp2_IR[:,4,3], exp2_IR[:,5,3],
        exp2_IR[:,5,4]]
data_nr = [exp2_NR[:,1,0], exp2_NR[:,2,0], exp2_NR[:,3,0], exp2_NR[:,4,0], exp2_NR[:,5,0],
        exp2_NR[:,2,1], exp2_NR[:,3,1], exp2_NR[:,4,1], exp2_NR[:,5,1],
        exp2_NR[:,3,2], exp2_NR[:,4,2], exp2_NR[:,5,2],
        exp2_NR[:,4,3], exp2_NR[:,5,3],
        exp2_NR[:,5,4]]

# %%
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 5),constrained_layout = True)

bplot1 = ax1.boxplot(data_cr,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=bp_labels)  # will be used to label x-ticks
ax1.set_title('Complete Reminder')
ax1.set_ylim([0,1])

bplot2 = ax2.boxplot(data_ir,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=bp_labels)  # will be used to label x-ticks
ax2.set_title('Incomplete Reminder')
ax2.set_ylim([0,1])

bplot3 = ax3.boxplot(data_nr,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=bp_labels)  # will be used to label x-ticks
ax3.set_title('No Reminder')
ax3.set_ylim([0,1])

# fill with colors
# colors = ['pink', 'lightblue', 'lightgreen']
# for bplot in (bplot1, bplot2):
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)

# adding horizontal grid lines
for ax in [ax1, ax2, ax3]:
    ax.yaxis.grid(True)
    # ax.set_xlabel('Channels')
    ax.set_ylabel('Connectivity')
ax3.set_xlabel('Channels')

plt.show()

# plt.boxplot(full_data,labels=bp_labels)
# plt.show()

# %% 3, 8 y 13
from scipy.stats import wilcoxon
w_values = []
p_values = []

# P3C3 con todos
for i in range(15):
    print(i)
    if i==3:
        w_values.append(0)
        p_values.append(1)       
    else:
        # w, p = wilcoxon(data_cr[3],data_cr[i])
        w, p = wilcoxon(data_ir[3],data_ir[i])
        # w, p = wilcoxon(data_nr[3],data_nr[i])
        w_values.append(w)
        p_values.append(p)

plt.figure()
plt.stem(p_values)
plt.show()

w_values = []
p_values = []
# P4C4 con todos
for i in range(15):
    print(i)
    if i==8:
        w_values.append(0)
        p_values.append(1)       
    else:
        # w, p = wilcoxon(data_cr[8],data_cr[i])
        w, p = wilcoxon(data_ir[8],data_ir[i])
        # w, p = wilcoxon(data_nr[8],data_nr[i])
        w_values.append(w)
        p_values.append(p)

plt.figure()
plt.stem(p_values)
plt.show()

w_values = []
p_values = []
# P4F4 con todos
for i in range(15):
    print(i)
    if i==13:
        w_values.append(0)
        p_values.append(1)       
    else:
        # w, p = wilcoxon(data_cr[13],data_cr[i])
        w, p = wilcoxon(data_ir[13],data_ir[i])
        # w, p = wilcoxon(data_nr[13],data_nr[i])
        w_values.append(w)
        p_values.append(p)

plt.figure()
plt.stem(p_values)
plt.show()



# %%
# bplot1['medians'][3].get_ydata()
media_cr = []
for i in range(15):
    media_cr.append(bplot1['medians'][i].get_ydata()[0])
media_ir = []
for i in range(15):
    media_ir.append(bplot2['medians'][i].get_ydata()[0])
media_nr = []
for i in range(15):
    media_nr.append(bplot3['medians'][i].get_ydata()[0])

plt.figure()
plt.plot(media_cr,'o--')
plt.plot(media_ir,'o--')
plt.plot(media_nr,'o--')
plt.legend(['CR','IR','NR'])
plt.show()
# %%
# ******** PLOT ******** #
label_names = ['C3', 'C4', 'F3', 'F4', 'P3', 'P4'] # Orden original
node_order = ['F3', 'C3', 'P3', 'P4', 'C4', 'F4']
node_angles = circular_layout(label_names, node_order, start_pos=90,
    group_sep=20,group_boundaries=[0, len(label_names) / 2])

fig = plt.figure(num=None, figsize=(12, 7), facecolor='black')
plot_connectivity_circle(exp2_CR[2,:,:],label_names,node_angles=node_angles,
    interactive=True,fig=fig,subplot=(1, 3, 1),
    title='All-to-All Connectivity (CR)')

plot_connectivity_circle(exp2_IR[0,:,:],label_names,node_angles=node_angles,
    interactive=True,fig=fig,subplot=(1, 3, 2),
    title='All-to-All Connectivity (IR)')

plot_connectivity_circle(exp2_NR[0,:,:],label_names,node_angles=node_angles,
    interactive=True,fig=fig,subplot=(1, 3, 3),
    title='All-to-All Connectivity (NR)')

plt.show()
# %%
raw = mne.io.read_raw_brainvision('Data/EXPERIMENT 2/COMPLETE REMINDER 40 MIN/S1.vhdr', preload=True)
# %%
