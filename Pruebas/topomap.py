# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne


biosemi_montage = mne.channels.make_standard_montage('standard_alphabetic')

# mne.channels.make_1020_channel_selections()

biosemi_montage.plot(show_names=False)
# %%
