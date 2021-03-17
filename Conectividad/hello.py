# %%
############## LIBRERÍAS ##############
import matplotlib.pyplot as plt
import numpy as np 
import random
import mne
import pywt
from sklearn.metrics import classification_report, confusion_matrix
from Annot import set_sw_kc_annot
import time
import sklearn.metrics as metrics
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, plot_model
from keras.regularizers import l1
# from skimage import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#%%
############## LIBRERÍAS ##############