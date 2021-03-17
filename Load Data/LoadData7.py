# %%
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
from scipy.fftpack import fft
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#%%

raw = mne.io.read_raw_brainvision('D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02.vhdr', preload=True)
info = raw.info
Data, sfreq, chan, fs = raw._data,raw.info['sfreq'], raw.info['ch_names'], raw.info['sfreq']

raw.drop_channels(chan[2:len(chan)])
raw.drop_channels('C3_1')
raw.filter(0.5,4,method='iir')

raw2,anot = set_sw_kc_annot(raw,'D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02_2020July23_23-25.txt')

t_idx = anot.onset
t_dur = anot.duration
sig_raw = raw.get_data()

X = sig_raw[0]

condition = np.logical_and(t_dur>0.5,t_dur<0.9)
tmp_dur = t_dur[condition]
tmp_idx = t_idx[condition]


#%%

swaves = []
k = 0
for i in tmp_idx:
    tmp_muestras = fs*(1-tmp_dur[k])
    if tmp_muestras+tmp_dur[k]*fs < fs:
        tmp_muestras = tmp_muestras + (tmp_dur[k]-tmp_muestras)
    swaves.append(X[int(np.round(i*200-(tmp_muestras/2))):int(np.round((i+tmp_dur[k])*200+(tmp_muestras/2)))])
    k = k + 1

## INICIO NO-HIT ALEATORIOS
# np.random.seed(0)  # seed for reproducibility
# pos= np.random.randint(len(X), size=len(swaves))  # One-dimensional array

# no_swaves = []
# for i in range(len(pos)):
#   no_swaves.append(X[pos[i]:pos[i]+200])

# print('Number of NO slow waves:',len(no_swaves))
## FIN NO-HIT ALEATORIOS

no_swaves = []
offset = 135
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

#%%

def plot_scalogram(coefs,siglen):
    plt.figure(figsize=(12,4))
    plt.imshow(abs(coefs),extent=[0,siglen-1,120,0],interpolation='bilinear', cmap='jet', aspect='auto',vmax=255,vmin=0)
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(0,121,10))
    plt.xticks(np.arange(0,siglen-1,100))
    plt.show()
#%%
x_train = []
y_train = []
x_test = []
y_test = []

# TRANSFORMADA WAVELET
for i in range(len(swaves)):
    print(i)
    print("NO-HIT")
    scales = np.arange(1,121,0.1)
    coef, freqs = pywt.cwt(no_swaves[i],scales,'morl',sampling_period=1/200)
    coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    coef = coef[0::30]

    if(i<100):
        x_train.append(coef)
        y_train.append(0)
    else:
        x_test.append(coef)
        y_test.append(0)

    print("HIT")
    scales = np.arange(1,121,0.1)
    coef, freqs = pywt.cwt(swaves[i],scales,'morl',sampling_period=1/200)
    coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    coef = coef[0::30]
    
    if(i<100):
        x_train.append(coef)
        y_train.append(1)
    else:
        x_test.append(coef)
        y_test.append(1)

#%%
# Cuantas dibujar
n=len(swaves)
n=18

for i in range(n):
    print(i)
    plt.plot(swaves[i])
    plt.legend([f'Slow wave {i}'], loc='lower left')
    plt.plot(no_swaves[i])
    plt.legend([f'No hit {i}'], loc='lower left')
    plt.show()

#%%
x_train = np.array(x_train)
x_test = np.array(x_test)

# SI SE USA WAVELET
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
#----

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/255.0
x_test = x_test/255.0

#%%
y_train_raw = np.array(y_train)
y_test_raw = np.array(y_test)

# one-hot encoding using keras' numpy-related utilities
n_classes = 2
print("Shape of y_train [before one-hot encoding]: ", y_test_raw.shape)

# Generamos las nuevas variables que vamos a usar como etiquetas obejtivo
y_train_raw = np_utils.to_categorical(y_train_raw, n_classes)
y_test_raw = np_utils.to_categorical(y_test_raw, n_classes)
print("Shape of Y_train [after one-hot encoding]: ", y_test_raw.shape)

# %%
# building a linear stack of layers with the sequential model
n_input_layer = x_train.shape[1]
model = Sequential()
model.add(Dense(n_input_layer))                        
# model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['categorical_accuracy'],
    optimizer='adam')
# training the model and saving metrics in history
history = model.fit(x_train, y_train_raw,
          batch_size=256, epochs=200,
          verbose=1,
          validation_data=(x_test, y_test_raw))

#%%
# plotting the metrics
fig = plt.figure(figsize = (10,8))
plt.subplot(2,1,1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()
# %%
# Usamos el modelo para predecir sobre todas las instancias en ambos sets
y_train_pred_NN = model.predict(x_train)
y_test_pred_NN = model.predict(x_test)

# Tomamos como clase predicha aquella con mayor probabilidad
train_accuracy =  metrics.accuracy_score(y_train_pred_NN.argmax(axis=1),y_train)
test_accuracy =  metrics.accuracy_score(y_test_pred_NN.argmax(axis=1), y_test)

print('Accuracy en el train set:', train_accuracy)
print('Accuracy en el test set:', test_accuracy)

print ("Classification Report")
print(classification_report(y_test, y_test_pred_NN.argmax(axis=1)))
#print(classification_report(y_test, y_test_pred.round()))

print ("Confusion Report")
print(confusion_matrix(y_test, y_test_pred_NN.argmax(axis=1)))
#print(confusion_matrix(y_test, y_test_pred.round()))

# %%
# tmp = y_test_pred_NN.argmax(axis=1)
# result = []
# for i in range(len(y_test)):
#     if y_test[i] != tmp[i]:
#         result.append(i)

# plt.plot(x_train[result])


# %%
# salida = model.predict(x_test[4:5])
# print('Shape de la salida:',salida.shape)
# print('Salida:',salida)
# %%
# seconds = np.round(X.shape[0]/200).astype('uint16')
# PARA SERIE TEMPORAL
# seconds = 1200
# win_pred_tmp = []
# for k in range(seconds):
#     win_pred_tmp.append(X[k*200:(k+1)*200])
#     win_pred_tmp.append(X[k*200+50:(k+1)*200+50])
#     win_pred_tmp.append(X[k*200+100:(k+1)*200+100])
#     win_pred_tmp.append(X[k*200+150:(k+1)*200+150])
# -------------------

# PARA WAVELET
# seconds = 40#50#40#40#40#10
# offset = 7788#0#4690#6452#7715#1436


seconds = 2#50#40#40#40#10
offset = 7805#0#4690#6452#7715#1436


Dw = 50 # muestras que se desplaza la ventana
win_pred_tmp = []
scales = np.arange(1,121,0.1)
for k in range(offset,seconds+offset+1):
    print(k)
    coef, freqs = pywt.cwt(X[k*200:(k+1)*200],scales,'morl',sampling_period=1/200)
    coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    coef = coef[0::30]
    win_pred_tmp.append(coef)
    # plot_scalogram(coef,len(coef))
    plt.plot(X[k*200:(k+1)*200])
    plt.show()

    coef, freqs = pywt.cwt(X[k*200+Dw:(k+1)*200+Dw],scales,'morl',sampling_period=1/200)
    coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    coef = coef[0::30]
    win_pred_tmp.append(coef)
    # plot_scalogram(coef,len(coef))
    plt.plot(X[k*200+Dw:(k+1)*200+Dw])
    plt.show()

    coef, freqs = pywt.cwt(X[k*200+2*Dw:(k+1)*200+2*Dw],scales,'morl',sampling_period=1/200)
    coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    coef = coef[0::30]
    win_pred_tmp.append(coef)
    # plot_scalogram(coef,len(coef))
    plt.plot(X[k*200+2*Dw:(k+1)*200+2*Dw])
    plt.show()

    coef, freqs = pywt.cwt(X[k*200+3*Dw:(k+1)*200+3*Dw],scales,'morl',sampling_period=1/200)
    coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    coef = coef[0::30]
    win_pred_tmp.append(coef)
    # plot_scalogram(coef,len(coef))
    plt.plot(X[k*200+3*Dw:(k+1)*200+3*Dw])
    plt.show()

    # win_pred_tmp.append(X[k*200:(k+1)*200])
    # win_pred_tmp.append(X[k*200+50:(k+1)*200+50])
    # win_pred_tmp.append(X[k*200+100:(k+1)*200+100])
    # win_pred_tmp.append(X[k*200+150:(k+1)*200+150])

win_pred = np.array(win_pred_tmp)
win_pred = win_pred.reshape(win_pred.shape[0], win_pred.shape[1]*win_pred.shape[2])
win_pred = win_pred.astype('float32')
win_pred = win_pred/255.0



# ------------

# PARA SERIE TEMPORAL
# win_pred = []
# for i in range(len(win_pred_tmp)):
#     x_value = win_pred_tmp[i]
#     x_value = ((x_value - x_value.min()) * (1/(x_value.max() - x_value.min()) * 255)).astype('uint8')
#     win_pred.append(x_value)

# win_pred = np.array(win_pred)
# win_pred = win_pred.reshape(win_pred.shape[0], win_pred.shape[1])
# win_pred = win_pred.astype('float32')
# win_pred = win_pred/255.0
# --------------------
#%%
off = 0
for i in range(4):
    print(i)
    plot_scalogram(win_pred_tmp[i+off],len(win_pred_tmp[i+off]))


#%%
salida = model.predict(win_pred)

#%%
#plt.stem(salida[:,0])
#plt.show()
# plt.figure()
# plt.plot(salida[:,0])
# plt.show()

#%%
salida = model.predict(win_pred)
Y_norm = salida
Y_norm_tmp = Y_norm.argmax(axis=1)
Y_norm = Y_norm.argmax(axis=1)

N = len(range(offset*200,(seconds+offset)*200))
Xticks = np.linspace(offset,(seconds+offset),N)
plt.figure()
plt.plot(Xticks,X[offset*200:(seconds+offset)*200])

idx = 0
for k in range(offset,seconds+offset+1):
    if Y_norm[idx]==1:
        ejeX = np.linspace((k)*200,((k)+1)*200-1,200)/200
        plt.plot(ejeX, X[k*200:(k+1)*200], 'r')

    if Y_norm[idx+1]==1:
        ejeX = np.linspace((k)*200+Dw,((k)+1)*200+Dw-1,200)/200
        plt.plot(ejeX, X[k*200+Dw:(k+1)*200+Dw], 'r')
 
    if Y_norm[idx+2]==1:
        ejeX = np.linspace((k)*200+2*Dw,((k)+1)*200+2*Dw-1,200)/200
        plt.plot(ejeX, X[k*200+2*Dw:(k+1)*200+2*Dw], 'r')

    if Y_norm[idx+3]==1:
        ejeX = np.linspace((k)*200+3*Dw,((k)+1)*200+3*Dw-1,200)/200
        plt.plot(ejeX, X[k*200+3*Dw:(k+1)*200+3*Dw], 'r')
    
    idx = idx + 4


N = (seconds+2)
XLine = np.linspace(offset,offset+N-1,N)
YLine = np.ones(N)*75e-6
plt.plot(XLine,YLine,'r')
plt.plot(XLine,-YLine,'r')
plt.grid(True)
plt.show()
# %%
# N = seconds*200
# XLine = np.linspace(0,N-1,N)
# YLine = np.ones(N)*7.5e-6
# plt.plot(XLine,YLine,'r')
# plt.plot(XLine,-YLine,'r')
# plt.show
# %%
