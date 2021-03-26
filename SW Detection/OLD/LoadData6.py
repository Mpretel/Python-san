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
# raw.filter(0.2,5,method='iir')

raw2,anot = set_sw_kc_annot(raw,'D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02_2020July23_23-25.txt')


t_idx = anot.onset
t_dur = anot.duration
sig_raw = raw.get_data()

X = sig_raw[0]

#%%
condition = np.logical_and(t_dur>0.5,t_dur<0.9)
tmp_dur = t_dur[condition]
tmp_idx = t_idx[condition]
# np.where(condition,t_dur,0)
# b = t_dur[condition]

# Se pueden usar 77 para entrenar, 26 para validación y 26 para test



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
offset = 126
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
# wavelist_cont = pywt.wavelist(kind='continuous')
# wavelist_disc = pywt.wavelist(kind='discrete')


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
    
    # plot_scalogram(coef,len(no_swaves[i]))
    coef = coef[0::30]
    # plot_scalogram(coef,len(no_swaves[i]))

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
    # plot_scalogram(coef,len(swaves[i]))
    coef = coef[0::30]
    # plot_scalogram(coef,len(swaves[i]))
    
    if(i<100):
        x_train.append(coef)
        y_train.append(1)
    else:
        x_test.append(coef)
        y_test.append(1)

    # N = 200
    # T = 1.0 / 200.0

    # y = swaves[i]
    # yf = fft(y)
    # xf = np.linspace(0.0, int(1.0/(2.0*T)), int(N/2))
    # plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))

    # y = no_swaves[i]
    # yf = fft(y)
    # xf = np.linspace(0.0, int(1.0/(2.0*T)), int(N/2))
    # plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))


    # plt.grid()
    # plt.title('Signal spectrum.')
    # plt.axis((0,20,0,100e-6))
    # plt.show()

# SERIES TEMPORALES
# for i in range(len(swaves)):
#     print(i)
#     print("NO-HIT")
#     #plot_scalogram(coef,no_swaves[i])
#     x_value = no_swaves[i]
#     x_value = ((x_value - x_value.min()) * (1/(x_value.max() - x_value.min()) * 255)).astype('uint8')
#     if(i<100):
#         x_train.append(x_value)
#         y_train.append(0)
#     else:
#         x_test.append(x_value)
#         y_test.append(0)


#     print("HIT")
#     x_value = swaves[i]
#     x_value = ((x_value - x_value.min()) * (1/(x_value.max() - x_value.min()) * 255)).astype('uint8')
    
#     if(i<100):
#         x_train.append(x_value)
#         y_train.append(1)
#     else:
#         x_test.append(x_value)
#         y_test.append(1)   
#%%
# plt.plot(no_swaves[1])
# plt.plot(no_swaves[4])
# plt.plot(no_swaves[5])
# plt.show    
#%%
    # N = 200
    # T = 1.0 / 200.0

    # y = swaves[i]
    # yf = fft(y)
    # xf = np.linspace(0.0, int(1.0/(2.0*T)), int(N/2))
    # plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))

    # y = no_swaves[i]
    # yf = fft(y)
    # xf = np.linspace(0.0, int(1.0/(2.0*T)), int(N/2))
    # plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))


    # plt.grid()
    # plt.title('Signal spectrum.')
    # plt.axis((0,100,0,100e-6))
    # plt.show()
#%%
#Cuantas dibujar
# n=len(swaves)
# n=10

# for i in range(n):
#   plt.plot(x_train[i])
#   plt.legend([f'Slow wave {i}'], loc='lower left')
#   plt.plot(no_swaves[i])
#   plt.legend([f'No hit {i}'], loc='lower left')
#   plt.show()

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
# model.add(Dense(n_input_layer/2, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
# model.add(Dense(n_input_layer/2, activation='relu'))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))
#%%
# compiling the sequential model
# model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
model.compile(loss='binary_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
#%%
# training the model and saving metrics in history
history = model.fit(x_train, y_train_raw,
          batch_size=32, epochs=200,
          verbose=1,
          validation_data=(x_test, y_test_raw))

#%%
# plotting the metrics
fig = plt.figure(figsize = (10,8))
plt.subplot(2,1,1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

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

#%%
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


seconds = 40#50#40#40#40#10
offset = 7788#0#4690#6452#7715#1436


Dw = 50 # muestras que se desplaza la ventana
win_pred_tmp = []
scales = np.arange(1,121,0.1)
for k in range(offset,seconds+offset+1):
    print(k)
    coef, freqs = pywt.cwt(X[k*200:(k+1)*200],scales,'morl',sampling_period=1/200)
    coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    coef = coef[0::30]
    win_pred_tmp.append(coef)

    coef, freqs = pywt.cwt(X[k*200+Dw:(k+1)*200+Dw],scales,'morl',sampling_period=1/200)
    coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    coef = coef[0::30]
    win_pred_tmp.append(coef)

    coef, freqs = pywt.cwt(X[k*200+2*Dw:(k+1)*200+2*Dw],scales,'morl',sampling_period=1/200)
    coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    coef = coef[0::30]
    win_pred_tmp.append(coef)

    coef, freqs = pywt.cwt(X[k*200+3*Dw:(k+1)*200+3*Dw],scales,'morl',sampling_period=1/200)
    coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    coef = coef[0::30]
    win_pred_tmp.append(coef)

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
# Y_norm = ((salida - salida.min()) * (1/(salida.max() - salida.min()))).astype('uint8')
# Y_norm = Y_norm.argmax(axis=1)
Y_norm = salida
Y_norm_tmp = Y_norm.argmax(axis=1)
Y_norm = Y_norm.argmax(axis=1)


# if Y_norm[0]==1 and Y_norm[1]==0:
#     Y_norm[0] = 0
# for i in range(1,Y_norm.shape[0]-1):
#     if Y_norm[i]==1 and Y_norm[i+1]==0 and Y_norm[i-1]==0:
#         Y_norm[i] = 0
# if Y_norm[Y_norm.shape[0]-1]==1 and Y_norm[Y_norm.shape[0]-2]==0:
#     Y_norm[Y_norm.shape[0]-1] = 0


N = len(range(offset*200,(seconds+offset)*200))
Xticks = np.linspace(offset,(seconds+offset),N)
plt.figure()
plt.plot(Xticks,X[offset*200:(seconds+offset)*200])
# plt.plot(X)

for k in range(offset,seconds+offset+1):

    # if Y_norm[k-offset]==1:
    #     # Ventana de 1 segundo
    #     ejeX = np.linspace((k-offset)*200,((k-offset)+1)*200-1,200)
    #     plt.plot(ejeX, X[k*200:(k+1)*200], 'r')

    if Y_norm[k-offset]==1:
        # Ventana de 250ms
        # ejeX = np.linspace((k-offset)*200,((k-offset)+1)*200-1,200)/200
        ejeX = np.linspace((k)*200,((k)+1)*200-1,200)/200
        plt.plot(ejeX, X[k*200:(k+1)*200], 'r')

    # if Y_norm[k-offset+1]==1:
    #     # Ventana de 250ms
    #     # ejeX = np.linspace((k-offset)*200+Dw,((k-offset)+1)*200+Dw-1,200)/200
    #     ejeX = np.linspace((k)*200+Dw,((k)+1)*200+Dw-1,200)/200
    #     # ejeX = np.linspace(k*200+Dw,k*200+2*Dw-1,Dw)/200
    #     plt.plot(ejeX, X[k*200+Dw:(k+1)*200+Dw], 'r')
    #     # plt.plot(ejeX, X[k*200+Dw:k*200+2*Dw], 'r')

    # if Y_norm[k-offset+2]==1:
    #     # Ventana de 250ms
    #     # ejeX = np.linspace((k-offset)*200+2*Dw,((k-offset)+1)*200+2*Dw-1,200)/200
    #     ejeX = np.linspace((k)*200+2*Dw,((k)+1)*200+2*Dw-1,200)/200
        
    #     plt.plot(ejeX, X[k*200+2*Dw:(k+1)*200+2*Dw], 'r')

    # if Y_norm[k-offset+3]==1:
    #     # Ventana de 250ms
    #     # ejeX = np.linspace((k-offset)*200+3*Dw,((k-offset)+1)*200+3*Dw-1,200)/200
    #     ejeX = np.linspace((k)*200+3*Dw,((k)+1)*200+3*Dw-1,200)/200
        
    #     plt.plot(ejeX, X[k*200+3*Dw:(k+1)*200+3*Dw], 'r')

    #     # ejeX = np.linspace((k-offset)*Dw,((k-offset)+1)*Dw-1,Dw)
    #     # plt.plot(ejeX, X[k*Dw:(k+1)*Dw], 'r')
        
    #     # ejeX = np.linspace((k-offset)*Dw+Dw,((k-offset)+1)*Dw+Dw-1,Dw)
    #     # plt.plot(ejeX, X[k*Dw+Dw:(k+1)*Dw+Dw], 'r')

    #     # ejeX = np.linspace((k-offset)*Dw+2*Dw,((k-offset)+1)*Dw+2*Dw-1,Dw)
    #     # plt.plot(ejeX, X[k*Dw+3*Dw:(k+1)*Dw+3*Dw], 'r')

    #     # ejeX = np.linspace((k-offset)*Dw+3*Dw,((k-offset)+1)*Dw+3*Dw-1,Dw)
    #     # plt.plot(ejeX, X[k*Dw+3*Dw:(k+1)*Dw+3*Dw], 'r')


    #     # ejeX = np.linspace(k*50,(k+1)*50-1,50)
    #     # plt.plot(ejeX, X[k*50:(k+1)*50], 'r')

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
