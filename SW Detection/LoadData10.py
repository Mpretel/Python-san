# %%
########## LIBRERÍAS ##########
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
########## DATASET ##########
# raw = mne.io.read_raw_brainvision('D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02.vhdr', preload=True)
raw = mne.io.read_raw_brainvision('Data\ExpS68-S02.vhdr', preload=True)
Data, sfreq, chan, fs = raw._data,raw.info['sfreq'], raw.info['ch_names'], raw.info['sfreq']
raw.drop_channels(chan[2:len(chan)])
raw.drop_channels('C3_1')

# Tomo la señal del C4, las anotaciones y la duración de cada SW
sig_orig_raw = raw.get_data()

# Se filtra la señal con filtro pasabanda entre 0.5Hz y 4Hz
raw.filter(0.5,4,method='iir')
raw2,anot = set_sw_kc_annot(raw,'Data\ExpS68-S02_2020July23_23-25.txt')

t_idx = anot.onset
t_dur = anot.duration
sig_raw = raw.get_data()

# Se guarda la señal original para hacer los gráficos al final
Xorig = sig_orig_raw[0]
X = sig_raw[0]

# Se eliminan las SW que tienen duración menor que 500ms y mayor a 900ms
condition = np.logical_and(t_dur>0.5,t_dur<0.9)
tmp_dur = t_dur[condition]
tmp_idx = t_idx[condition]

#%%

########## PREPARACIÓN DEL TRAINING SET ##########

# Ondas lentas (HIT)
swaves = []
swaves_orig = []
k = 0
for i in tmp_idx:
    tmp_muestras = fs*(1-tmp_dur[k])
    if tmp_muestras+tmp_dur[k]*fs < fs:
        tmp_muestras = tmp_muestras + (tmp_dur[k]-tmp_muestras)
    swaves.append(X[int(np.round(i*200-(tmp_muestras/2))):int(np.round((i+tmp_dur[k])*200+(tmp_muestras/2)))])
    swaves_orig.append(Xorig[int(np.round(i*200-(tmp_muestras/2))):int(np.round((i+tmp_dur[k])*200+(tmp_muestras/2)))])
    k = k + 1

# Ruido (NO-HIT)
no_swaves = []
no_swaves_orig = []
np.random.seed(0)
pos = np.random.randint(len(X), size=len(swaves)*10)  # One-dimensional array
nowaves_count = len(swaves)
i = 0
while nowaves_count > 0:
    tmp_max = np.max(Xorig[pos[i]:pos[i]+200]*1e6)
    tmp_min = np.min(Xorig[pos[i]:pos[i]+200]*1e6)
    if (tmp_max - tmp_min) <= 75+15:
        no_swaves.append(X[pos[i]:pos[i]+200])
        no_swaves_orig.append(Xorig[pos[i]:pos[i]+200])
        nowaves_count -= 1
    i += 1

  
#%%

########## ESCALOGRAMAS ##########

def plot_scalogram(coefs,siglen,titulo="Título",fname=0):
    
    plt.figure(figsize=(12,4))
    plt.imshow(abs(coefs),extent=[0,siglen-1,120,0],
        interpolation='bilinear', cmap='jet', 
        aspect='auto',vmax=coefs.max(),vmin=coefs.min())
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(0,121,10))
    plt.xticks([0,50,100,150,199],[0,0.25,0.5,0.75,1])
    plt.title(f'CWT Scalogram {titulo}')
    plt.ylabel('Scales')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    if fname!=0:
        plt.savefig("D:\Disco de 500\Work\Doctorado\Repository\Python-san\Python-san\Load Data\Images"+"\\"+fname, bbox_inches="tight")
    plt.show()

#%%

########## TRANSFORMADA WAVELET ##########
# Fa = Fc/(a*T)
# Fa: Pseudofrecuencia
# Fc: Frecuencia central de la transformada wavelet (0.8125 Hz)
# a: Escala
# T: Periodo de muestreo

x_train = []
y_train = []
x_test = []
y_test = []

minmin = -0.001018770385891903
maxmax = 0.0009449624220248187

for i in range(len(swaves)):
    print(np.round((i*100)/len(swaves),decimals=1),"%")
    scales = np.arange(1,121,4)
    coef, freqs = pywt.cwt(no_swaves[i],scales,'morl',sampling_period=1/200)
    coef = ((coef - minmin) * (1/(maxmax - minmin) * 255)).astype('uint8')
    # coef = coef[0::40]

    if(i<100):
        x_train.append(coef)
        y_train.append(0)
    else:
        x_test.append(coef)
        y_test.append(0)

    scales = np.arange(1,121,4)
    coef, freqs = pywt.cwt(swaves[i],scales,'morl',sampling_period=1/200)
    coef = ((coef - minmin) * (1/(maxmax - minmin) * 255)).astype('uint8')
    # coef = coef[0::40]
    
    if(i<100):
        x_train.append(coef)
        y_train.append(1)
    else:
        x_test.append(coef)
        y_test.append(1)

#%%

########## PLOT DE SEÑALES TEMPORALES ##########

saveimage=False
n=len(swaves)
n = 1

for i in range(n):
    print(i)
    plt.figure(figsize=(10,5))
    ejeX = np.linspace(0,1,200)
    plt.plot(ejeX,swaves_orig[i]*1e6)
    plt.plot(ejeX,no_swaves_orig[i]*1e6)

    plt.axhline(y=35,color='r')
    plt.axhline(y=-35,color='r')
    
    plt.ylim(top=200)
    plt.ylim(bottom=-200)
    plt.title('Señales de Onda Lenta y Ruido')
    plt.ylabel('Amplitud [uV]')
    plt.xlabel('Tiempo [seg]')
    plt.legend(["HIT", "NO-HIT"], loc='upper right')
    plt.grid(True)
    if saveimage == True:
        plt.savefig("D:\Disco de 500\Work\Doctorado\Repository\Python-san\Python-san\Load Data\Images"+"\\"+f'Signals {i}', bbox_inches="tight")
    plt.show()

#%%

########## ADAPTACIÓN DE TRAINING SET PARA LA RED NEURONAL ##########

x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255.0
x_test = x_test/255.0
y_train_raw = np.array(y_train)
y_test_raw = np.array(y_test)

# GENERACIÓN DE ETIQUETAS

# one-hot encoding using keras' numpy-related utilities
n_classes = 2
# Generamos las nuevas variables que vamos a usar como etiquetas obejtivo
y_train_raw = np_utils.to_categorical(y_train_raw, n_classes)
y_test_raw = np_utils.to_categorical(y_test_raw, n_classes)

#%%

########## RED NEURONAL ##########

# CREACIÓN DEL MODELO Y ENTRENAMIENTO

# building a linear stack of layers with the sequential model
n_input_layer = x_train.shape[1]
model = Sequential()
# model.add(Dense(n_input_layer))
model.add(Dense(4096, input_shape=(8000,)))  
# model.add(Dense(4096, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='softmax'))
# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['categorical_accuracy'],
    optimizer='adam')
# training the model and saving metrics in history
history = model.fit(x_train, y_train_raw,
          batch_size=256, epochs=200,
          verbose=1,
          validation_data=(x_test, y_test_raw))

# %%

########## METRICAS ##########

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
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()
# %%

########## PREDICCIÓN Y REPORTE ##########

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

########## PLOT DE PREDICCIONES ##########

def predict_signal(offset, seconds,saveimage=False):
    
    Dw = 20 # muestras que se desplaza la ventana
    win_pred_tmp = []
    scales = np.arange(1,121,0.1)
    porcentaje = -1
    for k in range(offset,seconds+offset+1):
        porcentaje = porcentaje + 1
        print(np.round(porcentaje*100/seconds,decimals=1),'%')

        for j in range(0,int(200/Dw)):
            coef, freqs = pywt.cwt(X[k*200+j*Dw:(k+1)*200+j*Dw],scales,'morl',sampling_period=1/200)
            coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
            coef = coef[0::30]
            win_pred_tmp.append(coef)
 
    win_pred = np.array(win_pred_tmp)
    win_pred = win_pred.reshape(win_pred.shape[0], win_pred.shape[1]*win_pred.shape[2])
    win_pred = win_pred.astype('float32')
    win_pred = win_pred/255.0

    salida = model.predict(win_pred)
    Y_norm = salida
    Y_norm_tmp = Y_norm.argmax(axis=1)
    Y_norm = Y_norm.argmax(axis=1)

    N = len(range(offset*200,(seconds+offset+1)*200))
    Xticks = np.linspace(offset,(seconds+offset+1),N)
    plt.figure(figsize=(12,4))

    plt.grid(True)
    plt.plot(Xticks,Xorig[offset*200:(seconds+offset+1)*200]*1e6)

    idx = 0
    for k in range(offset,seconds+offset+1):
        for j in range(0,int(200/Dw)):
            
            if Y_norm[idx+j]==1:
                tmp_max = np.max(Xorig[k*200+j*Dw:(k+1)*200+j*Dw]*1e6)
                tmp_min = np.min(Xorig[k*200+j*Dw:(k+1)*200+j*Dw]*1e6)
                if (tmp_max - tmp_min) >= 75+15:
                    ejeX = np.linspace((k)*200+j*Dw,((k)+1)*200+j*Dw-1,200)/200
                    plt.plot(ejeX, Xorig[k*200+j*Dw:(k+1)*200+j*Dw]*1e6, 'r')
        idx = idx + j + 1


    plt.axhline(y=35,color='r')
    plt.axhline(y=-35,color='r')

    plt.title('Clasificación de Ondas Lentas')
    plt.ylabel('Amplitud [uV]')
    plt.xlabel('Tiempo [seg]')
    plt.legend(['Señal original', 'Ondas lentas'], loc='upper right')
    if saveimage == True:
        plt.savefig("D:\Disco de 500\Work\Doctorado\Repository\Python-san\Python-san\Load Data\Images"+"\\"+f'Swaves {offset}', bbox_inches="tight")
    plt.show()

# %%

########## PREDICCIÓN DE SEÑALES ##########

for i in range(3000,10000,30):
    predict_signal(i, 15)

# %%

########## PREDICCIÓN DE SEÑALES DEL DATASET ##########

for i in tmp_idx:
    print(i)
    predict_signal(int(i)-2, 5)
