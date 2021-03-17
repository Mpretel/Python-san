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
# from skimage import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#%%

raw = mne.io.read_raw_brainvision('D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02.vhdr', preload=True)
info = raw.info
Data, sfreq, chan, fs = raw._data,raw.info['sfreq'], raw.info['ch_names'], raw.info['sfreq']

raw.drop_channels(chan[2:len(chan)])
raw.drop_channels('C3_1')
sig_orig_raw = raw.get_data()
raw.filter(0.5,4,method='iir')

raw2,anot = set_sw_kc_annot(raw,'D:/Disco de 500/Work/Doctorado/Repository/Python-san/Python-san/Load Data/Data/ExpS68-S02_2020July23_23-25.txt')

t_idx = anot.onset
t_dur = anot.duration
sig_raw = raw.get_data()

Xorig = sig_orig_raw[0]
X = sig_raw[0]

condition = np.logical_and(t_dur>0.5,t_dur<0.9)
tmp_dur = t_dur[condition]
tmp_idx = t_idx[condition]

#%%

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

print("Buscando ruido")
##NO HIT
no_swaves = []
no_swaves_orig = []
np.random.seed(0)  # seed for reproducibility
pos = np.random.randint(len(X), size=len(swaves)*10)  # One-dimensional array
nowaves_count = len(swaves)
i = 0
while nowaves_count > 0:
    print(nowaves_count)
    # for i in range(len(pos)):
    tmp_max = np.max(Xorig[pos[i]:pos[i]+200]*1e6)
    tmp_min = np.min(Xorig[pos[i]:pos[i]+200]*1e6)
    if (tmp_max - tmp_min) <= 75+15:
        no_swaves.append(X[pos[i]:pos[i]+200])
        no_swaves_orig.append(Xorig[pos[i]:pos[i]+200])
        nowaves_count -= 1
    i += 1


    



# no_swaves = []
# tmp_fs = 200

# offset = 145
# for i in range(0,46):
#     no_swaves.append(X[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
#     no_swaves_orig.append(Xorig[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
# offset = 1660
# for i in range(0,20):
#      no_swaves.append(X[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
# offset = 335
# for i in range(0,12):
#     no_swaves.append(X[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
#     no_swaves_orig.append(Xorig[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
# offset = 15940
# for i in range(0,20):
#     no_swaves.append(X[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
#     no_swaves_orig.append(Xorig[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
# offset = 3195
# for i in range(0,22):
#     no_swaves.append(X[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
#     no_swaves_orig.append(Xorig[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
# offset = 370
# for i in range(0,9):
#     no_swaves.append(X[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])
#     no_swaves_orig.append(Xorig[ tmp_fs*(i + offset) : tmp_fs*((i+1) + offset)])

#%%
def plot_scalogram(coefs,siglen,titulo,fname=0):
    
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
    # plt.savefig("D:\Disco de 500\Work\Doctorado\Repository\Python-san\Python-san\Load Data\Images"+"\hola.png",bbox_inches="tight")
    plt.show()
    # plt.savefig(fig,format='png')
    # plt.imsave("/hola.jpg",fig)

#%%
x_train = []
y_train = []
x_test = []
y_test = []

minmin = -0.001018770385891903
maxmax = 0.0009449624220248187

# TRANSFORMADA WAVELET
for i in range(len(swaves)):

    print("NO-HIT ",i)
    scales = np.arange(1,121,0.1)
    coef, freqs = pywt.cwt(no_swaves[i],scales,'morl',sampling_period=1/200)
    # coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    # print(i,": Min: ",np.min(coef)," Max: ",np.max(coef)," Dif: ",np.max(coef)-np.min(coef))
    coef = ((coef - minmin) * (1/(maxmax - minmin) * 255)).astype('uint8')
    # plot_scalogram(coef,coef.shape[1],"NO-HIT",f'No-Hit-{i}.png')
    # plot_scalogram(coef,coef.shape[1],"NO-HIT")
    
    
    # print(i,": Min: ",np.min(coef)," Max: ",np.max(coef)," Dif: ",np.max(coef)-np.min(coef))
    # print("********************************************")
    coef = coef[0::30]
    # plot_scalogram(coef,coef.shape[1])

    if(i<100):
        x_train.append(coef)
        y_train.append(0)
    else:
        x_test.append(coef)
        y_test.append(0)

    print("HIT ",i)
    scales = np.arange(1,121,0.1)
    coef, freqs = pywt.cwt(swaves[i],scales,'morl',sampling_period=1/200)
    # coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
    
    # plot_scalogram(coef,coef.shape[1],"HIT")
    # plot_scalogram(coef,coef.shape[1],"HIT",f'Hit-{i}.png')
    # plot_scalogram(coef,coef.shape[1],"HIT")
    coef = ((coef - minmin) * (1/(maxmax - minmin) * 255)).astype('uint8')
    # plot_scalogram(coef,coef.shape[1])
    coef = coef[0::30]
    # plot_scalogram(coef,coef.shape[1])
    # print(i,": Min: ",np.min(coef)," Max: ",np.max(coef))
    # print("********************************************")
    
    
    if(i<100):
        x_train.append(coef)
        y_train.append(1)
    else:
        x_test.append(coef)
        y_test.append(1)

#%%
# Cuantas dibujar
n=len(swaves)
# n = 5

for i in range(n):
    print(i)
    plt.figure(figsize=(10,5))
    ejeX = np.linspace(0,1,200)
    plt.plot(ejeX,swaves_orig[i]*1e6)
    plt.plot(ejeX,no_swaves_orig[i]*1e6)
    
    # Nsec = 1
    # YLine = np.ones(Nsec*200)*75
    # plt.plot(ejeX,YLine,'r')
    # plt.plot(ejeX,-YLine,'r')

    plt.axhline(y=35,color='r')
    plt.axhline(y=-35,color='r')
    
    plt.ylim(top=200)
    plt.ylim(bottom=-200)
    plt.title('Señales de Onda Lenta y Ruido')
    plt.ylabel('Amplitud [uV]')
    plt.xlabel('Tiempo [seg]')
    # plt.legend([f'Hit {i}', f'No Hit {i}'], loc='upper right')
    plt.legend(["HIT", "NO-HIT"], loc='upper right')
    plt.grid(True)
    plt.savefig("D:\Disco de 500\Work\Doctorado\Repository\Python-san\Python-san\Load Data\Images"+"\\"+f'Signals {i}', bbox_inches="tight")
    plt.show()

#%%
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
# one-hot encoding using keras' numpy-related utilities
n_classes = 2
print("Shape of y_train [before one-hot encoding]: ", y_test_raw.shape)
# Generamos las nuevas variables que vamos a usar como etiquetas obejtivo
y_train_raw = np_utils.to_categorical(y_train_raw, n_classes)
y_test_raw = np_utils.to_categorical(y_test_raw, n_classes)
print("Shape of Y_train [after one-hot encoding]: ", y_test_raw.shape)

#%%
# building a linear stack of layers with the sequential model
n_input_layer = x_train.shape[1]
model = Sequential()
model.add(Dense(n_input_layer))                        
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

#%%
plot_model(model)
# %%
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
# **************************************************
# **************************************************
# **************************************************
# PARA WAVELET
# seconds = 40#50#40#40#40#10
# offset = 7788#0#4690#6452#7715#1436
def predict_signal(offset, seconds,saveimage=False):
    # seconds = 30
    # offset = 1426

    Dw = 20 # muestras que se desplaza la ventana
    win_pred_tmp = []
    scales = np.arange(1,121,0.1)
    porcentaje = -1
    for k in range(offset,seconds+offset+1):
        porcentaje = porcentaje + 1
        print(porcentaje*100/seconds,'%')

        for j in range(0,int(200/Dw)):
            coef, freqs = pywt.cwt(X[k*200+j*Dw:(k+1)*200+j*Dw],scales,'morl',sampling_period=1/200)
            coef = ((coef - coef.min()) * (1/(coef.max() - coef.min()) * 255)).astype('uint8')
            # coef = ((coef - minmin) * (1/(maxmax - minmin) * 255)).astype('uint8')
            coef = coef[0::30]
            win_pred_tmp.append(coef)
            # plot_scalogram(coef,len(coef))
            # plt.plot(X[k*200+j*Dw:(k+1)*200+j*Dw])
            # plt.show()

    win_pred = np.array(win_pred_tmp)
    win_pred = win_pred.reshape(win_pred.shape[0], win_pred.shape[1]*win_pred.shape[2])
    win_pred = win_pred.astype('float32')
    win_pred = win_pred/255.0


    # off = 0
    # for i in range(100):
    #     print(i)
    #     plot_scalogram(win_pred_tmp[i+off],len(win_pred_tmp[i+off]),"Titulo")



    salida = model.predict(win_pred)
    Y_norm = salida
    Y_norm_tmp = Y_norm.argmax(axis=1)
    Y_norm = Y_norm.argmax(axis=1)

    N = len(range(offset*200,(seconds+offset+1)*200))
    Xticks = np.linspace(offset,(seconds+offset+1),N)
    # plt.figure(figsize=(12,4))
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

    # return Y_norm


    # frequency = 2500  # Set Frequency To 2500 Hertz
    # duration = 100  # Set Duration To 1000 ms == 1 second
    # winsound.Beep(frequency, duration)

# %%
for i in range(3000,10000,30):
    print(i)
    t = time.time()
    predict_signal(i, 15)
    elapsed = time.time() - t
# %%
for i in tmp_idx:
    print(i)
    predict_signal(int(i)-2, 5)
# %%
# import winsound
# file = open('pepe.txt', 'w')
# file.write(str(coef[0]))
# file.close()

# %%

# Matriz de confusión

# import matplotlib.pyplot as pltimport scikitplot as skplt
#Normalized confusion matrix for the K-NN 
# modelprediction_labels = knn_classifier.predict(X_test)
# skplt.metrics.plot_confusion_matrix(y_test, prediction_labels, normalize=True)plt.show()

print(confusion_matrix(y_test, y_test_pred_NN.argmax(axis=1),normalize='true'))

# %%
import scikitplot as skplt 
skplt.metrics.plot_confusion_matrix(y_test, y_test_pred_NN.argmax(axis=1), normalize=True)
# %%
t = time.time()
# do stuff
elapsed = time.time() - t

# %%
t = time.time()
predict_signal(3000, 1)
elapsed = time.time() - t
print(elapsed)
# %%
