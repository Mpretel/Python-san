import os
import numpy as np
import mne
import matplotlib.pyplot as plt

#SOLO FUNCIONA CUANDO ESTAN ETIQUETADOS COMO: 'sw_I','sw_II' o'sw_dudosa'
def set_sw_annot(raw,path):
    """ Read hypno(txt file) and raw (vhdr-fif file) and Return (start, duration, description, end) """
    """ Return hypnogram plus scoring """
    hfiletxt = open(path, 'r')
    file = hfiletxt
    start=[]
    duration=[]
    end=[]
    description=[]
    for line in file:
        row=line.strip().split(',') 
        if len(row) ==3:
            if row[2]=='sw_I' or row[2]=='sw_II' or row[2]=='sw_dudosa'or row[2]==None:
                aux=row[0]+row[1]           
                start.append(row[0])
                duration.append(row[1])
                description.append(row[2])
                end.append(aux)
    start=np.asarray(start)
    start= start.astype(np.float)
    duration=np.asarray(duration)
    duration= duration.astype(np.float)
    description=np.asarray(description)

    my_annot = mne.Annotations(start,duration,description, orig_time=raw.annotations.orig_time)    
    reraw = raw.copy().set_annotations(my_annot)
    return reraw, my_annot



#SOLO FUNCIONA CUANDO ESTAN ETIQUETADOS COMO: 'sw','kc' o 'ambigua'
def set_sw_kc_annot(raw,path):
    """ Read hypno(txt file) and raw (vhdr-fif file) and Return (start, duration, description, end) """
    """ Return hypnogram plus scoring """
    hfiletxt = open(path, 'r')
    file = hfiletxt
    start=[]
    duration=[]
    end=[]
    description=[]
    for line in file:
        row=line.strip().split(',') 
        if len(row) ==3:
            if row[2]=='sw' or row[2]=='kc' or row[2]=='ambiguo'or row[2]==None:
                aux=row[0]+row[1]           
                start.append(row[0])
                duration.append(row[1])
                description.append(row[2])
                end.append(aux)
    start=np.asarray(start)
    start= start.astype(np.float)
    duration=np.asarray(duration)
    duration= duration.astype(np.float)
    description=np.asarray(description)

    my_annot = mne.Annotations(start,duration,description, orig_time=raw.annotations.orig_time)    
    reraw = raw.copy().set_annotations(my_annot)
    return reraw, my_annot
