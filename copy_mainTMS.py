import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import pickle
from copy_FunctionsTMS import (TODTMS)
import copy_ClasesTMS as cd
import datetime
import Temperatures_input as Tin



######################## cambiar datos del ruido en el detector #########################
alpha=1
net=1e-7
fknee=0
N2=cd.minoise(cd.Ddetector,alpha,net,fknee)
###########################################################################################

weth_fil=cd.Dschedule.weather_atacama
outdir2="/scratch/aarriero/main_docs/secondyear/map_maker_test1"
outprefix2="toast_test_"

Noise2,data2=cd.mifuncion(cd.Ddetector,cd.Dschedule,N2,weth_fil,cd.focalplane,Tin.MAPA_SIM1,cd.Dsimulation,outdir2,outprefix2)

tod_er2 = data2.obs[0]["tod"]
dets = tod_er2.local_dets[::]
sky_signal_full2=[]
full_signal2=[]
ground_sig2=[]
atmosphere_signal2=[]
signal2=[]
for det in dets:
        sky_signal_full2.append(tod_er2.local_signal(det, "sky_signal"))
        full_signal2.append(tod_er2.local_signal(det, "full_signal"))
        #atmosphere_signal2.append(tod_er2.local_signal(det, "atmosphere"))#
        signal2.append(tod_er2.local_signal(det, "signal"))
times2 = np.array(tod_er2.local_times())
detss2= np.array(dets)
print('MAIN: GUARDADO DE DATOS NOISE',datetime.datetime.now())
with open('/scratch/aarriero/main_docs/secondyear/map_maker_test1/noise1.pickle', 'wb') as handle:
    pickle.dump(Noise2, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('MAIN: GUARDADO DE DATOS FULL SIGNAL',datetime.datetime.now())
with open('/scratch/aarriero/main_docs/secondyear/map_maker_test1/full.pickle', 'wb') as handle:
    pickle.dump(full_signal2, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('MAIN: GUARDADO DE DATOS SKY',datetime.datetime.now())
with open('/scratch/aarriero/main_docs/secondyear/map_maker_test1/sky.pickle', 'wb') as handle:
    pickle.dump(sky_signal_full2, handle, protocol=pickle.HIGHEST_PROTOCOL)    
print('MAIN: GUARDADO DE DATOS TIME',datetime.datetime.now())
with open('/scratch/aarriero/main_docs/secondyear/map_maker_test1/time.pickle', 'wb') as handle:
    pickle.dump(times2, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('MAIN: GUARDADO DE DATOS AZ',datetime.datetime.now())    
with open('/scratch/aarriero/main_docs/secondyear/map_maker_test1/az.pickle', 'wb') as handle:
    pickle.dump(data2.obs[0]['tod']._az, handle, protocol=pickle.HIGHEST_PROTOCOL)   
print('MAIN: GUARDADO DE DATOS EL',datetime.datetime.now())    
with open('/scratch/aarriero/main_docs/secondyear/map_maker_test1/el.pickle', 'wb') as handle:
    pickle.dump(data2.obs[0]['tod']._el, handle, protocol=pickle.HIGHEST_PROTOCOL)   
print('MAIN: GUARDADO DE DATOS META',datetime.datetime.now())    
with open('/scratch/aarriero/main_docs/secondyear/map_maker_test1/meta.pickle', 'wb') as handle:
    pickle.dump(data2.obs[0]['tod'].meta, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
