import numpy as np
import copy_ClasesTMS as cd
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import pickle


###################### Para calcular la temperatura de la atmosfera######################
nu, a, b, c = np.loadtxt("/scratch/aarriero/main_docs/secondyear/runarchivos/output_am_ElTeide_3.5mm_nu120.dat",unpack=True) 
ele=cd.params_iniciales.elvfix #degress
freq=cd.params_iniciales.frequ_obs #Hz
rad = np. deg2rad (ele)    
Temp_atmos=b/np.sin(rad)
Value_mmfreq=np.where(nu == freq)
T_atmos=b[Value_mmfreq]/np.sin(rad)
print(T_atmos)
###########################################################################################
########################## temperatura del cmb
T_cmb=2.72548
###########################################################################################
######################## Temperatura mapas background
planck_11 = hp.read_map("/scratch/aarriero/main_docs/secondyear/runarchivos/total_11GHz_0256.fits",field=None)

planck_10=planck_11*np.power((10/11),-3.1)

##########################################################################################
################# Temperatura instrumental



#########################################################################################

##################### SUMA TOTAL DE TODAS LAS TEMPERATURAS QUE AFECTAN EL MAPA DE ENTRADA ######################
Total_background=planck_11+T_cmb+T_atmos


##### aqui cambianos el NSIDE del mapa
NSIDE=128
Total_background_128=hp.ud_grade(Total_background,NSIDE)
hp.write_map("/scratch/aarriero/main_docs/secondyear/runarchivos/sim_map_d3.fits", 
                 hp.reorder(Total_background_128[0], r2n=True), nest=True, overwrite=True)
INPUT_MAP=hp.read_map('/scratch/aarriero/main_docs/secondyear/runarchivos/sim_map_d3.fits')
rot_mapdes=hp.Rotator(coord=['G','C']).rotate_map_pixel(INPUT_MAP)
hp.write_map("/scratch/aarriero/main_docs/secondyear/runarchivos/sim_map_D4mod.fits", 
                 hp.reorder(rot_mapdes, r2n=True), nest=True, overwrite=True)
MAPA_SIM1="/scratch/aarriero/main_docs/secondyear/runarchivos/sim_map_D4mod.fits"

######################################
# aqui estamos midiento la media del mapa final
lon = -55
lat = 28
vec = hp.ang2vec(lon, lat, lonlat=True)
nside = 128
large_disc = hp.query_disc(nside, vec, radius=np.radians(50))

bm = hp.read_map("/scratch/aarriero/main_docs/secondyear/runarchivos/sim_map_D4mod.fits")
bm[bm == 0] = hp.UNSEEN

h=bm[large_disc]

print(np.mean(h))#media del query

############################################