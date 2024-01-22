import toast.pipeline_tools
import numpy as np
import healpy as hp
from toast.tod import AnalyticNoise
from copy_FunctionsTMS import TODTMS
from toast.weather import Weather
import pickle
import pandas as pd
from astropy.time import Time, TimeDelta
from PyAstronomy import pyasl
import datetime
import astropy.units as u
#######_______________ definicion de clases_____________

def timedef(times_sec,f_sampling):
    t = Time(times_sec, format='fits', scale='utc')
    dt = t[1] - t[0] #me da el valor del tiempo en dias
    T_obs=dt*86400 # para pasar el valor de dias a segundos 1 dia tiene 86400 s
    t_sampling = 1.0 / f_sampling # time between samples (s)
    nsamp = (T_obs /t_sampling) # total number of samples (samples)
    total_time=np.linspace(0., int(T_obs.value),num=int(nsamp.value))
    dates_sec=t[0]+ dt * np.linspace(0., 1.,int(nsamp.value) )
    tconv = Time(dates_sec, format='fits')
    tconv.format = 'unix'
    return dates_sec,total_time,nsamp,tconv


def minoise(Ddetector,alpha,net,fknee):
    new_alpha=0
    new_net=0
    new_fknee=0
    new_fknee=Ddetector.detfknee.copy()
    new_net=Ddetector.detnet.copy()
    new_alpha=Ddetector.detalpha.copy()
    
    for key, value in new_alpha.items():
        new_alpha[key]*=alpha
        new_net[key]*=net
        new_fknee[key]*=fknee
        
        
    noise_ar1=AnalyticNoise(rate=Ddetector.detrate ,fmin=Ddetector.detfmin, detectors=Ddetector.detnames,
                       fknee=new_fknee , alpha=new_alpha ,NET=new_net)
    return noise_ar1  


def mifuncion(Ddetector,Dschedule,noise_ar,weather_atacama,focalplane,MAPA_SIM,Dsimulation,outdirD,outprefixD):
    #communication MPI
    data=0
    mpiworld, procs, rank = toast.mpi.get_world()
    comm = toast.mpi.Comm(mpiworld)
    obs = {}
    noisew=0
    det1=0
    obs["noise"] = noise_ar
    obs['weather']= weather_atacama
    obs['focalplane']= focalplane
    obs["site_id"] = 123
    obs["telescope_id"] = 1234
    obs["altitude"] = Dschedule.OT_ALTITUDE
    obs["fpradius"] = 3 #radius = 1.1558552389176189 deg
    obs["id"] = int(Dschedule.START_TIME * 10000)
    obs["tod"] = todgb = TODTMS(comm.comm_group,
                                Ddetector.detquat,
                                int(Dschedule.totsamples),
                                firsttime=Dschedule.START_TIME,
                                el=Dschedule.ELEVATION,
                                site_lon=Dschedule.OT_LON,
                                site_lat=Dschedule.OT_LAT,
                                site_alt=Dschedule.OT_ALTITUDE,
                                scanrate=Dschedule.SCANRATE,
                                rate=Dschedule.DET_RATE,
                                az_i=Dschedule.AZ_I,
                                az_f=Dschedule.AZ_F)
    data = toast.Data(comm) 
    data.obs.append(obs)
    print('TOD',datetime.datetime.now())
    #name = "signal_D"
    toast.tod.OpCacheClear("signal").exec(data)
    toast.todmap.OpPointingHpix(nside=Dsimulation.NSIDE, nest=True, mode="I").exec(data)
    print('OpPointingHpix',datetime.datetime.now())
  
    distmap = toast.map.DistPixels(
        data,
        nnz=Dsimulation.nnz,
        dtype=np.float32,)
    
    distmap.read_healpix_fits(MAPA_SIM)
    print('distmap',datetime.datetime.now())
    toast.todmap.OpSimScan(input_map=distmap, out="signal").exec(data)
    print('OpSimScan',datetime.datetime.now())
    # Copy the sky signal
    toast.tod.OpCacheCopy(input="signal", output="sky_signal", force=True).exec(data)
    # Simulate noise
    toast.tod.OpSimNoise(out="signal", realization=0).exec(data)
    print('OpSimNoise',datetime.datetime.now())
    toast.tod.OpCacheCopy(input="signal", output="full_signal", force=True).exec(data)
    
    

    mapmaker = toast.todmap.OpMapMaker(
        nside=Dsimulation.NSIDE,
        nnz=Dsimulation.nnz,
        name="signal",
        outdir=outdirD,
        outprefix=outprefixD,
        baseline_length=Dsimulation.baseline_length,
        iter_max=Dsimulation.iter_max,
        use_noise_prior=Dsimulation.use_noise_prior,)
    print('INICIA MAPMAKER',datetime.datetime.now())
    mapmaker.exec(data)
    print('FIN MAPMAKER',datetime.datetime.now())
    return noise_ar,data

class params_iniciales:
    latitude_degrees = 28 + 18*1/60 + 8*1/60/60
    longitude_degrees = -(17 + 29*1/60 + 16*1/60/60)
    elevation = 2390  # en metros
    #lugar = EarthLocation(lat=latitude_degrees*u.deg, lon=longitude_degrees*u.deg, height=elevation*u.m)

    times_sec1 = ['2023-06-01T06:00:00.1', '2023-06-01T06:00:20.1']
    f_sampling = 1  # sample rate (Hz)
    dates_se1,total_ti1,nsamp1,tconv=timedef(times_sec1,f_sampling)

    #elvfix=75.58
    #azi_fix=203 #deg
    #azf_fix=292 #deg
    elvfix=60
    azi_fix=70 #deg
    azf_fix=300 #deg
    Vaz=4 #deg/s
    frequ_obs=10 #Hz 
    
class Dschedule:
    START_TIME = params_iniciales.tconv[0].value #unix time
    DURATION = np.size(params_iniciales.total_ti1) #(duration in seconds)
    STOP_TIME = params_iniciales.tconv[np.size(params_iniciales.tconv)-1].value
    SAMPLE_RATE = params_iniciales.f_sampling #Hz
    OT_ALTITUDE = params_iniciales.elevation # m
    OT_LON = '{}'.format(params_iniciales.longitude_degrees)
    OT_LAT = '{}'.format(params_iniciales.latitude_degrees)
    totsamples = params_iniciales.nsamp1.value
    ELEVATION =params_iniciales.elvfix #deg
    SCANRATE = params_iniciales.Vaz #deg/s
    AZ_I = params_iniciales.azi_fix #deg
    AZ_F = params_iniciales.azf_fix #deg
    DET_RATE =params_iniciales.f_sampling #Hz 1000
    weather_atacama = Weather(fname='/scratch/aarriero/main_docs/secondyear/runarchivos/weather_Atacama.fits')       

class sensor_dat:
    epsilon=0
    fsample=params_iniciales.f_sampling
    alpha=1
    NET=7e-6
    fmin=0
    fknee=0
    fwhm_arcmin=120
        

#### para cambiar el archivo pkl
testdo=1
if testdo==1:
    
    sensor_data_Fake ={'0': {'quat': np.array([0.,         0.,         0, 1]),
                              'polangle_deg': 0.0,
    'epsilon': sensor_dat.epsilon,
    'fsample': sensor_dat.fsample,
    'alpha': sensor_dat.alpha, #
    'NET': sensor_dat.NET, #0.0008
    'fmin': sensor_dat.fmin,
    'fknee': sensor_dat.fknee, #0.004
    'fwhm_arcmin': sensor_dat.fwhm_arcmin},
                       '1': {'quat': np.array([0. ,         0.,         0.70710678, 0.70710678]),
                             'polangle_deg': 1.5707963267948966,
    'epsilon': sensor_dat.epsilon,
    'fsample': sensor_dat.fsample,
    'alpha': sensor_dat.alpha, #
    'NET': sensor_dat.NET, #0.0008
    'fmin': sensor_dat.fmin,
    'fknee': sensor_dat.fknee, #0.004
    'fwhm_arcmin': sensor_dat.fwhm_arcmin}
                      }
    with open('/scratch/aarriero/main_docs/secondyear/runarchivos/sensor_data_f.pkl', 'wb') as f:  # open a text file
        pickle.dump(sensor_data_Fake, f) # serialize the list
    f.close()

with open('/scratch/aarriero/main_docs/secondyear/runarchivos/sensor_data_f.pkl','rb') as f:
    data2=pickle.load(f)    

focalplane=data2 # un detector    
class Ddetector:
    detnames = list(sorted(data2.keys()))
    detquat = {x: data2[x]["quat"] for x in detnames}
    detfwhm = {x: data2[x]["fwhm_arcmin"] for x in detnames}#
    detalpha = {x: data2[x]["alpha"] for x in detnames}#
    detrate = {x: data2[x]["fsample"] for x in detnames}#
    detfmin = {x: data2[x]["fmin"] for x in detnames} #
    detfknee = {x: data2[x]["fknee"] for x in detnames}#
    detnet = {x: (data2[x]["NET"]) for x in detnames} # 
    detlabels = {x: x for x in detnames} #separete labels 
    detpolcol = {x: "red" if i % 2 == 0 else "blue" for i, 
                 x in enumerate(detnames)} #red or blue color to the image  
    
class Dsimulation:
    NSIDE=128 #1024
    nnz=1 #nnz (int): the number of values per pixel.
    baseline_length=3
    iter_max=100
    use_noise_prior=False
    npix = 12 * NSIDE ** 2
    fwhm=2    






    
