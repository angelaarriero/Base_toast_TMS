import numpy as np
from scipy.constants import degree
import healpy as hp
import datetime
import toast
try:
    import ephem
except:
    ephem = None
from toast import qarray as qa
from toast.timing import function_timer, Timer
from toast.tod import Interval, TOD
from toast.healpix import ang2vec
from toast.todmap.pointing_math import quat_equ2ecl, quat_equ2gal, quat_ecl2gal
from toast.todmap.sim_tod import simulate_hwp
from toast.weather import Weather


class TODTMS(toast.todmap.TODGround):
    '''GB TOD
    '''
    @function_timer
    def __init__(
        self,
        mpicomm,
        detectors,
        samples,
        firsttime,
        el,
        site_lon,
        site_lat,
        site_alt,
        scanrate,
        rate,
        az_i,
        az_f,
        boresight_angle=0,
        scan_accel=0.1,
        CES_start=None,
        CES_stop=None,
        sun_angle_min=90,
        sampsizes=None,
        sampbreaks=None,
        coord="C",
        report_timing=True,
        hwprpm=None,
        hwpstep=None,
        hwpsteptime=None,
        sinc_modulation=False,
        **kwargs
    ):
        if samples < 1:
            raise RuntimeError(
                "TODGround must be instantiated with a positive number of "
                "samples, not samples == {}".format(samples)
            )

        if ephem is None:
            raise RuntimeError("Cannot instantiate a TODGround object without pyephem.")

        if sampsizes is not None or sampbreaks is not None:
            raise RuntimeError(
                "TODGround will synthesize the sizes to match the subscans."
            )

        if CES_start is None:
            CES_start = firsttime
        elif firsttime < CES_start:
            raise RuntimeError(
                "TODGround: firsttime < CES_start: {} < {}"
                "".format(firsttime, CES_start)
            )

        lasttime = firsttime + samples / rate
        if CES_stop is None:
            CES_stop = lasttime
        elif lasttime > CES_stop:
            raise RuntimeError(
                "TODGround: lasttime > CES_stop: {} > {}" "".format(lasttime, CES_stop)
            )
        print('SCAN INICIO',datetime.datetime.now())

        self._firsttime = firsttime
        self._lasttime = lasttime
        self._rate = rate
        self._site_lon = site_lon
        self._site_lat = site_lat
        self._site_alt = site_alt
        
        self._min_az = az_i
        self._min_el = az_f

        if el < 1 or el > 89:
            raise RuntimeError("Impossible CES at {:.2f} degrees".format(el))

        self._boresight_angle = boresight_angle * degree
        self._el_ces = el * degree
        self._scanrate = scanrate * degree
        self._CES_start = CES_start
        self._CES_stop = CES_stop
        self._sun_angle_min = sun_angle_min
        if coord not in "CEG":
            raise RuntimeError("Unknown coordinate system: {}".format(coord))
        self._coord = coord
        self._report_timing = report_timing
        self._sinc_modulation = sinc_modulation

        self._observer = ephem.Observer()
        self._observer.lon = self._site_lon
        self._observer.lat = self._site_lat
        self._observer.elevation = self._site_alt  # In meters
        self._observer.epoch = ephem.J2000  # "2000"
        # self._observer.epoch = -9786 # EOD
        self._observer.compute_pressure()

        self._max_el = None
        self._scanrate = None

        self._az = None
        self._commonflags = None
        self._boresight_azel = None
        self._boresight = None
        print('SCAN tms inicio',datetime.datetime.now())
        sizes, starts = self.simulate_scanTMS(samples,el,scanrate,az_i,az_f)
        print('SCAN tms fin',datetime.datetime.now())
        self._fp = detectors
        self._detlist = sorted(list(self._fp.keys()))
        # call base class constructor to distribute data
        props = {
            "site_lon": site_lon,
            "site_lat": site_lat,
            "site_alt": site_alt,
            "el": el,
            "scanrate": scanrate,
            "scan_accel": scan_accel,
            "sun_angle_min": sun_angle_min,
        }
        super(toast.todmap.TODGround, self).__init__(
            mpicomm,
            self._detlist,
            samples,
            sampsizes=[samples],
            sampbreaks=None,
            meta=props,
            **kwargs        )
        self._AU = 149597870.7
        self._radperday = 0.01720209895
        self._radpersec = self._radperday / 86400.0
        self._radinc = self._radpersec / self._rate
        self._earthspeed = self._radpersec * self._AU
        self.translate_pointing()
        self.crop_vectors()
        print('SCAN FIN',datetime.datetime.now())
        return 

    @function_timer
    def simulate_scanTMS(self, samples,el,scanrate,az_i,az_f):
        self._times = self._CES_start + np.arange(samples) / self._rate
  
        AZF= []
        AZ= []
        res1= []
        w=0
        a=0
        mk=0
        w1=0
        test=0
        res=0
        fi=0
        contador=0
        time_count=int(np.size(self._times)/10)### /10
        #timezeros=np.zeros(np.size(self._times))
        timezeros=self._times
        for i in range(time_count):
            if contador<=np.size(self._times):
                
                for j in range(mk,np.size(timezeros)):
                    cons1=timezeros[mk]*scanrate
                    w1=az_i+(timezeros[j]*scanrate)-cons1
                    AZ.append(w1)
                    contador+=1
                    if AZ[j] > az_f:
                        a=j
                        break
                for k in range(a,np.size(timezeros)):
                    cons=timezeros[a]*scanrate
                    w=az_f-(timezeros[k]*scanrate)+cons
                    contador+=1
                    AZ.append(w)
                    if w <= az_i:
                        break
                    
            elif contador==np.size(self._times):
                break
        AZ= AZ[:(np.size(timezeros))]
        fi=np.array(AZ)
        self._az=(fi*np.pi)/180
        self._el = np.full(samples, self._el_ces) 
        self._min_el = (el*np.pi)/180
        self._max_el = (el*np.pi)/180
        self._commonflags = np.array([0]*samples, dtype=np.uint8)
        
        return [samples], [self._CES_start]