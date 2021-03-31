#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
    ***
    MakePSF
    ***
"""
__author__ = 'Alan Loh, Julien Girard'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh','Julien Girard']
__maintainer__ = 'Julien'
__email__ = 'julien.girard@cea.fr'
__status__ = 'Production'
__all__ = [
    'MakePSF'
]

import logging
log = logging.getLogger(__name__)

import numpy as np
import numexpr as ne
import MeerKAT

import astropy.units as u
from astropy.time import Time, TimeDelta

from astropy.coordinates import (
    EarthLocation,
    Angle,
    AltAz,
    ICRS,
    Longitude,
    FK5,
    SkyCoord
)
from astropy.constants import c as lspeed

MeerKATarr=MeerKAT.MeerKATarray()


# ============================================================= #
# ---------------------------- lst ---------------------------- #
# ============================================================= #

def lst(time, kind):
    """ Local sidereal time
        :param time:
            Time
        :type time: :class:`~astropy.time.Time`
        :param kind:
            ``'fast'`` computes an approximation of local sidereal
            time, ``'mean'`` accounts for precession and ``'apparent'``
            accounts for precession and nutation.
        :type kind: str
        :returns: LST time
        :rtype: :class:`~astropy.coordinates.Longitude`
    """
    if kind.lower() == 'fast':
        # http://www.roma1.infn.it/~frasca/snag/GeneralRules.pdf
        # Number of days since 2000 January 1, 12h UT
        nDays = time.jd - 2451545.
        # Greenwich mean sidereal time
        gmst = 18.697374558 + 24.06570982441908 * nDays
        gmst %= 24.
        # Local Sidereal Time
        lst = gmst + MeerKATarr.Loc.lon.hour
        if np.isscalar(lst):
            if lst < 0:
                lst += 24
        else:
            lst[lst < 0] += 24.   
        return Longitude(lst, 'hour')
    else:
        location = MeerKATarr.Loc
        lon = location.to_geodetic().lon
        lst = time.sidereal_time(kind, lon)
        return lst

# ============================================================= #
# ---------------------------- lha ---------------------------- #
# ============================================================= #
def lha(lst, skycoord):
    """ Local Hour Angle of an object in the observer's sky
        
        :param lst:
            Local Sidereal Time, such as returned by
            :func:`~nenupy.astro.astro.lst` for instance.
        :type lst: :class:`~astropy.coordinates.Longitude`
        :param skycoord:
            Sky coordinates to convert to Local Hour Angles. This
            must be converted to FK5 coordinates with the
            corresponding equinox in order to give accurate
            results (see :func:`~nenupy.astro.astro.toFK5`).
        :type skycoord: :class:`~astropy.coordinates.SkyCoord`
        :returns: LHA time
        :rtype: :class:`~astropy.coordinates.Angle`
    """
    if skycoord.equinox is None:
        log.warning(
            (
                'Given skycoord for LHA computation does not '
                'have an equinox attribute, make sure the '
                'precession is taken into account.'
            )
        )
    ha = lst - skycoord.ra
    twopi = Angle(360.000000 * u.deg)
    if ha.isscalar:
        if ha.deg < 0:
            ha += twopi
        elif ha.deg > 360:
            ha -= twopi
    else:
        ha[ha.deg < 0] += twopi
        ha[ha.deg > 360] -= twopi
    return ha

# ============================================================= #


# ============================================================= #
# ------------------------ wavelength ------------------------- #
# ============================================================= #
def wavelength(freq):
    """ Convert radio frequency in wavelength.

        :param freq:
            Frequency (assumed in MHz unless a
            :class:`~astropy.units.Quantity` is provided)
        :type freq: `float`, :class:`~numpy.ndarray` or
            :class:`~astropy.units.Quantity`

        :returns: Wavelength in meters
        :rtype: :class:`~astropy.units.Quantity`
    """
    if not isinstance(freq, u.Quantity):
        freq *= u.MHz
    freq = freq.to(u.Hz)
    wavel = lspeed / freq
    return wavel.to(u.m)



# ============================================================= #
# ------------------------- ho_coord -------------------------- #
# ============================================================= #
def ho_coord(alt, az, time):
    """ Horizontal coordinates
    
        :param alt:
            Altitude in degrees
        :type alt: `float` or :class:`~astropy.units.Quantity`
        :param az:
            Azimuth in degrees
        :type az: `float` or :class:`~astropy.units.Quantity`
        :param time:
            Time at which the local zenith coordinates should be 
            computed. It can either be provided as an 
            :class:`~astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: str, :class:`~astropy.time.Time`
        :returns: :class:`~astropy.coordinates.AltAz` object
        :rtype: :class:`~astropy.coordinates.AltAz`
        :Example:
            >>> from nenupysim.astro import ho_coord
            >>> altaz = ho_coord(
                    alt=45,
                    az=180,
                    time='2020-01-01 12:00:00'
                )
    """
    if not isinstance(az, u.Quantity):
        az *= u.deg
    if not isinstance(alt, u.Quantity):
        alt *= u.deg
    if not isinstance(time, Time):
        time = Time(time)
    return AltAz(
        az=az,
        alt=alt,
        location=MeerKATarr.Loc,
        obstime=time
    )

def ho_zenith(time):
    """ Horizontal coordinates of local zenith above the array
        :param time:
            Time at which the local zenith coordinates should be 
            computed. It can either be provided as an 
            :class:`~astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: `str`, :class:`~astropy.time.Time`
        :returns: :class:`~astropy.coordinates.AltAz` object
        :rtype: :class:`~astropy.coordinates.AltAz`
        :Example:
            >>> zen_altaz = ho_zenith(time='2020-01-01 12:00:00')
    """
    if not isinstance(time, Time):
        time = Time(time)
    if time.isscalar:
        return ho_coord(
            az=0.,
            alt=90.,
            time=time
        )
    else:
        return ho_coord(
            az=np.zeros(time.size),
            alt=np.ones(time.size) * 90.,
            time=time
        )

# ============================================================= #
# ------------------------- eq_zenith ------------------------- #
# ============================================================= #
def eq_zenith(time):
    """ Equatorial coordinates of local zenith above the array
        
        :param time:
            Time at which the local zenith coordinates should be 
            computed. It can either be provided as an 
            :class:`~astropy.time.Time` object or a string in ISO
            or ISOT format.
        :type time: `str`, :class:`~astropy.time.Time`
        
        :returns: :class:`~astropy.coordinates.ICRS` object
        :rtype: :class:`~astropy.coordinates.ICRS`
        :Example:
            >>> zen_radec = eq_zenith(time='2020-01-01 12:00:00')
    """
    altaz_zenith = ho_zenith(
        time=time
    )
    return to_radec(altaz_zenith)
    
def to_radec(altaz):
    """ Transform altaz coordinates to ICRS equatorial system
        
        :param altaz:
            Horizontal coordinates
        :type altaz: :class:`~astropy.coordinates.AltAz`
        :returns: :class:`~astropy.coordinates.ICRS` object
        :rtype: :class:`~astropy.coordinates.ICRS`
        :Example:
            >>> from nenupysim.astro import eq_coord
            >>> radec = eq_coord(
                    ra=51,
                    dec=39,
                )
    """
    if not isinstance(altaz, AltAz):
        raise TypeError(
            'AltAz object expected.'
        )
    return altaz.transform_to(ICRS)

def toFK5(skycoord, time):
    """ Converts sky coordinates ``skycoord`` to FK5 system with
        equinox given by ``time``.
        :param skycoord:
            Sky Coordinates to be converted to FK5 system.
        :type skycoord: :class:`~astropy.coordinates.SkyCoord`
        :param time:
            Time that defines the equinox to be accounted for.
        :type time: :class:`~astropy.time.Time`
        :returns: FK5 sky coordinates
        :rtype: :class:`~astropy.coordinates.SkyCoord`
    """
    return skycoord.transform_to(
        FK5(equinox=time)
    )

# ============================================================= #
# ------------------- Multiprocessing Image ------------------- #
# ============================================================= #
def _init(arr_to_populate1, arr_to_populate2, lg, mg, u, v):
    """ Each pool process calls this initializer. Load the array to be populated into that process's global namespace """
    global arr1
    global arr2
    global larr
    global marr
    global uarr
    global varr
    arr1 = arr_to_populate1
    arr2 = arr_to_populate2
    larr = lg
    marr = mg
    uarr = u
    varr = v

def fill_per_block(args):
    x0, x1, y0, y1 = args.astype(int)
    tmp_r = np.ctypeslib.as_array(arr1)
    tmp_i = np.ctypeslib.as_array(arr2)
    na = np.newaxis

    lg = larr[na, x0:x1, y0:y1]
    mg = marr[na, x0:x1, y0:y1]
    pi = np.pi
    expo = ne.evaluate('exp(2j*pi*(uarr*lg+varr*mg))')

    tmp_r[:, x0:x1, y0:y1] = expo.real
    tmp_i[:, x0:x1, y0:y1] = expo.imag

def mp_expo(npix, ncpus, lg, mg, u, v):
    # print('inside mp_expo 1')
    block_size = int(npix/np.sqrt(ncpus))
    result_r = np.ctypeslib.as_ctypes(
        np.zeros((u.shape[0], npix, npix))
    )
    result_i = np.ctypeslib.as_ctypes(
        np.zeros_like(result_r)
    )
    shared_array_r = sharedctypes.RawArray(
        result_r._type_,
        result_r
    )
    shared_array_i = sharedctypes.RawArray(
        result_i._type_,
        result_i
    )
    # print('inside mp_expo 2')
    n_windows = int(np.sqrt(ncpus))
    block_idxs = np.array([
        (i, i+1, j, j+1) for i in range(n_windows) for j in range(n_windows)
    ])*block_size
    # pool = Pool(ncpus)
    # res = pool.map(
    #     fill_per_block,
    #     (block_idxs, shared_array_r, shared_array_i, block_size, lg, mg)
    # )
    # print('inside mp_expo 3')
    pool = Pool(
        processes=ncpus,
        initializer=_init,
        initargs=(shared_array_r, shared_array_i, lg, mg, u, v)
    )
    # print('inside mp_expo 4')
    res = pool.map(fill_per_block, (block_idxs))
    pool.close()
    # print('inside mp_expo 5')
    result_r = np.ctypeslib.as_array(shared_array_r)
    result_i = np.ctypeslib.as_array(shared_array_i)
    del shared_array_r, shared_array_i
    return result_r + 1j * result_i
# ============================================================= #


# ============================================================= #
# ---------------------------- UVW ---------------------------- #
# ============================================================= #
class UVW(object):
    """
    """

    def __init__(self, radioarray, freqs=None):
        self.bsl_xyz = None
        #self.times = times

        self.freqs = freqs
        # Meaning ?
        self._radioarray = radioarray

        # Question:  what is this
        # Meaning ?
        self.antpos = self._radioarray.T #np.array([a.tolist() for a in self._radioarray])

        # RGF93 to ITRF97
        # See http://epsg.io/?q=itrf97 to find correct EPSG
        #t = Transformer.from_crs(
        #    crs_from='EPSG:2154', # RGF93
        #    crs_to='EPSG:4896'# ITRF2005
        #)
        #antpos[:, 0], antpos[:, 1], antpos[:, 2] = t.transform(
        #    xx=antpos[:, 0],
        #    yy=antpos[:, 1],
        #    zz=antpos[:, 2]
        #)
        m=self.antpos.shape[0]
        # print(m)
        xyz = self.antpos[..., None]
        xyz = xyz[:, :, 0][:, None]
        # xyz = xyz - xyz.transpose(1, 0, 2)
        xyz = xyz.transpose(1, 0, 2) - xyz
        # self.bsl = xyz[np.triu_indices(m.size)]
        # Question:  what is this
        # Meaning ?
        self.bsl = xyz[np.tril_indices(m)]
        self._ants = m
        return


    @property
    def freqs(self):
        return self._freqs
    @freqs.setter
    def freqs(self, f):
        if f is None:
            self._freqs = None
        else:
            if not isinstance(f, u.Quantity):
                f *= u.MHz
            if f.isscalar:
                f = np.array([f.value]) * u.MHz
            self._freqs = f
        return


    @property
    def uvw(self):
        """ UVW in meters.
            :getter: (times, baselines, UVW)
            
            :type: :class:`~numpy.ndarray`
        """
        if not hasattr(self, '_uvw'):
            raise Exception(
                'Run .compute() first.'
            )
        return self._uvw


    @property
    def uvw_wave(self):
        """ UVW in lambdas.
            :getter: (times, freqs, baselines, UVW)
            
            :type: :class:`~numpy.ndarray`
        """
        if not hasattr(self, '_uvw'):
            raise Exception(
                'Run .compute() first.'
            )
        if self.freqs is None:
            raise ValueError(
                'No frequency input, fill self.freqs.'
            )
        lamb = wavelength(self.freqs).value
        na = np.newaxis
        return self._uvw[:, na, :, :]/lamb[na, :, na, na]


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def compute(self,ha,dec):
        r""" Compute the UVW at a given ``phase_center`` for all
            the :attr:`~nenupy.crosslet.uvw.UVW.times` and baselines
            formed by :attr:`~nenupy.crosslet.uvw.UVW.mas`.
            :param phase_center: Observation phase center. If
                ``None``, local zenith is considered as phase
                center for all :attr:`~nenupy.crosslet.uvw.UVW.times`.
            :type phase_center: :class:`~astropy.coordinates.SkyCoord`
            UVW are computed such as:
            .. math::
                \pmatrix{
                    u \\
                    v \\
                    w
                } =
                \pmatrix{
                    \sin(h) & \cos(h) & 0\\
                    -\sin(\delta) \cos(h) & \sin(\delta) \sin(h) & \cos(\delta)\\
                    \cos(\delta)\cos(h) & -\cos(\delta) \sin(h) & \sin(\delta)
                }
                \pmatrix{
                    \Delta x\\
                    \Delta y\\
                    \Delta z
                }
            :math:`u`, :math:`v`, :math:`w` are in meters. :math:`h`
            is the hour angle (see :func:`~nenupy.astro.astro.lha`)
            at which the phase center is observed, :math:`\delta`
            is the phase center's declination, :math:`(\Delta x,
            \Delta y, \Delta z)` are the baselines projections
            with the convention of :math:`x` to the South, :math:`y`
            to the East and :math:`z` to :math:`\delta = 90` deg.
            Result of the computation are stored as a :class:`~numpy.ndarray`
            in :attr:`~nenupy.crosslet.uvw.UVW.uvw` whose shape is
            (times, cross-correlations, 3), 3 being :math:`(u, v, w)`.
            """

        # Transformations
        self._uvw = np.zeros(
            (
                ha.size,
                self.bsl.shape[0],
                3
            )
        )
        # print('RAS', self._uvw.shape)
        # print('RAS', self.bsl.shape)

        xyz = np.array(self.bsl).T
        # rot = np.radians(-90) # x to the south, y to the east
        # rotation = np.array(
        #     [
        #         [ np.cos(rot), np.sin(rot), 0],
        #         [-np.sin(rot), np.cos(rot), 0],
        #         [ 0,           0,           1]
        #     ]
        # )
        self._ha=ha
        if self._ha.size ==1:
            sr = np.sin(ha)
            cr = np.cos(ha)
            sd = np.sin(dec)
            cd = np.cos(dec)
            rot_uvw = np.array([
                [    sr,     cr,  0],
                [-sd*cr,  sd*sr, cd],
                [ cd*cr, -cd*sr, sd]
            ])
            self._uvw[0, ...] = - np.dot(rot_uvw, xyz).T
        else:
            for i in range(self._ha.size):
                sr = np.sin(ha[i])
                cr = np.cos(ha[i])
                sd = np.sin(dec)
                cd = np.cos(dec)
                rot_uvw = np.array([
                [    sr,     cr,  0],
                [-sd*cr,  sd*sr, cd],
                [ cd*cr, -cd*sr, sd]
            ])
            # self.uvw[i, ...] = - np.dot(
            #     np.dot(rot_uvw, xyz).T,
            #     rotation
            # )
                self._uvw[i, ...] = - np.dot(rot_uvw, xyz).T

                
        return

    def compute2(self, phase_center=None):
        r""" Compute the UVW at a given ``phase_center`` for all
            the :attr:`~nenupy.crosslet.uvw.UVW.times` and baselines
            formed by :attr:`~nenupy.crosslet.uvw.UVW.mas`.
            :param phase_center: Observation phase center. If
                ``None``, local zenith is considered as phase
                center for all :attr:`~nenupy.crosslet.uvw.UVW.times`.
            :type phase_center: :class:`~astropy.coordinates.SkyCoord`
            UVW are computed such as:
            .. math::
                \pmatrix{
                    u \\
                    v \\
                    w
                } =
                \pmatrix{
                    \sin(h) & \cos(h) & 0\\
                    -\sin(\delta) \cos(h) & \sin(\delta) \sin(h) & \cos(\delta)\\
                    \cos(\delta)\cos(h) & -\cos(\delta) \sin(h) & \sin(\delta)
                }
                \pmatrix{
                    \Delta x\\
                    \Delta y\\
                    \Delta z
                }
            :math:`u`, :math:`v`, :math:`w` are in meters. :math:`h`
            is the hour angle (see :func:`~nenupy.astro.astro.lha`)
            at which the phase center is observed, :math:`\delta`
            is the phase center's declination, :math:`(\Delta x,
            \Delta y, \Delta z)` are the baselines projections
            with the convention of :math:`x` to the South, :math:`y`
            to the East and :math:`z` to :math:`\delta = 90` deg.
            Result of the computation are stored as a :class:`~numpy.ndarray`
            in :attr:`~nenupy.crosslet.uvw.UVW.uvw` whose shape is
            (times, cross-correlations, 3), 3 being :math:`(u, v, w)`.
            """
        # Phase center
        if phase_center is None:
            print('UVW phase centered at local zenith.')
            
            phase_center = eq_zenith(self.times)
        else:
            if not isinstance(phase_center, SkyCoord):
                raise TypeError(
                    'phase_center should be a SkyCoord object'
                )
            if phase_center.isscalar:
                ones = np.ones(self.times.size)
                ra_tab = ones * phase_center.ra
                dec_tab = ones * phase_center.dec
                phase_center = SkyCoord(ra_tab, dec_tab)
            else:
                if phase_center.size != self.times.size:
                    raise ValueError(
                        'Size of phase_center != times'
                    )
            print('UVW phase centered at RA={}, Dec={}'.format(
                    phase_center.ra[0].deg,
                    phase_center.dec[0].deg
                )
            )
        # Hour angles
        lstTime = lst(
            time=self.times,
            kind='apparent'
        )
        phase_center = toFK5(
            skycoord=phase_center,
            time=self.times
        )
        ha = lha(
            lst=lstTime,
            skycoord=phase_center
        )

        # Transformations
        self._uvw = np.zeros(
            (
                self.times.size,
                self.bsl.shape[0],
                3
            )
        )
        # print('RAS', self._uvw.shape)
        # print('RAS', self.bsl.shape)

        xyz = np.array(self.bsl).T
        # rot = np.radians(-90) # x to the south, y to the east
        # rotation = np.array(
        #     [
        #         [ np.cos(rot), np.sin(rot), 0],
        #         [-np.sin(rot), np.cos(rot), 0],
        #         [ 0,           0,           1]
        #     ]
        # )
        self._ha=ha
        for i in range(self.times.size):
            sr = np.sin(ha[i].rad)
            cr = np.cos(ha[i].rad)
            sd = np.sin(phase_center.dec[i].rad)
            cd = np.cos(phase_center.dec[i].rad)
            rot_uvw = np.array([
                [    sr,     cr,  0],
                [-sd*cr,  sd*sr, cd],
                [ cd*cr, -cd*sr, sd]
            ])
            # self.uvw[i, ...] = - np.dot(
            #     np.dot(rot_uvw, xyz).T,
            #     rotation
            # )
            self._uvw[i, ...] = - np.dot(rot_uvw, xyz).T
        return
    

# ============================================================= #

class Imager(UVW):
    """
    """

    def __init__(self, crosslets, fov=60, tidx=None, ncpus=4):
        self.skymodel = None
        self.srcnames = None
        # Meaning ?
        self.fov = fov
        # Meaning ?
        self.ncpus = ncpus
        # Meaning ?
        self._uvw=crosslets

        # self.cross = crosslets
        # self.vis = self.cross.reshape(
        #     tidx=tidx,
        #     fmean=False,
        #     tmean=False
        # )

        # start = self.cross.time[0]
        # stop = self.cross.time[-1]
        # self.phase_center = eq_zenith(
        #         time=start + (stop - start)/2,
        #     )
        
        # Initialize the UVW Class
        # super().__init__(
        #     array=NenuFAR(
        #         miniarrays=self.cross.meta['ma']
        #     ),
        #     freq=self.cross.meta['freq']
        # )
        # Compute the UVW coordinates at the zenith
        # self.compute(
        #     time=self.cross.time if tidx is None else self.cross.time[tidx],
        #     ra=None,
        #     dec=None,
        # )


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #

    @property
    def fov(self):
        return self._fov
    @fov.setter
    def fov(self, f):
        #lmmax = np.cos(np.radians(f))
        lmmax = np.cos(np.radians(90 - f/2))
        self.lmax = lmmax
        self.mmax = lmmax
        self._fov = f
        return


    @property
    def ncpus(self):
        return self._ncpus
    @ncpus.setter
    def ncpus(self, n):
        if not np.sqrt(n).is_integer():
            raise ValueError(
                'Number of CPUs must be a sqrtable'
            )
  #      if not ( n <= mp.cpu_count()):
  #          raise Exception(
  #              'Number of CPUs should be <={}'.format(n)
  #          )
        self._ncpus = n
        return


    @property
    def npix(self):
        return self._npix
    @npix.setter
    def npix(self, n):
        if not np.log2(n).is_integer():
            raise ValueError(
                'Number of pixels must be a power of 2'
            )
        self._npix = n
        return


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #


    def make_psf(self, npix=None, freq=None):
        """ Make the PSF regarding the UV distribution
            :param npix:
                Size in pixels of the image
            :type npix: int, optional
            :param freq:
                Frequency
            :type freqi: Freq in MHz
        """

        self.freq=freq

        if npix is None:
            npix = self.npix * 2

        uvw=self._uvw

        # Transform UVW in lambdas units, take frequency

        uvw = uvw[...] / wavelength(self.freq)

        # Prepare image parameters

        max_uv=np.max(np.abs(uvw[...,0:2])) # max uv to compute delta_u delta_v

        cell_size_l = cell_size_m = np.rad2deg((1 / (2 * max_uv.value))) # cell size on the sky depends on max uv = a.k.a. angular resolution (IN DEGREES)

        Nx = npix//2 #np.max([int(np.round(self.fov / cell_size_l)),npix//2]) # guessing the number of pixels from fov / resolution # FOV IN DEGREES
        Ny = npix//2 #np.max([int(np.round(self.fov / cell_size_m)),npix//2])
        
        uvwscaled=np.copy(uvw[...,0:2])
        uvwscaled[...,0]*=np.deg2rad(cell_size_l*Nx) # scaling the uv values to match
        uvwscaled[...,1]*=np.deg2rad(cell_size_m*Ny)

        uvw2=uvwscaled.reshape(-1, uvwscaled.shape[-1]) # ravel (Ntimes,Nbl,3) to (NtimesxNbl,3)
        
        print("(Time steps, baselines, 3)= (%d,%d,3) "%(uvw.shape[0],uvw.shape[1]))
        print("Total of %d visibilities"%(uvw2.shape[0]))
        tabulated_filter = AA_filter(5,63,"gaussian_sinc") # convolution kernel for convolutional gridding
        psf,samplingfunc = grid_ifft2(uvw2, Nx, Ny, tabulated_filter) # do the gridding (slow python)
        
        self.samplingfunc=samplingfunc
        self.psf = psf / psf.max()

        return self.psf,self.samplingfunc


class AA_filter:
    """
    Anti-Aliasing filter
    
    Keyword arguments for __init__:
    filter_half_support --- Half support (N) of the filter; the filter has a full support of N*2 + 1 taps
    filter_oversampling_factor --- Number of spaces in-between grid-steps (improves gridding/degridding accuracy)
    filter_type --- box (nearest-neighbour), sinc or gaussian_sinc
    """
    half_sup = 0
    oversample = 0
    full_sup_wo_padding = 0
    full_sup = 0
    no_taps = 0
    filter_taps = None
    def __init__(self, filter_half_support, filter_oversampling_factor, filter_type):
        self.half_sup = filter_half_support
        self.oversample = filter_oversampling_factor
        self.full_sup_wo_padding = (filter_half_support * 2 + 1)
        self.full_sup = self.full_sup_wo_padding + 2 #+ padding
        self.no_taps = self.full_sup + (self.full_sup - 1) * (filter_oversampling_factor - 1)
        taps = np.arange(self.no_taps)/float(filter_oversampling_factor) - self.full_sup / 2
        if filter_type == "box":
            self.filter_taps = np.where((taps >= -0.5) & (taps <= 0.5),
                                        np.ones([len(taps)]),np.zeros([len(taps)]))
        elif filter_type == "sinc":
            self.filter_taps = np.sinc(taps)
        elif filter_type == "gaussian_sinc":
            alpha_1=1.55
            alpha_2=2.52
            self.filter_taps = np.sin(np.pi/alpha_1*(taps+0.00000000001))/(np.pi*(taps+0.00000000001))*np.exp(-(taps/alpha_2)**2)
        else:
            raise ValueError("Expected one of 'box','sinc' or 'gaussian_sinc'")


def grid_ifft(vis, uvw, ref_lda, Nx, Ny, convolution_filter):
    """
    Convolutional gridder (continuum)

    Keyword arguments:
    vis --- Visibilities as sampled by the interferometer
    uvw --- interferometer's scaled uvw coordinates
            (Prerequisite: these uv points are already scaled by the similarity
            theorem, such that -N_x*Cell_l*0.5 <= theta_l <= N_x*Cell_l*0.5 and
            -N_y*Cell_m*0.5 <= theta_m <= N_y*Cell_m*0.5)
    ref_lda --- array of reference lambdas (size of vis channels)
    Nx,Ny --- size of image in pixels
    convolution_filter --- pre-instantiated AA_filter anti-aliasing
                           filter object
    """
    assert vis.shape[1] == ref_lda.shape[0], (vis.shape[1], ref_lda.shape[0])
    filter_index = \
        np.arange(-convolution_filter.half_sup,convolution_filter.half_sup+1)
    # one grid for the resampled visibilities per correlation:
    measurement_regular = \
        np.zeros([vis.shape[2],Ny,Nx],dtype=np.complex)
    # for deconvolution the PSF should be 2x size of the image (see 
    # Hogbom CLEAN for details), one grid for the sampling function:
    sampling_regular = \
        np.zeros([2*Ny,2*Nx],dtype=np.complex)
    for r in np.xrange(uvw.shape[0]):
        for c in np.xrange(vis.shape[1]):
            scaled_uv = uvw[r,:] / ref_lda[c]
            disc_u = int(np.round(scaled_uv[0]))
            disc_v = int(np.round(scaled_uv[1]))
            frac_u_offset = int((1 + convolution_filter.half_sup +
                                 (-scaled_uv[0] + disc_u)) *
                                convolution_filter.oversample)
            frac_v_offset = int((1 + convolution_filter.half_sup +
                                 (-scaled_uv[1] + disc_v)) *
                                convolution_filter.oversample)
            disc_u_psf = int(np.round(scaled_uv[0]*2))
            disc_v_psf = int(np.round(scaled_uv[1]*2))
            frac_u_offset_psf = int((1 + convolution_filter.half_sup +
                                     (-scaled_uv[0]*2 + disc_u_psf)) *
                                    convolution_filter.oversample)
            frac_v_offset_psf = int((1 + convolution_filter.half_sup +
                                     (-scaled_uv[1]*2 + disc_v_psf)) *
                                    convolution_filter.oversample)
            if (disc_v + Ny // 2 + convolution_filter.half_sup >= Ny or
                disc_u + Nx // 2 + convolution_filter.half_sup >= Nx or
                disc_v + Ny // 2 - convolution_filter.half_sup < 0 or
                disc_u + Nx // 2 - convolution_filter.half_sup < 0):
                continue
            for conv_v in filter_index:
                v_tap = \
                    convolution_filter.filter_taps[conv_v *
                                                   convolution_filter.oversample
                                                   + frac_v_offset]
                v_tap_psf = \
                    convolution_filter.filter_taps[conv_v *
                                                   convolution_filter.oversample
                                                   + frac_v_offset_psf]

                grid_pos_v = disc_v + conv_v + Ny // 2
                grid_pos_v_psf = disc_v_psf + conv_v + Ny
                for conv_u in filter_index:
                    u_tap = \
                        convolution_filter.filter_taps[conv_u *
                                                       convolution_filter.oversample
                                                       + frac_u_offset]
                    u_tap_psf = \
                        convolution_filter.filter_taps[conv_u *
                                                       convolution_filter.oversample
                                                       + frac_u_offset_psf]
                    conv_weight = v_tap * u_tap
                    conv_weight_psf = v_tap_psf * u_tap_psf
                    grid_pos_u = disc_u + conv_u + Nx // 2
                    grid_pos_u_psf = disc_u_psf + conv_u + Nx
                    for p in range(vis.shape[2]):
                        measurement_regular[p, grid_pos_v, grid_pos_u] += \
                            vis[r, c, p] * conv_weight
                    # assuming the PSF is the same for different correlations:
                    sampling_regular[grid_pos_v_psf, grid_pos_u_psf] += \
                        (1+0.0j) * conv_weight_psf

    dirty = np.zeros(measurement_regular.shape, dtype=measurement_regular.dtype)
    psf = np.zeros(sampling_regular.shape, dtype=sampling_regular.dtype)

    for p in range(vis.shape[2]):
        dirty[p,:,:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(measurement_regular[p,:,:])))
    psf[:,:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sampling_regular[:,:])))
    return dirty,psf

def grid_ifft2(uvw, Nx, Ny, convolution_filter):
    import matplotlib.pyplot as plt
    """
    Convolutional gridder (continuum)

    Keyword arguments:
    uvw --- interferometer's scaled uvw coordinates
            (Prerequisite: these uv points are already scaled by the similarity
            theorem, such that -N_x*Cell_l*0.5 <= theta_l <= N_x*Cell_l*0.5 and
            -N_y*Cell_m*0.5 <= theta_m <= N_y*Cell_m*0.5)
    ref_lda --- array of reference lambdas (size of vis channels)
    Nx,Ny --- size of image in pixels
    convolution_filter --- pre-instantiated AA_filter anti-aliasing
                           filter object
    """
    filter_index = \
        np.arange(-convolution_filter.half_sup,convolution_filter.half_sup+1)

    # for deconvolution the PSF should be 2x size of the image (see 
    # Hogbom CLEAN for details), one grid for the sampling function:
    sampling_regular = \
        np.zeros([2*Ny,2*Nx],dtype=np.complex)

    for r in range(uvw.shape[0]):
        scaled_uv = uvw[r,:]
        if scaled_uv[0].value == 0 and scaled_uv[1].value == 0: # skipping null frequency
            continue
        disc_u_psf = int(np.round(scaled_uv[0].value*2))
        disc_v_psf = int(np.round(scaled_uv[1].value*2))
       
        frac_u_offset_psf = int((1 + convolution_filter.half_sup +
                                     (-scaled_uv[0].value*2 + disc_u_psf)) *
                                    convolution_filter.oversample)
        frac_v_offset_psf = int((1 + convolution_filter.half_sup +
                                     (-scaled_uv[1].value*2 + disc_v_psf)) *
                                    convolution_filter.oversample)

        for conv_v in filter_index:
 
            v_tap_psf = \
                    convolution_filter.filter_taps[conv_v *
                                                   convolution_filter.oversample
                                                   + frac_v_offset_psf]

            grid_pos_v_psf = disc_v_psf + conv_v + Ny
            for conv_u in filter_index:
                u_tap_psf = \
                        convolution_filter.filter_taps[conv_u *
                                                       convolution_filter.oversample
                                                       + frac_u_offset_psf]
  
                conv_weight_psf = v_tap_psf * u_tap_psf

                grid_pos_u_psf = disc_u_psf + conv_u + Nx
                #print(grid_pos_u_psf,grid_pos_v_psf)
                # assuming the PSF is the same for different correlations:
                if np.abs(grid_pos_v_psf) < sampling_regular.shape[0] and np.abs(grid_pos_u_psf) < sampling_regular.shape[1]:
                    sampling_regular[grid_pos_v_psf, grid_pos_u_psf] += \
                        (1+0.0j) * conv_weight_psf
    
    psf = np.zeros(sampling_regular.shape, dtype=sampling_regular.dtype)

    psf[:,:] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sampling_regular[:,:])))
    return psf,sampling_regular

def RandomPointing(N,Elevmin=0.,Obsmin=2):
    lat=MeerKATarr.Loc.lat.value
    
    deltaDec=0.5 #in degrees
    deltaHA=0.1 #in hours
    tabDec=np.arange(-90.,90.,deltaDec)
    tabHA=np.arange(-12,12,deltaHA)

    tabtabDec,tabtabHA=np.meshgrid(tabDec,tabHA)
    elev=np.degrees(np.arcsin(np.cos(np.radians(tabtabHA*15.))*np.cos(np.radians(tabtabDec))*np.cos(np.radians(lat))+np.sin(np.radians(tabtabDec))*np.sin(np.radians(lat))))
    elev2=elev.copy()*np.NaN
    mask=np.where(elev>Elevmin)

    elev2[mask]=elev[mask]

    Nelev=len(tabDec)

    tabHAmax=np.zeros(Nelev)
    tabHAmin=np.zeros(Nelev)

    for i in range(Nelev):
        if np.all(np.isnan(elev2[:,i])):
        #print('Badline %d'%i)
            continue
        tmpmin=np.where(elev2[:,i] == np.nanmin(elev2[:,i]))
        tmpmax=np.where(elev2[:,i] == np.nanmax(elev2[:,i]))
        tmpHAmax=tmpmin[0][0]
        tmpHAmin=tmpmax[0][0]
        tabHAmin[i]=tmpHAmin
        tabHAmax[i]=tmpHAmax

        if len(tmpmin[0])==2:
            tabwidth[i]=(tmpmin[0][1]-tmpmin[0][0])
        
    SelectHA=tabHA[tabHAmax.astype(int)][0:-61]
    wloc=np.where(np.abs(SelectHA) >=Obsmin)
    d=wloc[0][-1]
    DEC_upperlimits=tabDec[d]
    # draw DEC
    tabDECsources=np.random.rand(N)*(DEC_upperlimits+90)-90

    tabHAstart=np.zeros(N)
    for i in range(N):
        locdec=tabDec.flat[np.abs(tabDec - tabDECsources[i]).argmin()]
        tmpHAstart=np.random.rand()*(tabHAmax[locdec.astype(int)]-Obsmin-tabHAmin[locdec.astype(int)])+tabHAmin[locdec.astype(int)]
        tabHAstart[i]=tmpHAstart

    return tabDECsources,tabHAstart


def compute(N,Npix,pixelscale,Obslength=2.,Timedelta=300,Elevmin=0,F=1420,split=False):
    
    FOV=pixelscale*Npix/3600*1.0 # Desired Field of View of the image (in degrees) ===> gives size of pixel on the sky. Should be compatible with galaxy size range in degrees.

    tabDEC,tabHAstart=RandomPointing(N,Elevmin,Obsmin=Obslength)
    tabPSFList=[]
    tabSamplingList=[]
    Ndates=Obslength*3600./Timedelta  #Timedelta in seconds
    uvw=UVW(MeerKATarr.Array,F) # setting times, array and frequency

    for icase in range(N):
        tmpdec=np.radians(tabDEC[icase])

        if split == False:
            tmptabHA=np.radians((np.arange(Ndates)*Timedelta*1./3600+tabHAstart[icase])*15.)
            uvw.compute(tmptabHA,tmpdec) # computing uv coverage from pointing
            tmpimg=Imager(uvw.uvw[...,0:2],fov=FOV) # preparing the imager
            tmpPSF,tmpSampling=tmpimg.make_psf(npix=Npix,freq=F)  # gridding
            tabPSFList.append(tmpPSF)
            tabSamplingList.append(tmpSampling)
            del tmpimg

        else:
            tabPSF=[]
            tabSampling=[]
            for idate in range(int(Ndates)):
                tmptabHA=np.radians((idate*Timedelta*1./3600+tabHAstart[icase])*15.)
                uvw.compute(tmptabHA,tmpdec) # computing uv coverage from pointing
                tmpimg=Imager(uvw.uvw[...,0:2],fov=FOV) # preparing the imager
                tmpPSF,tmpSampling=tmpimg.make_psf(npix=Npix,freq=F)  # gridding
                tabPSF.append(tmpPSF)
                tabSampling.append(tmpSampling)
                
            tabPSFList.append(tabPSF)
            tabSamplingList.append(tabSampling)
                
        #print(np.degrees(tmptabHA))
        #print(tmpdec)
    del tmpimg
    del uvw
        
    #Pointing_RA=5.4     # Right ascension in degrees. (15° = 1h of RA)
    #Pointing_DEC=-30.83 # Declination in degrees.
    #Pointing=SkyCoord(Pointing_RA,Pointing_DEC,unit='deg') # generating pointing object

# Time settings
    #Obs_start='2020-10-05T20:00:00' # in Universal Time (UT)
    #Obs_end='2020-10-05T20:05:00'
    #tstart=Time(Obs_start,format='isot',scale='utc')
    #tend=Time(Obs_end,format='isot',scale='utc')

# Time intervals and time integration
    #delta_t=300. # integration time per uv component (small will increase computing time for long observations)
    #tstep=TimeDelta(delta_t,format='sec') # using convenient time object
    #Ndates=round(((tend-tstart)/tstep).value) # number of time steps
    #timetab=Time([tstart+i*tstep for i in range(int(Ndates))])
    
 # Setting 
    
    
    return tabPSFList,tabSamplingList,tabDEC,tabHAstart

if __name__ == '__main__':

    sys.exit(main())

