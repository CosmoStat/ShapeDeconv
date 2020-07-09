#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:48:43 2019

@author: Julien N. Girard
"""
import numpy as np
import galsim
from astropy.io import fits
import matplotlib.pyplot as plt

def pause():
    input('Press ENTER to continue...')

is_saved = True    
    
big_fft_params=galsim.GSParams(maximum_fft_size=81488)
pixelscale=0.01 # size of pixel in arcsec
stampimage = galsim.ImageF(96, 96, scale=pixelscale)
b=galsim.BoundsI(1,96,1,96)
stamp=stampimage[b]

catalogsfg=fits.open('catalogue_SFGs_complete_deep.fits')
catalogsfg.info()
cat1=catalogsfg[1]
catdatasfg=cat1.data

flux1400sfg=catdatasfg['I1400']  # flux density at 1400 MHz
sizesfg=catdatasfg['size'] # angular size on the sky (in arcsec)
e1=catdatasfg['e1']  # first ellipticity
e2=catdatasfg['e2']  # second ellipticity

plt.plot(sizesfg,'.')
plt.ylim([0,10])

filterobj=np.logical_and(sizesfg > 45*pixelscale, sizesfg <50*pixelscale)#10-100
filterobj2=np.where(filterobj == True)[0]

nobj=100
Ntotobj=len(filterobj2)
print(Ntotobj)
randidx=np.random.choice(Ntotobj,nobj)

# Star-forming galaxies (T-RECS)

gals = np.zeros((nobj,96,96))
for i,iobj in enumerate(randidx):
    gauss_gal=galsim.Gaussian(fwhm=sizesfg[iobj],flux=flux1400sfg[iobj])
    gal = galsim.Exponential(half_light_radius=gauss_gal.half_light_radius, 
                             flux=flux1400sfg[iobj], gsparams=big_fft_params)
    ellipticity = galsim.Shear(e1=e1[iobj],e2=e2[iobj])
    gal = gal.shear(ellipticity)
    gal2=gal.drawImage(stamp,scale=pixelscale)
    gals[i] = gal2.array
#    plt.imshow(gal2.array)
#    plt.show()
#    plt.close()
#    pause()

if not(is_saved):
    np.savez("Cat-SFG.npz",nobj=nobj,listgal=np.array(gals),
         flux1400sfg=flux1400sfg[randidx],sizesfg=sizesfg[randidx],
         randidx=randidx,e1=e1[randidx],e2=e2[randidx])

plt.plot(e1[randidx],'.')
plt.plot(e2[randidx],'.')