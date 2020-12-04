# CFIS PSF FWHM Sampler

## PYTHON CONTENT

*. seeing_distribution.npy : Numpy Array of FWHM of CFIS PSF

*. seeing_distribution_class : class to read the data in seeing_distribution.npy

## Usage

Example for generating 2 FWHM samples.

```PYTHON
$ import seeing_distribution_class
$ size_dist = seeing_distribution('seeing_distribution.npy')
$ size_dist.get(2) #returns 2 samples
```

## Remarks

1. The sampler is able to return an infinity of FWHM values that are generated
from the distribution of the FWHM of CFIS PSF.

2. To plot an histogram generate many realisations of FWHM values (e.g. 1e8) and
to plot its distribution, the following link provides a way to do it:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html
