from distutils.core import setup,Extension
from Cython.Build import cythonize
import numpy

minRiskUtils=[Extension("DeepDeconv.utils.min_risk_utils",["DeepDeconv/utils/min_risk_utils.pyx"],include_dirs=[numpy.get_include()])]

setup(
    ext_modules = cythonize(minRiskUtils)
)
