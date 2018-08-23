from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

# cmd SET VS90COMNTOOLS=%VS120COMNTOOLS%
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("SalaryBenchmark_MF", ["SalaryBenchmark_MF.pyx"], include_dirs=[np.get_include()]),
                   Extension("SalaryBenchmark_MF_SideMatrix", ['SalaryBenchmark_MF_SideMatrix.pyx'], include_dirs=[np.get_include()])]
)