# python setup.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension("MLS",["MLS.pyx"])]
setup(
    name = "MLS pyx",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)