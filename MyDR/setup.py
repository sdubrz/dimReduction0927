from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules=cythonize(Extension(name="_utils", sources=["_utils.pyx"])))