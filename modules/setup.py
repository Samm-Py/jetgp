from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

exts = []
openmp_compile_flag = True
mod1 = Extension('sobol',
                 sources=['sobol.pyx'],
                 include_dirs=[np.get_include()],  # include header from numpy.
                 # enables flag for c99 standard.
                 extra_compile_args=["-I.", "-I.", "-O3", '-fopenmp'],
                 extra_link_args=['-fopenmp', "-O3"],
                 # Link to math library.
                 libraries=['m', 'ifcore', 'ifcoremt', 'svml', 'intlc'],
                 )


setup(
    ext_modules=cythonize(mod1, annotate=False),
)


exts = []
openmp_compile_flag = True
mod2 = Extension('lhs',
                 sources=['lhs.pyx'],
                 include_dirs=[np.get_include()],  # include header from numpy.
                 # enables flag for c99 standard.
                 extra_compile_args=["-I.", "-I.", "-O3", '-fopenmp'],
                 extra_link_args=['-fopenmp', "-O3"],
                 # Link to math library.
                 libraries=['m', 'ifcore', 'ifcoremt', 'svml', 'intlc'],
                 )


setup(
    ext_modules=cythonize(mod2, annotate=False),
)

exts = []
openmp_compile_flag = True
mod3 = Extension('primes',
                 sources=['primes.pyx'],
                 include_dirs=[np.get_include()],  # include header from numpy.
                 # enables flag for c99 standard.
                 extra_compile_args=["-I.", "-I.", "-O3", '-fopenmp'],
                 extra_link_args=['-fopenmp', "-O3"],
                 # Link to math library.
                 libraries=['m', 'ifcore', 'ifcoremt', 'svml', 'intlc'],
                 )


setup(
    ext_modules=cythonize(mod3, annotate=False),
)


exts = []
openmp_compile_flag = True
mod4 = Extension('vdc',
                 sources=['vdc.pyx'],
                 include_dirs=[np.get_include()],  # include header from numpy.
                 # enables flag for c99 standard.
                 extra_compile_args=["-I.", "-I.", "-O3", '-fopenmp'],
                 extra_link_args=['-fopenmp', "-O3"],
                 # Link to math library.
                 libraries=['m', 'ifcore', 'ifcoremt', 'svml', 'intlc'],
                 )


setup(
    ext_modules=cythonize(mod4, annotate=False),
)


exts = []
openmp_compile_flag = True
mod5 = Extension('halton',
                 sources=['halton.pyx'],
                 include_dirs=[np.get_include()],  # include header from numpy.
                 # enables flag for c99 standard.
                 extra_compile_args=["-I.", "-I.", "-O3", '-fopenmp'],
                 extra_link_args=['-fopenmp', "-O3"],
                 # Link to math library.
                 libraries=['m', 'ifcore', 'ifcoremt', 'svml', 'intlc'],
                 )


setup(
    ext_modules=cythonize(mod5, annotate=False),
)


exts = []
openmp_compile_flag = True
mod6 = Extension('hammersley',
                 sources=['hammersley.pyx'],
                 include_dirs=[np.get_include()],  # include header from numpy.
                 # enables flag for c99 standard.
                 extra_compile_args=["-I.", "-I.", "-O3", '-fopenmp'],
                 extra_link_args=['-fopenmp', "-O3"],
                 # Link to math library.
                 libraries=['m', 'ifcore', 'ifcoremt', 'svml', 'intlc'],
                 )


setup(
    ext_modules=cythonize(mod6, annotate=False),
)
