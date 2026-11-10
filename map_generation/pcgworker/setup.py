from distutils.core import setup, Extension
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules=[
    Extension("c_acc",
              ["c_acc.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native" ],
              )
]

setup(name = "c_acc",include_dirs=numpy.get_include(),cmdclass = {"build_ext": build_ext}, ext_modules = ext_modules)
