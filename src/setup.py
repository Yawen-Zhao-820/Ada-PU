from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np

extensions = [
    Extension("src.decisionstump", ["src/decisionstump.pyx"], include_dirs=[np.get_include()]),
]

setup(
    ext_modules=cythonize(extensions)
)