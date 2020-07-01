from numpy.distutils.core import setup, Extension

setup(name = 'write_pfb', version = '0.1', \
      ext_modules=[Extension("write_pfb", sources=['write_pfb.f90'], requires=['numpy'])])
