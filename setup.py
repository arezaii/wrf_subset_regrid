#from numpy.distutils.core import setup, Extension

#setup(name = 'write_pfb', version = '0.1', \
# ext_modules=[Extension("write_pfb", sources=['write_pfb.f90'], requires=['numpy'])])
def configuration(parent_package='', top_path=None):
	from numpy.distutils.misc_util import Configuration
	config = Configuration('pf_pytools',parent_package, top_path)
	config.add_extension('pf_fort_io',sources=['pf_fort_io.pyf','pf_fort_io.f90'])
	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
