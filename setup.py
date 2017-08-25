"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
#	 long_description = f.read()

# Assume .description adheres to a structure like:
#	name			...
#	version			...
#	author_email	...
# And grab vital package info from it. After the first empty line, the rest
# of the file contains the description.

kwargs = { }
with open(path.join(here, '.description'), encoding='utf-8') as f:
	for line in f:
		line = line.rstrip()
		if (not line):
			break
		kwargs.update([line.split(None, 1)])
	kwargs['description'] = ''.join(f)

# Versions should comply with PEP440.  For a discussion on single-sourcing
# the version across setup.py and the project code, see
# https://packaging.python.org/en/latest/single_source_version.html

setup(
	# See https://pypi.python.org/pypi?%3Aaction=list_classifiers
	classifiers=[
		# How mature is this project? Common values are
		#   3 - Alpha
		#   4 - Beta
		#   5 - Production/Stable
		'Development Status :: 4 - Beta',

		# Indicate who your project is intended for
		'Intended Audience :: Developers',

		# Pick your license as you wish (should match "license" above)
		'License :: Unlicense',

		# Specify the Python versions you support here. In particular, ensure
		# that you indicate whether you support Python 2, Python 3 or both.
		'Programming Language :: Python :: 2',
		'Programming Language :: Python :: 2.7',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
	],

	# You can just specify the packages manually here if your project is
	# simple. Or you can use find_packages().
	packages=find_packages(exclude=['contrib*', 'doc*', 'test*', 'tutorial*']),

	# Alternatively, if you want to distribute just a my_module.py, uncomment
	# this:
	#   py_modules=["my_module"],

	# List run-time dependencies here.  These will be installed by pip when
	# your project is installed. For an analysis of "install_requires" vs pip's
	# requirements files see:
	# https://packaging.python.org/en/latest/requirements.html
	#
	install_requires=[
		"dlib >= 18.18",
		"docopt >= 0.6.2",
		"numpy >= 1.11.2",
		"tqdm >= 4.10"
	],


	# List additional groups of dependencies here (e.g. development
	# dependencies). You can install these using the following syntax,
	# for example:
	# $ pip install -e .[dev,test]
	extras_require={
		'dev': ['check-manifest'],
		'test': ['coverage'],
	},

	# If there are data files included in your packages that need to be
	# installed, specify them here.  If using Python 2.6 or less, then these
	# have to be included in MANIFEST.in as well.
	package_data={
		'faceswap': ['faceswap/data/*.dat', 'data/*.dat'],
	},

	# Although 'package_data' is the preferred approach, in some case you may
	# need to place data files outside of your packages. See:
	# http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
	# In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
	# data_files=[('my_data', ['data/data_file'])],

	# To provide executable scripts, use entry points in preference to the
	# "scripts" keyword. Entry points provide cross-platform support and allow
	# pip to create the appropriate form of executable for the target platform.
	entry_points = {
		'console_scripts': ['faceswap=faceswap.cli:main'],
	},
	**kwargs
)
