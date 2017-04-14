from setuptools import setup

setup(name='faceswap',
      use_vcs_version=True,
      description='Packaging of github.com/matthewearl/faceswap with command-line interface',
      url='http://github.com/matt-hayden/faceswap',
	  maintainer="Matt Hayden",
	  maintainer_email="github.com/matt-hayden",
      license='MIT',
      packages=find_packages(),
	  include_package_data=True,
	  package_data={'faceswap': ['faceswap/data/*.dat', 'data/*.dat']},
	  entry_points = {
	    'console_scripts': ['faceswap=faceswap.cli:main'],
	  },
      # TODO: I don't know how to include OpenCV python bindings as a dependency
      install_requires=[
		"dlib >= 18.18",
		"docopt >= 0.6.2",
		"numpy >= 1.11.2",
		"tqdm >= 4.10"
      ],
      zip_safe=False,
	  setup_requires = [ "setuptools_git >= 1.2", ]
     )
