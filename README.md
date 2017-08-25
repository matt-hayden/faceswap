# faceswap #
This is the code behind the [Switching Eds blog 
post](http://matthewearl.github.io/2015/07/28/switching-eds-with-python/) 
packaged as a command-line utility.

## Installation ##

This is in apprehensive development, and setup goes as follows:

1. Install OpenCV with Python bindings on your system. My Debian-based 
experience is `sudo apt-get install opencv-python`.
   I don't know how to pull this into Python setuptools, or if that's wise.
   As this is unstable, I recommend virtualenv -- `pip install 
python-virtualenv`.
1. Clone this repo. For future reference, let's say you cloned to `~/faceswap`.
1. Form the training set with `make -C faceswap/data`. This ought to download a 
large landmarks file and decompress it (I don't think I can re-distribute it). 
Expect 100MB or more.
1. Decide on a working directory for python development. Say, `~/faceswap-dev`, 
and begin with `virtualenv --python=python2.7 ~/faceswap-dev`
1. Switch to this directory and into the virtualenv with `. bin/activate`.
1. Install from your clone directory `pip install -e ~/faceswap`.
1. The command-line utility faceswap will appear in your path.

### Simplest usage: ###

```
faceswap <head image> <face image>
```

If successful, a file `output.jpg` will be produced with the facial features 
from `<head image>` replaced with the facial features from `<face image>`.

