This is the code behind the [Switching Eds blog post](http://matthewearl.github.io/2015/07/28/switching-eds-with-python/) packaged as a command-line utility.

This is in apprehensive development, and setup goes as follows:

1. Install OpenCV with Python bindings on your system. My Debian-based experience is `sudo apt-get install opencv-python`.
   I don't know how to pull this into Python setuptools.
2. I'd also install python-virtualenv
3. Clone this repo. For future reference, let's say you cloned to `~/faceswap`.
4. Form the training set with `make -C faceswap/data`. This ought to download a large landmarks file and decompress it. Expect 100MB or more.
5. Decide on a working directory for python development. Say, `~/faceswap-env`, and begin with `virtualenv --python=python2.7 ~/faceswap-env`
6. Switch to this virtualenv using `~/faceswap-env/bin/activate`.
7. Install from your clone directory `pip install -e ~/faceswap`.
8. The command-line utility faceswap will appear in your path.

Simplest usage:

`
faceswap <head image> <face image>
`

If successful, a file `output.jpg` will be produced with the facial features from `<head image>` replaced with the facial features from `<face image>`.

