#!/usr/bin/env python2
import os, os.path

__version__ = '0.1.3'

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.isfile(PREDICTOR_PATH):
        dirname, basename = os.path.split(__file__)
        PREDICTOR_PATH = os.path.join(dirname, PREDICTOR_PATH)

from cli import swap_many

__all__ = [ '__version__', 'PREDICTOR_PATH', 'swap_many' ]
