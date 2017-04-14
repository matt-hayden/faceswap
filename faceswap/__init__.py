import os, os.path

__version__ = '1.0.2-dev1'

class FaceDetectError(Exception):
	pass

# used with the predictor below
JAW_POINTS		= list(range(0, 17))
FACE_POINTS		= list(range(17, 68))

RIGHT_BROW_POINTS	= list(range(17, 22))
LEFT_BROW_POINTS	= list(range(22, 27))
NOSE_POINTS		= list(range(27, 35))
RIGHT_EYE_POINTS	= list(range(36, 42))
LEFT_EYE_POINTS		= list(range(42, 48))
MOUTH_POINTS		= list(range(48, 61))
# what about?		= list(range(62, 68))

PREDICTOR_PATH = "data/shape_predictor_68_face_landmarks.dat"
if not os.path.isfile(PREDICTOR_PATH):
        dirname, basename = os.path.split(__file__)
        PREDICTOR_PATH = os.path.join(dirname, PREDICTOR_PATH)

from cli import swap_many

__all__ = [ 'FaceDetectError', '__version__', 'swap_many' ]
__all__.extend(['PREDICTOR_PATH', 'FACE_POINTS', 'MOUTH_POINTS', 'RIGHT_BROW_POINTS', 'LEFT_BROW_POINTS',
	        'RIGHT_EYE_POINTS', 'LEFT_EYE_POINTS', 'NOSE_POINTS', 'JAW_POINTS'])
