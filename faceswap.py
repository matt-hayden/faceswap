#! /usr/bin/env python2
import os, os.path

import cv2
import dlib
import numpy as np

from . import *		# some constants are moved into __init__.py
import mutil		# some matrix methods are reimplemented in mutil.py


if not os.path.isfile(PREDICTOR_PATH):
	raise FaceDetectError("'{}' not found".format(PREDICTOR_PATH))


# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
				RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
#ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_BROW_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
#OVERLAY_POINTS = [
#	LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
#	NOSE_POINTS + MOUTH_POINTS,
#]
OVERLAY_POINTS = [ FACE_POINTS, ]


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def annotate_landmarks(im, landmarks):
	im = im.copy()
	for idx, point in enumerate(landmarks):
		pos = (point[0, 0], point[0, 1])
		cv2.putText(im, str(idx), pos,
					fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
					fontScale=0.4,
					color=(0, 0, 255))
		cv2.circle(im, pos, 3, color=(0, 255, 255))
	return im


def draw_convex_hull(im, points, color, dtype=np.uint):
	points = cv2.convexHull(points.astype(dtype))
	cv2.fillConvexPoly(im, points, color=color)


def transform_from_points(points1, points2, dtype=np.float64):
	"""
	Return an affine transformation [s * R | T] such that:

		sum ||s*R*p1,i + T - p2,i||^2

	is minimized.

	"""
	# Solve the procrustes problem by subtracting centroids, scaling by the
	# standard deviation, and then using the SVD to calculate the rotation. See
	# the following for more details:
	#   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

	points1 = points1.astype(dtype)
	points2 = points2.astype(dtype)

	# these are ordered pairs
	c1 = np.mean(points1, axis=0)
	c2 = np.mean(points2, axis=0)
	points1 -= c1
	points2 -= c2

	# standard deviation is a measure of size (like dispersion)
	s1 = np.std(points1)
	s2 = np.std(points2)
	scale = (s2/s1)
	points1 /= s1
	points2 /= s2

	U, S, Vt = np.linalg.svd(points1.T * points2)

	# The R we seek is in fact the transpose of the one given by U * Vt. This
	# is because the above formulation assumes the matrix goes on the right
	# (with row vectors) where as our solution requires the matrix to be on the
	# left (with column vectors).
	R = (U * Vt).T
	angle = np.arccos(R[0,0]) # radians
	# each of [ np.arccos(R[0,0]), -np.arcsin(R[1,0]), np.arcsin(R[0,1]), np.arccos(R[1,1]) ] is the same angle
	translation = c2.T - scale*R * c1.T
	
	return scale, angle, translation.T


def transform_matrix_from_points(*args):
	"""A stub function that preserves the functionality of Matt's transformation_from_points()
	"""
	scale, angle, translation = transform_from_points(*args)
	return mutil.make_transform_matrix(scale, angle, translation)


def warp_im(im, M, dshape):
	output_im = np.zeros(dshape, dtype=im.dtype)
	cv2.warpAffine(im,
				   M[:2],
				   (dshape[1], dshape[0]),
				   dst=output_im,
				   borderMode=cv2.BORDER_TRANSPARENT,
				   flags=cv2.WARP_INVERSE_MAP)
	return output_im


