#! /usr/bin/env python2
import os, os.path

import cv2
import dlib
import numpy as np

from . import PREDICTOR_PATH


class FaceDetectError(Exception):
	pass


if not os.path.isfile(PREDICTOR_PATH):
	raise FaceDetectError("'{}' not found".format(PREDICTOR_PATH))
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
							   RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
	LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
	NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

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


def draw_convex_hull(im, points, color):
	points = cv2.convexHull(points)
	cv2.fillConvexPoly(im, points, color=color)


def transformation_from_points(points1, points2):
	"""
	Return an affine transformation [s * R | T] such that:

		sum ||s*R*p1,i + T - p2,i||^2

	is minimized.

	"""
	# Solve the procrustes problem by subtracting centroids, scaling by the
	# standard deviation, and then using the SVD to calculate the rotation. See
	# the following for more details:
	#   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

	points1 = points1.astype(np.float64)
	points2 = points2.astype(np.float64)

	c1 = np.mean(points1, axis=0)
	c2 = np.mean(points2, axis=0)
	points1 -= c1
	points2 -= c2

	s1 = np.std(points1)
	s2 = np.std(points2)
	points1 /= s1
	points2 /= s2

	U, S, Vt = np.linalg.svd(points1.T * points2)

	# The R we seek is in fact the transpose of the one given by U * Vt. This
	# is because the above formulation assumes the matrix goes on the right
	# (with row vectors) where as our solution requires the matrix to be on the
	# left (with column vectors).
	R = (U * Vt).T

	return np.vstack([ np.hstack(( (s2 / s1) * R,
								   c2.T - (s2 / s1) * R * c1.T )),
					   np.matrix([0., 0., 1.]) ])


def warp_im(im, M, dshape):
	output_im = np.zeros(dshape, dtype=im.dtype)
	cv2.warpAffine(im,
				   M[:2],
				   (dshape[1], dshape[0]),
				   dst=output_im,
				   borderMode=cv2.BORDER_TRANSPARENT,
				   flags=cv2.WARP_INVERSE_MAP)
	return output_im


def correct_colours(im1, im2, landmarks1):
	blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
				   np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
				   np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
	blur_amount = int(blur_amount)
	if blur_amount % 2 == 0:
		blur_amount += 1
	im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
	im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

	# Avoid divide-by-zero errors.
	im2_blur += 128 * (im2_blur <= 1.0)

	return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
												im2_blur.astype(np.float64))

