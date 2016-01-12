#!/usr/bin/python2
import os, os.path

import cv2
import numpy as np

#from faceswap import FaceDetectError, OVERLAY_POINTS, detector, draw_convex_hull, predictor
from faceswap import *


def get_landmarks(im):
	rects = detector(im, 1)
	for f in rects:
		yield np.matrix( [[ p.x, p.y ] for p in predictor(im, f).parts()] )


def get_face_mask(shape, face_landmarks, feather_amount=11, dtype=np.float64):
	im = np.zeros(shape[:2], dtype=dtype)

	for group in OVERLAY_POINTS:
		draw_convex_hull(im, face_landmarks[group], color=1)

	im = np.array([im, im, im]).transpose((1, 2, 0))

	im = (cv2.GaussianBlur(im, (feather_amount,)*2, 0) > 0) * 1.0
	im = cv2.GaussianBlur(im, (feather_amount,)*2, 0)

	return im


def pdistance(landmarks):
	return np.linalg.norm(
		   np.mean(landmarks[LEFT_EYE_POINTS], axis=0) -
		   np.mean(landmarks[RIGHT_EYE_POINTS], axis=0))


class HeadImageError(Exception):
	pass


class HeadImage:
	def close(self, arg):
		# use with contextlib.closing
		# try to same some memory
		del self.im
		self.im = []
	def __init__(self, arg):
		self.filename = ''
		self.has_cache = False
		self.im = []			# always has a valid len()
		self.landmarks = []
		self.modified = False
		self.shape = []
		self.size = None
		if isinstance(arg, basestring):
			assert os.path.isfile(arg)
			self.filename = arg
			dirname, basename = os.path.split(arg)
			self.label, ext = os.path.splitext(basename)
			if ext.lower() in [ '.npz' ]:
				self.loadz(arg)
			else:
				self.read()
				self.detect_faces()
			print "Loaded", len(self.landmarks), "face(s) from", arg
		elif isinstance(arg, np.ndarray):
			self.im = arg
			self.modified = True
			self.shape = arg.shape
			self.size = arg.nbytes
		else:
			raise NotImplemented()
	def loadz(self, arg):
		d = np.load(arg)
		self.has_cache = True
		self.filename, _ = os.path.splitext(arg)
		dirname, basename = os.path.split(self.filename)
		self.label, _ = os.path.splitext(basename)
		# casting to matrix is required to avoid a broadcasting error
		self.landmarks = [ np.matrix(a) for a in d['landmarks'] ]
		self.shape = d['shape']
		try:
			self.size = os.path.getsize(self.filename)
		except:
			self.size = None
	def savez(self, arg=''):
		np.savez(arg or self.filename+'.npz', landmarks=self.landmarks, shape=self.shape)
		self.has_cache = True
		return True
	def reread(self, filename=None):
		if filename:
			self.filename = filename
		self.size = os.path.getsize(self.filename)
		self.im = cv2.imread(self.filename, cv2.IMREAD_COLOR)
		self.shape = self.im.shape
	def read(self, filename=None):
		if not len(self.im):
			self.reread(filename=filename)
	def detect_faces(self):
		self.read()
		self.landmarks = list(get_landmarks(self.im))
		#self.landmarks = np.array(get_landmarks(self.im))
		for L in self.landmarks:
			for p in L:
				assert (p <= self.shape[:2]).all()
		return len(self.landmarks)
	def get_rescaled(self, scale, **kwargs):
		assert isinstance(scale, float)
		params = kwargs
		if 0 < scale:
			params = { 'fx': scale, 'fy': scale, 'dsize': (0,0) }
		self.read()
		new_image = cv2.resize(self.im, **params)
		new_hi = HeadImage(new_image)
		new_hi.landmarks = [ scale*L for L in self.landmarks ]
		for L in new_hi.landmarks:
			for p in L:
				assert (p <= self.shape[:2]).all()
		return new_hi
	def get_mask(self, arg=0, **kwargs):
		# .im could be None
		if not len(self.landmarks):
			if self.detect_faces() < 1:
				raise HeadImageError("No faces detected")
		return get_face_mask(self.shape, self.landmarks[arg], **kwargs)
	def __len__(self):
		return len(self.landmarks)
	def describe(self):
		"""Form a list of characteristics
		"""
		return [ self.label,
				 self.filename,
				 "{:} b image {}".format(self.size, self.shape),
				 "{} faces detected".format(len(self)),
				 ("Loaded" if len(self.im) else "Not loaded") ]
	def __repr__(self):
		return 'HeadImage<'+','.join(self.describe())+'>'
	def get_pdistances(self):
		"""Pupillary distances for each face
		"""
		return [ pdistance(L) for L in self.landmarks ]
		#return np.fromiter((pdistance(L) for L in self.landmarks), dtype=np.int)
	def correct_colours(self, other_image, face_number=0, blur_frac=0.6, dtype=np.float64):
		"""Colors from self images are overlayed onto another image
		Stolen from Matt's original code
		"""
		blur_amount = int(blur_frac * self.get_pdistances()[face_number])
		if blur_amount % 2 == 0:
			blur_amount += 1
		im1_blur = cv2.GaussianBlur(self.im, (blur_amount,)*2, 0)
		im2_blur = cv2.GaussianBlur(other_image, (blur_amount,)*2, 0)
	
		# Avoid divide-by-zero errors.
		im2_blur += 128 * (im2_blur <= 1.0)
	
		return (other_image.astype(dtype) * im1_blur.astype(dtype) /
												 im2_blur.astype(dtype))


