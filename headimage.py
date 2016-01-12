#!/usr/bin/python2
import os, os.path

import cv2
import numpy as np

from faceswap import FaceDetectError, FEATHER_AMOUNT, OVERLAY_POINTS, detector, draw_convex_hull, predictor


def get_landmarks(im):
	rects = detector(im, 1)
	# np.matrix([[p.x, p.y] for p in predictor(im, rects[face_number or 0]).parts()])
	for f in rects:
		yield np.matrix( [[ p.x, p.y ] for p in predictor(im, f).parts()] )


def get_face_mask(shape, face_landmarks):
	im = np.zeros(shape[:2], dtype=np.float64)

	for group in OVERLAY_POINTS:
		draw_convex_hull(im,
						 face_landmarks[group],
						 color=1)

	im = np.array([im, im, im]).transpose((1, 2, 0))

	im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
	im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

	return im


class HeadImageError(FaceDetectError):
	pass


class HeadImage:
	def close(self, arg):
		# use with contextlib.closing
		# try to same some memory
		del self.im
		self.im = []
		del self.mask
		self.mask = []
	def __init__(self, arg):
		self.filename = ''
		self.has_cache = False
		self.im = []
		self.landmarks = []
		self.mask = []
		self.modified = False
		self.shape = []
		self.size = None
		if os.path.isfile(arg):
			self.filename = arg
			dirname, basename = os.path.split(arg)
			self.label, ext = os.path.splitext(basename)
			if ext.lower() in [ '.npz' ]:
				self.loadz(arg)
				self.set_mask()
			else:
				self.read()
				if 0 < self.detect_faces():
					self.set_mask()
			print "Loaded", len(self.landmarks), "face(s) from", arg
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
	def read(self, filename=None):
		if filename:
			self.filename = filename
		self.size = os.path.getsize(self.filename)
		self.im = cv2.imread(self.filename, cv2.IMREAD_COLOR)
		self.shape = self.im.shape
	def detect_faces(self):
		if not len(self.im):
			self.read()
		self.landmarks = list(get_landmarks(self.im))
		return len(self.landmarks)
	def resize(self, factor=1., shape=None):
		assert isinstance(factor, float) # or Decimal, etc.
		if factor == 1.:
			return
		elif factor <= 0:
			raise ValueError("factor must be positive")
		if not shape:
			shape = (self.shape[1]*factor, self.shape[0]*factor)
		s_image = cv2.resize(self.im, shape) if self.im else None
		s_mask = cv2.resize(self.mask, shape) if self.mask else None
		for L in self.landmarks:
			L[0] *= factor_x
			L[1] *= factor_y
		self.im, self.modified, self.shape = s_image, True, shape
		self.set_mask(s_mask)
	def horizontal_flip(self, axis=None):
		f_image = np.fliplr(self.im) if self.im else None
		f_mask = np.fliplr(self.mask) if self.mask else None
		for L in self.landmarks:
			m = self.shape[0]
			f_x = m-L[:, 1]
			L[:, 1] = f_x
		self.im, self.modified = f_image, True
		self.set_mask(f_mask)
	def set_mask(self, arg=0, **kwargs):
		if isinstance(arg, int):
			if not len(self.landmarks):
				raise HeadImageError("'{}' has no detected faces".format(self.filename))
			self.mask = get_face_mask(self.shape, self.landmarks[arg], **kwargs)
		elif len(arg): # assume arg is an image
			self.mask = arg
		else:
			raise HeadSwapError("Cannot set mask to {}".format(type(arg)) )
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
	@property
	def alpha(self):
		if len(self.mask):
			return self.mask[:,:,0]*256
		
		
