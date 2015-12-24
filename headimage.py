#!/usr/bin/python2
import os, os.path

import cv2
import numpy as np

from faceswap import FEATHER_AMOUNT, OVERLAY_POINTS, detector, draw_convex_hull, predictor

def get_landmarks(im):
	rects = detector(im, 1)
	# np.matrix([[p.x, p.y] for p in predictor(im, rects[face_number or 0]).parts()])
	for f in rects:
		yield np.matrix( [[ p.x, p.y ] for p in predictor(im, f).parts()] )

def get_face_mask(shape, landmarks):
	im = np.zeros(shape[:2], dtype=np.float64)

	for group in OVERLAY_POINTS:
		draw_convex_hull(im,
						 landmarks[group],
						 color=1)

	im = np.array([im, im, im]).transpose((1, 2, 0))

	im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
	im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

	return im

class HeadImage:
	def __init__(self, arg):
		self.filename = ''
		self.im = []
		self.landmarks = []
		self.mask = []
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
		self.filename = arg[:-4]
		dirname, basename = os.path.split(self.filename)
		self.label, _ = os.path.splitext(basename)
		# casting to matrix is required to avoid a broadcasting error
		self.landmarks = [ np.matrix(a) for a in d['landmarks'] ]
		self.shape = d['shape']
		try:
			self.size = os.path.getsize(self.filename)
		except:
			self.size = None
	def savez(self, arg):
		np.savez(arg, landmarks=self.landmarks, shape=self.shape)
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
	def resize(self, factor=1.):
		if factor not in (0., 1.):
			scaled = cw2.resize(self.im, (self.shape[1]*factor, self.shape[0]*factor))
			self.im = scaled
			self.shape = self.im.shape
	def set_mask(self, arg=None, **kwargs):
		if arg:
			self.mask = arg
		elif len(self.landmarks):
			self.mask = get_face_mask(self.shape, self.landmarks[0], **kwargs)
		else:
			print self.filename, ": no faces found"
	def __len__(self):
		return len(self.landmarks)
	def describe(self):
		label = self.label
		return [ label, self.filename, "{:} b image {}".format(self.size, self.shape),
				 "{} faces detected".format(len(self)), ("Loaded" if len(self.im) else "Not loaded") ]
	def __repr__(self):
		return 'HeadImage<'+', '.join(self.describe())+'>'
	@property
	def alpha(self):
		if len(self.mask):
			return self.mask[:,:,0]*256
		
		

