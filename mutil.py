#! /usr/bin/env python2
import numpy as np


def make_transform_matrix(scale, angle, translation, dtype=np.float64):
	M = np.identity(3, dtype=dtype)

	c, s = np.cos(angle), np.sin(angle)
	R = np.matrix( [(c, s), (-s, c)], dtype=dtype)
	M[0:2, 0:2] = scale*R
	M[0:2, 2] = translation

	return M


