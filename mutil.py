#! /usr/bin/env python2
import numpy as np


def make_transform_matrix(scale, angle, translation):
	M = np.identity(3, dtype=np.float64)

	c, s = np.cos(angle), np.sin(angle)
	R = np.matrix( [(c, s), (-s, c)], dtype=np.float64)
	M[0:2, 0:2] = scale*R
	M[0:2, 2] = translation

	return M


