#! /usr/bin/env python3
import os, os.path
import tempfile

import cv2
import numpy as np

import tqdm

from faceswap import FaceDetectError, ALIGN_POINTS, correct_colours, transform_matrix_from_points, warp_im
from util import *

from headimage import HeadImage


class FaceSwapError(FaceDetectError):
	pass


def swap_many(head_filenames, face_filenames, working_directory='', output_directory=''):
	"""
	Returns a list of image files that can layer for further processing
	"""
	def verify_dir(dirname):
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		if not os.path.isdir(dirname):
			raise FaceSwapError("'{}' not valid".format(dirname))
	if working_directory:
		verify_dir(working_directory)
	if output_directory:
		verify_dir(output_directory)
	head_files = []
	for f in tqdm.tqdm(head_filenames, desc="Scanning heads"):
		i = HeadImage(f)
		if not len(i.landmarks)==1:
			print "Skipping", i.filename
			continue
		if not i.has_cache:
			i.savez()
		del i.im # saves memory
		i.im = []
		head_files.append(i)
	face_files = []
	for f in tqdm.tqdm(face_filenames, desc="Scanning faces"):
		i = HeadImage(f)
		if not len(i.landmarks)==1:
			print "Skipping", i.filename
			continue
		if not i.has_cache:
			i.savez()
		del i.im # saves memory
		i.im = []
		face_files.append(i)
	assert head_files and face_files
	for hf in tqdm.tqdm(head_files):
		output_filename = hf.label+'.tiff'
		if output_directory:
			output_filename = os.path.join(output_directory, output_filename)
		layer_filenames = [ hf.filename ]
		h_landmarks = hf.landmarks[0]
		h_align = h_landmarks[ALIGN_POINTS]

		for ff in face_files:
			f_landmarks = ff.landmarks[0]
			f_align = f_landmarks[ALIGN_POINTS]
			results_file = hf.label+'_'+ff.label
			if working_directory:
				results_file = os.path.join(working_directory, results_file)

			M = transform_matrix_from_points(h_align, f_align)
			warped_mask = warp_im(ff.mask, M, hf.shape)
			combined_mask = np.max([hf.mask, warped_mask], axis=0)
			face_alpha = combined_mask[:,:,0]*256 # 0=pick a channel to be substituted for alpha
			if not len(ff.im):
				print "loading", ff.filename
				ff.read()
			warped_im2 = warp_im(ff.im, M, hf.shape).astype(np.float64)
			"""
			layer_filenames += [ results_file+'-orig.png' ]
			cv2.imwrite(layer_filenames[-1], cv2.merge((im1[:,:,0],
									im1[:,:,1],
									im1[:,:,2],
									head_alpha)) )
			"""
			layer_filenames += [ results_file+'-alpha.png' ]
			cv2.imwrite(layer_filenames[-1], cv2.merge((warped_im2[:,:,0],
									warped_im2[:,:,1],
									warped_im2[:,:,2],
									face_alpha)) )
			"""
			layer_filenames += [ results_file+'-head.png' ]
			cv2.imwrite(layer_filenames[-1], cv2.merge((hf.im[:,:,0],
									hf.im[:,:,1],
									hf.im[:,:,2],
									256.0-head_alpha)) )
			"""
			if not len(hf.im):
				print "loading", hf.filename
				hf.read()
			warped_corrected_im2 = correct_colours(hf.im, warped_im2, h_landmarks)
			layer_filenames += [ results_file+'-alpha-color-corrected.png' ]
			cv2.imwrite(layer_filenames[-1], cv2.merge((warped_corrected_im2[:,:,0],
									warped_corrected_im2[:,:,1],
									warped_corrected_im2[:,:,2],
									face_alpha)) )
			blend_im = hf.im * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
			layer_filenames += [ results_file+'-blended.png' ]
			cv2.imwrite(layer_filenames[-1], blend_im)
		if make_layers(layer_filenames, output_filename)==0: # command successfully exits with 0
			#if not working_directory: # TODO: dangerous
			#	trash(layer_filenames[1:])
			yield hf.filename, True
			del hf
		else:
			yield hf.filename, False
			#del hf


def scanz(args, extensions=['.jpg', '.png', '.jpeg', '.jp2']):
	for arg in tqdm.tqdm(args, desc="Caching results"):
		if os.path.isfile(arg+'.npz'):
			continue
		imh = HeadImage(arg)
		if len(imh.landmarks) == 1:
			imh.savez()
		else:
			print "Skipping", arg


def main(**kwargs):
	output_directory = kwargs.pop('--output')
	working_directory = kwargs.pop('--temp')
	if kwargs.pop('scan'):
		args = expand_directories_in_args(kwargs.pop('<FILES>'))
		return scanz(args)
	elif kwargs.pop('swap'):
		rcode = True
		head_files = expand_directories_in_args([kwargs.pop('<HEAD_DIR>')])
		print "Head files:", ','.join(head_files)
		face_files = expand_directories_in_args([kwargs.pop('<FACE_DIR>')])
		print "Face files:", ','.join(face_files)
		for hf, result in tqdm.tqdm(swap_many(head_filenames=head_files,
											  face_filenames=face_files,
											  output_directory=output_directory or '',
											  working_directory=working_directory or tempfile.mkdtemp()),
									total=len(head_files)):
			if not result:
				#print head_file, "failed"
				rcode = False
		return rcode
