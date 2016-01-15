#! /usr/bin/env python3
import os, os.path
import tempfile

import cv2
import numpy as np

import tqdm

from . import *
from faceswap import *
from util import *

from headimage import HeadImage


class FaceSwapError(FaceDetectError):
	pass


def swap_many(head_filenames, face_filenames, **kwargs):
	"""Expanded face swap routine:
		Multiple filenames for head image
		Multiple filenames for face image
		Transparent layering

		options:
			allow_rescale		head images are rescaled to the largest face image
			mirror_also			head images are duplicated and flipped (at least 2x slowdown)
			output_directory
			working_directory
	"""
	def verify_dir(dirname):
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		if not os.path.isdir(dirname):
			raise FaceSwapError("'{}' not valid".format(dirname))
	allow_rescale = kwargs.pop('allow_rescale', None)
	mirror_also = kwargs.pop('mirror_also', True)
	working_directory = kwargs.pop('working_directory', '')
	if working_directory:
		verify_dir(working_directory)
	output_directory = kwargs.pop('output_directory', '')
	if output_directory:
		verify_dir(output_directory)
	#
	head_files, face_files = [], []
	params = [ ('heads', head_filenames, head_files), ('faces', face_filenames, face_files) ]
	total_steps = len(head_filenames)*len(face_filenames)
	bar = tqdm.tqdm(desc="Scanning...", total=total_steps)
	for label, files, dest in params:
		for f in files:
			i = HeadImage(f)
			if not len(i.landmarks)==1:
				print "Skipping", i.filename
				continue
			if not i.has_cache:
				i.savez()
			#del i.im # saves memory?
			i.im = []
			dest.append(i)
			bar.update(1)
	assert head_files and face_files
	#
	total_steps = len(head_files)*(len(face_files)+2)
	if mirror_also:
		total_steps *= 2
		def head_file_iterator(his):
			for hi in his:
				yield hi
				yield hi.get_horizontally_flipped()
	else:
		def head_file_iterator(his):
			return his
	
	bar = tqdm.tqdm(desc="Swapping...", total=total_steps)
	for orig_hf in head_file_iterator(head_files):
		hf = orig_hf
		output_filename = hf.label+'.tiff'
		if output_directory:
			output_filename = os.path.join(output_directory, output_filename)

		"""First, we dry-run through the face files
		"""
		h_landmarks = hf.landmarks[0]
		h_align = h_landmarks[ALIGN_POINTS]
		if allow_rescale:
			largest_face_filename = ''
			for ff in face_files:
				#
				f_landmarks = ff.landmarks[0]
				f_align = f_landmarks[ALIGN_POINTS]
				#
				scale, angle, translation = transform_from_points(h_align, f_align)
				if (1 < scale):
					if __debug__: print "Rescaling..."
					hf = orig_hf.get_rescaled(scale)
					h_landmarks = hf.landmarks[0]
					h_align = scale*h_align
					largest_face_filename = ff.filename
			if largest_face_filename:
				dirname, basename = os.path.split(largest_face_filename)
				label, _ = os.path.splitext(basename)
				hf.label = orig_hf.label + '_for_' + label
		head_results_file = hf.label
		if working_directory:
			head_results_file = os.path.join(working_directory, head_results_file)
		# we build a stack of output files
		if hf.modified:
			hf.filename = head_results_file+'-background.png'
			cv2.imwrite(hf.filename, hf.im)
		layer_filenames = [ hf.filename ]
		if __debug__: print 'head|face|scale|angle|translation|output'
		for ff in face_files:
			f_landmarks = ff.landmarks[0]
			f_align = f_landmarks[ALIGN_POINTS]
			#
			results_file = hf.label+'_'+ff.label
			if working_directory:
				results_file = os.path.join(working_directory, results_file)

			scale, angle, translation = transform_from_points(h_align, f_align)

			if __debug__: print '|'.join(str(_) for _ in ( hf.filename,
														   ff.filename,
														   scale,
														   angle,
														   translation,
														   results_file ))

			center = (0,0) # gets thrown out, anyway
			rot_mat = cv2.getRotationMatrix2D( center, angle, scale );
			rot_mat[0:2, 2] = translation
			# maybe try to guess the fit here by testing M*f_align - h_align
			warped_mask = warp_im(ff.get_mask(), rot_mat, hf.shape)
			combined_mask = np.max([hf.get_mask(), warped_mask], axis=0)
			face_alpha = combined_mask[:,:,0]*256 # 0=pick a channel to be substituted for alpha
			ff.read()
			warped_im2 = warp_im(ff.im, rot_mat, hf.shape).astype(np.float64)
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
			hf.read()
			warped_corrected_im2 = hf.correct_colours(warped_im2)
			layer_filenames += [ results_file+'-alpha-color-corrected.png' ]
			cv2.imwrite(layer_filenames[-1], cv2.merge((warped_corrected_im2[:,:,0],
									warped_corrected_im2[:,:,1],
									warped_corrected_im2[:,:,2],
									face_alpha)) )
			blend_im = hf.im * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
			layer_filenames += [ results_file+'-blended.png' ]
			cv2.imwrite(layer_filenames[-1], blend_im)
			bar.update(1)
		if make_layers(layer_filenames, output_filename)==0: # command successfully exits with 0
			yield orig_hf.filename, layer_filenames, True
			bar.update(2)
		else:
			yield hf.filename, layer_filenames, False
	if __debug__: print "Intermediate files are in '{}'".format(working_directory or '.')


def scanz(args, extensions=['.jpg', '.png', '.jpeg', '.jp2']):
	for arg in tqdm.tqdm(args, desc="Caching..."):
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
		if __debug__: print "Head files:", ','.join(head_files)
		face_files = expand_directories_in_args([kwargs.pop('<FACE_DIR>')])
		if __debug__: print "Face files:", ','.join(face_files)
		for hf, _, result in swap_many(head_filenames=head_files,
										face_filenames=face_files,
										output_directory=output_directory or '',
										working_directory=working_directory or tempfile.mkdtemp() ):
			if not result:
				print head_file, "failed"
				rcode = False
		return rcode


# vim: tabstop=4 shiftwidth=4 softtabstop=4
