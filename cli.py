#!/usr/bin/python3
import os, os.path

import cv2
#import dlib
import numpy as np
import tqdm

from faceswap import ALIGN_POINTS, correct_colours, transformation_from_points, warp_im
from util import *

from headimage import HeadImage

def swap_many(head_filenames, face_filenames):
	"""
	Returns a list of image files that can layer for further processing
	"""
	head_files = [ HeadImage(f) for f in tqdm.tqdm(head_filenames, desc="Scanning heads") ]
	face_files = [ HeadImage(f) for f in tqdm.tqdm(face_filenames, desc="Scanning faces") ]
	#
	for hf in tqdm.tqdm(head_files):
		if not len(hf):
			print "No faces found in", hf.filename
			continue
		output_filename = hf.label+'.tiff'
		layer_filenames = [ hf.filename ]

		for ff in face_files:
			filename_prefix = hf.label+'_'+ff.label

			M = transformation_from_points(hf.landmarks[0][ALIGN_POINTS],
										   ff.landmarks[0][ALIGN_POINTS])
			warped_mask = warp_im(ff.mask, M, hf.shape)
			combined_mask = np.max([hf.mask, warped_mask], axis=0)
			face_alpha = combined_mask[:,:,0]*256 # pick a channel to be substituted for alpha
			if not len(ff.im):
				print "loading", ff.filename
				ff.read()
			warped_im2 = warp_im(ff.im, M, hf.shape).astype(np.float64)
			#layer_filenames += [ filename_prefix+'-orig.png' ]
			#cv2.imwrite(layer_filenames[-1], cv2.merge((im1[:,:,0],
									#im1[:,:,1],
									#im1[:,:,2],
									#head_alpha)) )
			layer_filenames += [ filename_prefix+'-alpha.png' ]
			cv2.imwrite(layer_filenames[-1], cv2.merge((warped_im2[:,:,0],
									warped_im2[:,:,1],
									warped_im2[:,:,2],
									face_alpha)) )
			#layer_filenames += [ filename_prefix+'-head.png' ]
			#cv2.imwrite(layer_filenames[-1], cv2.merge((hf.im[:,:,0],
									#hf.im[:,:,1],
									#hf.im[:,:,2],
									#256.0-head_alpha)) )
			if not len(hf.im):
				print "loading", hf.filename
				hf.read()
			warped_corrected_im2 = correct_colours(hf.im, warped_im2, hf.landmarks[0])
			layer_filenames += [ filename_prefix+'-alpha-color-corrected.png' ]
			cv2.imwrite(layer_filenames[-1], cv2.merge((warped_corrected_im2[:,:,0],
									warped_corrected_im2[:,:,1],
									warped_corrected_im2[:,:,2],
									face_alpha)) )
			blend_im = hf.im * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
			layer_filenames += [ filename_prefix+'-blended.png' ]
			cv2.imwrite(layer_filenames[-1], blend_im)
		if make_layers(layer_filenames, output_filename)==0:
			trash(layer_filenames[1:])
			del hf.im
			yield hf.filename, True
		else:
			yield hf.filename, False

def scanz(args, extensions=['.jpg', '.png', '.jpeg', '.jp2']):
	for arg in tqdm.tqdm(args, desc="Caching results"):
		if os.path.isfile(arg+'.npz'):
			continue
		imh = HeadImage(arg)
		if len(imh.landmarks) == 1:
			imh.savez(arg+'.npz')
		else:
			print "skipping", arg


def main(**kwargs):
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
											  face_filenames=face_files), total=len(head_files)):
			if not result:
				print head_file, "failed"
				rcode = False
		return rcode
