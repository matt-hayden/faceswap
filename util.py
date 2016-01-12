#! /usr/bin/env python2
import os, os.path
import subprocess


def make_layers(files_in, file_out):
	# relies on ImageMagick's convert utility
	if os.path.exists(file_out):
		trash([file_out])
	proc = subprocess.Popen(['convert']+files_in+[file_out])
	print "Merging {} files".format(len(files_in))
	return proc.wait()


def trash(files):
	proc = subprocess.Popen(['trash']+files)
	print "Trashing {} files".format(len(files))
	return proc.wait()


def expand_directories_in_args(args, extensions=['.jpg', '.png', '.jpeg', '.jp2', '.tiff']):
	"""
	Args that are files are passed through, args that are directories have their
	valid files tacked on. No recursion.
	"""
	assert isinstance(args, (list, tuple))
	files = [ f for f in args if os.path.isfile(f) ]
	dirs = [ d for d in args if os.path.isdir(d) ]
	for d in dirs:
		dfiles = [ os.path.join(d, f) for f in os.listdir(d) ]
		files.extend(f for f in dfiles if os.path.isfile(f))
	for f in files[:]:
		of, x = os.path.splitext(f)
		x = x.lower()
		if '.npz' == x:
			cachefile, f = f, of
			if f in files:
				f_time, cachefile_time = os.path.getmtime(f), os.path.getmtime(cachefile)
				if f_time < cachefile_time:
					files.remove(f)
				else:
					files.remove(cachefile)
					trash([cachefile])
		elif x not in extensions:
			print "warning:", f, "maybe not image"
	return files
