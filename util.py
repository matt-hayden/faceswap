#!/usr/bin/python2
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
