
shape_predictor_68_face_landmarks.dat:
	[ -f $@.bz2 ] || wget -c 'http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2'
	bunzip2 $@.bz2
	sha256sum -c SHA256SUMS
