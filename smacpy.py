#!/bin/env python
#
# smacpy - simple-minded audio classifier in python
# 
# Copyright (c) 2012 Dan Stowell and Queen Mary University of London
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os.path
import sys
import numpy as np
from glob import glob
from scikits.audiolab import Sndfile
from scikits.audiolab import Format
from sklearn.mixture import GMM

from MFCC import melScaling

#######################################################################
# some settings

framelen = 1024
fs = 44100.0
verbose = True

#######################################################################
# main class

class Smacpy:
	"""Smacpy - simple-minded audio classifier in python.
This is a classifier that you can train on a set of labelled audio files, and then it predicts a label for further audio files.
It is designed with two main aims:
 (1) to provide a baseline against which to test more advanced audio classifiers;
 (2) to provide a simple code example of a classifier which people are free to build on.

It uses the very common workflow of taking audio, converting to MFCCs, and modelling the MFCC "bag of frames" with a GMM.

USAGE EXAMPLE:
In this hypothetical example we train on four audio files, labelled as either 'usa' or 'uk', and then test on a separate audio file of someone called hubert:

from smacpy import Smacpy
model = Smacpy("wavs/training", {'karen01.wav':'usa', 'john01.wav':'uk', 'steve02.wav':'usa', 'joe03.wav':'uk'})
model.classify('wavs/testing/hubert01.wav')
	"""

	def __init__(self, wavfolder, trainingdata):
		"""Initialise the classifier and train it on some WAV files.
		'wavfolder' is the base folder, to be prepended to all WAV paths.
		'trainingdata' is a dictionary of wavpath:label pairs."""

		allfeatures = {wavpath:file_to_features(os.path.join(wavfolder, wavpath)) for wavpath in trainingdata}

		# Now determine the normalisation stats, remember them
		self.means = np.mean(anarray, 0)
		self.theinvstds = np.std(anarray, 0)
		for i,val in enumerate(self.theinvstds):
			if val == 0.0:
				self.theinvstds[i] = 1.0
			else:
				self.theinvstds[i] = 1.0 / val		

		# For each label, compile a normalised concatenated list of features
		aggfeatures = {}
		for wavpath, features in allfeatures.iteritems():
			label = trainingdata[wavpath]
			if label not in aggfeatures:
				aggfeatures[label] = np.array([])
			aggfeatures[label] = np.hstack((aggfeatures[label], self.__normalise(features)))

		# For each label, train a GMM and remember it
		self.gmms = {}
		for label, aggf in aggfeatures.iteritems():
			if verbose:
				print "    Training a GMM for label %s, using data of shape %s" % (label, str(np.shape(aggf)))
			self.gmms[label] = GMM(n_components=10, cvtype='full')
			self.gmms[label].fit(aggf)
		if verbose:
			print "  Trained %i classes from %i input files" % (len(self.gmms), len(trainingdata))

	def __normalise(self, data):
		"Normalises data using the mean and stdev of the training data - so that everything is on a common scale."
		return (data - self.means) * self.invstds

	def classify(self, wavpath):
		"Specify the path to an audio file, and this returns the max-likelihood class, as a string label."
		features = self.__normalise(file_to_features(wavpath))
		# For each label GMM, find the overall log-likelihood and choose the strongest
		bestlabel = ''
		bestll = -9e99
		# Choose the biggest
		for label, gmm in self.gmms.iteritems():
			ll = np.sum(gmm.eval(features))
			if ll > bestll:
				bestll = ll
				bestlabel = label
		return bestlabel

#######################################################################
# auxiliary functions

def file_to_features(wavpath):
	"Reads through a mono WAV file, converting each frame to the required features. Returns a 2D array."
	if verbose: print "Reading %s" % wavpath
	if not os.path.isfile(wavpath): raise ValueError("path %s not found" % path)
	sf = Sndfile(wavpath, "r")
	if sf.channels != 1:            raise ValueError("sound file has multiple channels (%i) - mono audio required." % sf.channels)
	if sf.samplerate != fs:         raise ValueError("wanted sample rate %g - got %g." % (fs, sf.samplerate))
	window = np.hamming(framelen)
	features = []
	while(True):
		try:
			chunk = sf.read_frames(framelen, dtype=np.float32)
			if len(chunk) != framelen:
				print "Not read sufficient samples - returning"
				break
			framespectrum = np.fft.fft(window * chunk)
			magspec = abs(framespectrum[:framelen/2])

			# do the frequency warping and MFCC computation
			mfccMaker = melScaling(int(fs), framelen/2, 40)
			melSpectrum = mfccMaker.warpSpectrum(magspec)
			melCepstrum = mfccMaker.getMFCCs(melSpectrum,cn=True)
			melCepstrum = melCepstrum[1:]   # exclude zeroth coefficient
			melCepstrum = melCepstrum[:13] # limit to lower MFCCs

			framefeatures = melCepstrum   # todo: include deltas? that can be your homework.

			features.append(framefeatures)						
		except RuntimeError:
			break
	sf.close()
	ret = np.array(features)
	if verbose:
		print "file_to_features() produced array shape " + str(np.shape(ret))
	return ret

#######################################################################
if __name__ == '__main__':
	foldername = 'wavs'
	if len(sys.argv) > 1:
		foldername = sys.argv[1]

	trainingdata = {}
	pattern = os.path.join(foldername, '*.wav')
	for wavpath in glob(pattern):
		label = os.path.basename(wavpath).split('-')[0]
		shortwavpath = os.path.relpath(wavpath, foldername)
		trainingdata{shortwavpath} = label
	if len(trainingdata)==0:
		raise RuntimeError("Found no files using this pattern: %s" % pattern)
	if verbose:
		print "Class-labels and filenames to be used in training:"
		for wavpath,label in trainingdata.iteritems():
			print " %s: %s" % (label, wavpath)

	model = Smacpy(foldername, trainingdata)

	#################################
	print "Inferred classifications:"
	for wavpath,label in trainingdata.iteritems():
		print " %s" % wavpath
		print " true: %s" % label
		result = model.classify(os.path.join(foldername, wavpath))
		print " inferred: %s" % result

