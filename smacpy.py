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
import numpy as np
import argparse
from glob import glob
from scikits.audiolab import Sndfile
from scikits.audiolab import Format
from sklearn.mixture import GMM
import csv

from MFCC import melScaling

#######################################################################
# some settings
framelen = 1024
fs = 44100.0
verbose = True

#######################################################################
# main class

class Smacpy:
	"""Smacpy - simple-minded audio classifier in python. See the README file for more details.

	USAGE EXAMPLE:
	In this hypothetical example we train on four audio files, labelled as either 'usa' or 'uk', and then test on a separate audio file of someone called hubert:

	from smacpy import Smacpy
	model = Smacpy("wavs/training", {'karen01.wav':'usa', 'john01.wav':'uk', 'steve02.wav':'usa', 'joe03.wav':'uk'})
	model.classify('wavs/testing/hubert01.wav')

	Note for developers: this code should aim to be understandable, and not too long. Don't add too much functionality, or efficiency ;)
	"""

	def __init__(self, wavfolder, trainingdata):
		"""Initialise the classifier and train it on some WAV files.
		'wavfolder' is the base folder, to be prepended to all WAV paths.
		'trainingdata' is a dictionary of wavpath:label pairs."""

		self.mfccMaker = melScaling(int(fs), framelen/2, 40)
		self.mfccMaker.update()

		allfeatures = {wavpath:self.file_to_features(os.path.join(wavfolder, wavpath)) for wavpath in trainingdata}

		# Determine the normalisation stats, and remember them
		allconcat = np.vstack(list(allfeatures.values()))
		self.means = np.mean(allconcat, 0)
		self.invstds = np.std(allconcat, 0)
		for i,val in enumerate(self.invstds):
			if val == 0.0:
				self.invstds[i] = 1.0
			else:
				self.invstds[i] = 1.0 / val

		# For each label, compile a normalised concatenated list of features
		aggfeatures = {}
		for wavpath, features in allfeatures.items():
			label = trainingdata[wavpath]
			normed = self.__normalise(features)
			if label not in aggfeatures:
				aggfeatures[label] = normed
			else:
				aggfeatures[label] = np.vstack((aggfeatures[label], normed))

		# For each label's aggregated features, train a GMM and remember it
		self.gmms = {}
		for label, aggf in aggfeatures.items():
			if verbose: print("    Training a GMM for label %s, using data of shape %s" % (label, str(np.shape(aggf))))
			self.gmms[label] = GMM(n_components=10) # , cvtype='full')
			self.gmms[label].fit(aggf)
		if verbose: print("  Trained %i classes from %i input files" % (len(self.gmms), len(trainingdata)))

	def __normalise(self, data):
		"Normalises data using the mean and stdev of the training data - so that everything is on a common scale."
		return (data - self.means) * self.invstds

	def classify(self, wavpath):
		"Specify the path to an audio file, and this returns the max-likelihood class, as a string label."
		features = self.__normalise(self.file_to_features(wavpath))
		# For each label GMM, find the overall log-likelihood and choose the strongest
		bestlabel = ''
		bestll = -9e99
		for label, gmm in self.gmms.items():
			ll = gmm.eval(features)[0]
			ll = np.sum(ll)
			if ll > bestll:
				bestll = ll
				bestlabel = label
		return bestlabel

	def file_to_features(self, wavpath):
		"Reads through a mono WAV file, converting each frame to the required features. Returns a 2D array."
		if verbose: print("Reading %s" % wavpath)
		if not os.path.isfile(wavpath): raise ValueError("path %s not found" % path)
		sf = Sndfile(wavpath, "r")
		#if (sf.channels != 1) and verbose: print(" Sound file has multiple channels (%i) - channels will be mixed to mono." % sf.channels)
		if sf.samplerate != fs:         raise ValueError("wanted sample rate %g - got %g." % (fs, sf.samplerate))
		window = np.hamming(framelen)
		features = []
		while(True):
			try:
				chunk = sf.read_frames(framelen, dtype=np.float32)
				if len(chunk) != framelen:
					print("Not read sufficient samples - returning")
					break
				if sf.channels != 1:
					chunk = np.mean(chunk, 1) # mixdown
				framespectrum = np.fft.fft(window * chunk)
				magspec = abs(framespectrum[:framelen/2])

				# do the frequency warping and MFCC computation
				melSpectrum = self.mfccMaker.warpSpectrum(magspec)
				melCepstrum = self.mfccMaker.getMFCCs(melSpectrum,cn=True)
				melCepstrum = melCepstrum[1:]   # exclude zeroth coefficient
				melCepstrum = melCepstrum[:13] # limit to lower MFCCs

				framefeatures = melCepstrum   # todo: include deltas? that can be your homework.

				features.append(framefeatures)
			except RuntimeError:
				break
		sf.close()
		return np.array(features)

#######################################################################
def trainAndTest(trainpath, trainwavs, testpath, testwavs):
	"Handy function for evaluating your code: trains a model, tests it on wavs of known class. Returns (numcorrect, numtotal, numclasses)."
	print("TRAINING")
	model = Smacpy(trainpath, trainwavs)
	print("TESTING")
	ncorrect = 0
	for wavpath,label in testwavs.items():
		result = model.classify(os.path.join(testpath, wavpath))
		if verbose: print(" inferred: %s" % result)
		if result == label:
			ncorrect += 1
	return (ncorrect, len(testwavs), len(model.gmms))

#######################################################################
# This handles the invocation as set up for the AASP challenge
# eg:  python smacpy.py --trainlist trainlist.example.txt --testlist testlist.example.txt --outlist output.txt
if __name__ == '__main__':
	# Handle the command-line arguments for where the train/test data comes from:
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--trainlist', default='NOT_SPECIFIED', help="Path to the file listing WAV files used for training")
	parser.add_argument('-T', '--testlist',  default='NOT_SPECIFIED', help="Path to the file listing WAV files used for testing")
	parser.add_argument('-o', '--outlist',   default='output.txt',    help="Path to write results to")
	parser.add_argument('-q', dest='quiet', action='store_true', help="Be less verbose, don't output much text during processing")
	args = vars(parser.parse_args())
	verbose = not args['quiet']

	# Load the training list as a dictionary of "filename->classification" data
	trainlist = {}
	for row in csv.reader(file(args['trainlist']), delimiter="\t"):
		if len(row)==2:
			trainlist[row[0]] = row[1]
		elif len(row)>2:
			raise ValueError("Row has more than one tab character in: ", row)

	print("Training files to be used:")
	for wavpath,label in sorted(trainlist.items()):
		print(" %s: \t %s" % (label, wavpath))

	# Load the test list
	testlist = []
	for row in csv.reader(file(args['testlist']), delimiter="\t"):
		if len(row)==1:
			testlist.append(row[0])
		elif len(row)>1:
			raise ValueError("Row has a tab character in - SHOULD NOT happen for testlist: ", row)
	print("Testing files to be used:")
	for item in testlist: print(item)

	# OK so let's go
	outlist = file(args['outlist'], 'wb')
	print("TRAINING")
	model = Smacpy('/', trainlist)
	print("TESTING")
	for wavpath in testlist:
		result = model.classify(wavpath)
		outlist.write("%s\t%s\n" % (wavpath, result))
	outlist.close()
	print("Finished.")

