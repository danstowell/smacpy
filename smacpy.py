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
import random

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

	def __init__(self, wavfolder, trainingdata, minpc, maxpc, reductionmode):
		"""Initialise the classifier and train it on some WAV files.
		'wavfolder' is the base folder, to be prepended to all WAV paths.
		'trainingdata' is a dictionary of wavpath:label pairs."""

		if verbose:
			print("Smacpy training using pcile range: %g--%g" % (minpc, maxpc))
		self.mfccMaker = melScaling(int(fs), framelen/2, 40)
		self.mfccMaker.update()

		allfeatures = {wavpath:self.file_to_features(os.path.join(wavfolder, wavpath), minpc, maxpc, reductionmode) for wavpath in trainingdata}

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

	def classify(self, wavpath, minpc, maxpc, reductionmode):
		"Specify the path to an audio file, and this returns the max-likelihood class, as a string label."
		features = self.__normalise(self.file_to_features(wavpath, minpc, maxpc, reductionmode))
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

	def file_to_features(self, wavpath, minpc, maxpc, reductionmode):
		"Reads through a mono WAV file, converting each frame to the required features. Returns a 2D array."
		if minpc==None: minpc = 0.
		if maxpc==None: maxpc = 1.
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

				# Get signal power and prepend to framefeatures - for pcsubset
				sigpow = ((window * chunk) ** 2).sum
				framefeatures = (sigpow, framefeatures)  # note, sigpow is first entry - for easy sorting

				features.append(framefeatures)
			except RuntimeError:
				break
		sf.close()

		minpc_int = int(max(0.,minpc)*len(features))
		maxpc_int = int(min(1.,maxpc)*len(features))
		if reductionmode=='random':    # downsample irregularly
			features = [item[1] for item in features]
			random.shuffle(features)
			features = features[:maxpc_int]
		elif reductionmode=='regular': # downsample regularly
			features = [item[1] for item in features]
			hopsize = int(np.round(1./maxpc))
			features = features[::hopsize]
		else:                          # power percentiles
			# sort "features" by signal power, then drop it back to being just the mfccs, then trim to percentiles - for pcsubset
			features.sort()
			features = [item[1] for item in features]
			features = features[minpc_int:maxpc_int]
		return np.array(features)

#######################################################################
def trainAndTest(trainpath, trainwavs, testpath, testwavs, minpc, maxpc, reductionmode):
	"Handy function for evaluating your code: trains a model, tests it on wavs of known class. Returns (numcorrect, numtotal, numclasses)."
	print("TRAINING")
	model = Smacpy(trainpath, trainwavs, minpc, maxpc, reductionmode)
	print("TESTING")
	ncorrect = 0
	for wavpath,label in testwavs.items():
		result = model.classify(os.path.join(testpath, wavpath), minpc, maxpc, reductionmode)
		if verbose: print(" inferred: %s" % result)
		if result == label:
			ncorrect += 1
	return (ncorrect, len(testwavs), len(model.gmms))

#######################################################################
# If this file is invoked as a script, it carries out a simple runthrough
# of training on some wavs, then testing, with classnames being the start of the filenames
# python smacpy.py -t wavs -T wavs -p 0,0.25,0.5,0.75,1
# python smacpy.py -t ~/aasp_temp/scenes_FROMRDR/scenes_stereo/scenes_stereo/ -T ~/aasp_temp/scenes_FROMRDR/scenes_stereo/scenes_stereo/ -p 0,0.2,0.4,0.6,0.8,1 -n -6
#
# python smacpy.py -t ~/aasp_temp/scenes_FROMRDR/scenes_stereo/scenes_stereo/ -T ~/aasp_temp/scenes_FROMRDR/scenes_stereo/scenes_stereo/ -p 0,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.75,1 -n -6 -m random
# python plotpcresults.py -i output/pcresults_random.csv --jitter=0
#
if __name__ == '__main__':

	def string2floatlist(string): return [float(x) for x in string.split(",")]

	# Handle the command-line arguments for where the train/test data comes from:
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--trainpath', default='wavs', help="Path to the WAV files used for training")
	parser.add_argument('-T', '--testpath',                  help="Path to the WAV files used for testing")
	parser.add_argument('-q', dest='quiet', action='store_true', help="Be less verbose, don't output much text during processing")
	parser.add_argument('-p', '--pcrange' ,  default=[0,0.5,1]  ,    help="Comma-separated list of percentile cutoffs to consider (0--1)", type=string2floatlist)
	parser.add_argument('-m', '--reductionmode', default='percentile', help="Data reduction mode: 'percentile', 'random', 'regular' subsampling of frames")
	parser.add_argument('-o', '--outpath', default=None, help="Path to write pcsubset results to")
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-c', '--charsplit',  default='_',    help="Character used to split filenames: anything BEFORE this character is the class")
	group.add_argument('-n', '--numchars' ,  default=0  ,    help="Instead of splitting using 'charsplit', use this fixed number of characters from the start of the filename", type=int)
	args = vars(parser.parse_args())
	verbose = not args['quiet']

	if args['testpath']==None:
		args['testpath'] = args['trainpath']
	if args['outpath']==None:
		if args['reductionmode']=='percentile':
			args['outpath'] = "output/pcresults.csv"
		else:
			args['outpath'] = "output/pcresults_%s.csv" % args['reductionmode']

	# Build up lists of the training and testing WAV files:
	wavsfound = {'trainpath':{}, 'testpath':{}}
	for onepath in ['trainpath', 'testpath']:
		pattern = os.path.join(args[onepath], '*.wav')
		for wavpath in glob(pattern):
			if args['numchars'] != 0:
				label = os.path.basename(wavpath)[:args['numchars']]
			else:
				label = os.path.basename(wavpath).split(args['charsplit'])[0]
			shortwavpath = os.path.relpath(wavpath, args[onepath])
			wavsfound[onepath][shortwavpath] = label
		if len(wavsfound[onepath])==0:
			raise RuntimeError("Found no files using this pattern: %s" % pattern)
		if verbose:
			print("Class-labels and filenames to be used from %s:" % onepath)
			for wavpath,label in sorted(wavsfound[onepath].items()):
				print(" %s: \t %s" % (label, wavpath))


	outfile = open(args['outpath'], 'wb', 1)
	outfile.write("minpc,maxpc,acc,pcrange,accgain,acc_ci\n")

	if args['reductionmode'] in ['random', 'regular']: # minpc not really used
		minpcrange = [0]
	else:
		minpcrange = args['pcrange'][:-1]

	for whichminpc, minpc in enumerate(minpcrange):
		for maxpc in args['pcrange'][whichminpc+1:]:
			print("-------------------------------------------")
			print((minpc, maxpc))

			if args['testpath'] != args['trainpath']:
				# Separate train-and-test collections
				totcorrect, tottotal, nclasses = trainAndTest(args['trainpath'], wavsfound['trainpath'], args['testpath'], wavsfound['testpath'], minpc, maxpc, args['reductionmode'])
				accuracies = [float(totcorrect)/tottotal]
				print("Got %i correct out of %i (trained on %i classes)" % (ncorrect, ntotal, nclasses))
			else:
				# This runs "stratified leave-one-out crossvalidation": test multiple times by leaving one-of-each-class out and training on the rest.
				# First we need to build a list of files grouped by each classlabel
				labelsinuse = sorted(list(set(wavsfound['trainpath'].values())))
				grouped = {label:[] for label in labelsinuse}
				for wavpath,label in wavsfound['trainpath'].items():
					grouped[label].append(wavpath)
				numfolds = min(len(collection) for collection in grouped.values())
				# Each "fold" will be a collection of one item of each label
				folds = [{wavpaths[index]:label for label,wavpaths in grouped.items()} for index in range(numfolds)]
				totcorrect, tottotal = (0,0)
				accuracies = []
				# Then we go through, each time training on all-but-one and testing on the one left out
				for index in range(numfolds):
					print("Fold %i of %i" % (index+1, numfolds))
					chosenfold = folds[index]
					alltherest = {}
					for whichfold, otherfold in enumerate(folds):
						if whichfold != index:
							alltherest.update(otherfold)
					ncorrect, ntotal, nclasses = trainAndTest(args['trainpath'], alltherest, args['trainpath'], chosenfold, minpc, maxpc, args['reductionmode'])
					totcorrect += ncorrect
					tottotal   += ntotal
					accuracies.append(float(ncorrect)/ntotal)

				print("Got %i correct out of %i (using stratified leave-one-out crossvalidation, %i folds)" % (totcorrect, tottotal, numfolds))

			# calc mean and confidence interval of accuracy
			accuracy = np.mean(accuracies)
			acc_ci = (np.std(accuracies) / np.sqrt(len(accuracies))) * 1.96

			# here we write our stats line to file
			outfile.write("%g,%g,%g,%g,%g,%g\n" % (minpc, maxpc, accuracy, maxpc-minpc, accuracy - (1./nclasses), acc_ci))
	outfile.close()

