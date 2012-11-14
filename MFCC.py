'''MFCC.py

Calculation of MFCC coefficients from frequency-domain data

Adapted from the Vampy example plugin "PyMFCC" by Gyorgy Fazekas
http://code.soundsoftware.ac.uk/projects/vampy/repository/entry/Example%20VamPy%20plugins/PyMFCC.py

Centre for Digital Music, Queen Mary University of London.
Copyright (C) 2009 Gyorgy Fazekas, QMUL.
'''

import sys,numpy
from numpy import abs,log,exp,floor,sum,sqrt,cos,hstack
from numpy.fft import *

class melScaling(object):

	def __init__(self,sampleRate,inputSize,numBands,minHz = 0,maxHz = None):
		'''Initialise frequency warping and DCT matrix. 
		Parameters:
		sampleRate: audio sample rate
		inputSize: length of magnitude spectrum (half of FFT size assumed)
		numBands: number of mel Bands (MFCCs)
		minHz: lower bound of warping  (default = DC)
		maxHz: higher bound of warping (default = Nyquist frequency)
		'''
		self.sampleRate = sampleRate
		self.NqHz = sampleRate / 2.0
		self.minHz = minHz
		if maxHz is None : maxHz = self.NqHz
		self.maxHz = maxHz
		self.inputSize = inputSize
		self.numBands = numBands
		self.valid = False
		self.updated = False
		
	def update(self): 
		# make sure this will run only once 
		# if called from a vamp process
		if self.updated: return self.valid
		self.updated = True
		self.valid = False
		print 'Updating parameters and recalculating filters: '
		print 'Nyquist: ',self.NqHz
		
		if self.maxHz > self.NqHz : 
			raise Exception('Maximum frequency must be smaller than the Nyquist frequency')
		
		self.maxMel = 1000*log(1+self.maxHz/700.0)/log(1+1000.0/700.0)
		self.minMel = 1000*log(1+self.minHz/700.0)/log(1+1000.0/700.0)
		print 'minHz:%s\nmaxHz:%s\nminMel:%s\nmaxMel:%s\n' \
		%(self.minHz,self.maxHz,self.minMel,self.maxMel)
		self.filterMatrix = self.getFilterMatrix(self.inputSize,self.numBands)
		self.DCTMatrix = self.getDCTMatrix(self.numBands)
		self.filterIter = self.filterMatrix.__iter__()
		self.valid = True
		return self.valid
				
	def getFilterCentres(self,inputSize,numBands):
		'''Calculate Mel filter centres around FFT bins.
		This function calculates two extra bands at the edges for
		finding the starting and end point of the first and last 
		actual filters.'''
		centresMel = numpy.array(xrange(numBands+2)) * (self.maxMel-self.minMel)/(numBands+1) + self.minMel
		centresBin = numpy.floor(0.5 + 700.0*inputSize*(exp(centresMel*log(1+1000.0/700.0)/1000.0)-1)/self.NqHz)
		return numpy.array(centresBin,int)
		
	def getFilterMatrix(self,inputSize,numBands):
		'''Compose the Mel scaling matrix.'''
		filterMatrix = numpy.zeros((numBands,inputSize))
		self.filterCentres = self.getFilterCentres(inputSize,numBands)
		for i in xrange(numBands) :
			start,centre,end = self.filterCentres[i:i+3]
			self.setFilter(filterMatrix[i],start,centre,end)
		return filterMatrix.transpose()

	def setFilter(self,filt,filterStart,filterCentre,filterEnd):
		'''Calculate a single Mel filter.'''
		k1 = numpy.float32(filterCentre-filterStart)
		k2 = numpy.float32(filterEnd-filterCentre)
		up = (numpy.array(xrange(filterStart,filterCentre))-filterStart)/k1
		dn = (filterEnd-numpy.array(xrange(filterCentre,filterEnd)))/k2
		filt[filterStart:filterCentre] = up
		filt[filterCentre:filterEnd] = dn
						
	def warpSpectrum(self,magnitudeSpectrum):
		'''Compute the Mel scaled spectrum.'''
		return numpy.dot(magnitudeSpectrum,self.filterMatrix)
		
	def getDCTMatrix(self,size):
		'''Calculate the square DCT transform matrix. Results are 
		equivalent to Matlab dctmtx(n) with 64 bit precision.'''
		DCTmx = numpy.array(xrange(size),numpy.float64).repeat(size).reshape(size,size)
		DCTmxT = numpy.pi * (DCTmx.transpose()+0.5) / size
		DCTmxT = (1.0/sqrt( size / 2.0)) * cos(DCTmx * DCTmxT)
		DCTmxT[0] = DCTmxT[0] * (sqrt(2.0)/2.0)
		return DCTmxT
		
	def dct(self,data_matrix):
		'''Compute DCT of input matrix.'''
		return numpy.dot(self.DCTMatrix,data_matrix)
		
	def getMFCCs(self,warpedSpectrum,cn=True):
		'''Compute MFCC coefficients from Mel warped magnitude spectrum.'''
		mfccs=self.dct(numpy.log(warpedSpectrum))
		if cn is False : mfccs[0] = 0.0
		return mfccs

