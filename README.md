
smacpy - simple-minded audio classifier in python
=================================================

Copyright (c) 2012 Dan Stowell and Queen Mary University of London
(incorporating code Copyright (c) 2009 Gyorgy Fazekas and Queen Mary University of London)
- for licence information see the file named COPYING.

This is a classifier that you can train on a set of labelled audio files, and then it predicts a label for further audio files.
It is designed with two main aims:

1. to provide a baseline against which to test more advanced audio classifiers;
2. to provide a simple code example of a classifier which people are free to build on.

It uses the very common workflow of taking audio, converting it frame-by-frame into MFCCs, and modelling the MFCC "bag of frames" with a GMM.

Requirements
------------
* Python 2.7 or later (it uses the 'argparse' module, not available earlier)
* Python modules:
    * numpy
    * [scikits.audiolab](http://pypi.python.org/pypi/scikits.audiolab)
    * [sckikit-learn](http://scikit-learn.sourceforge.net/)

It has been tested on python 2.7 (on ubuntu 11.10 and 12.04). Not yet tested on python3 but it should be fine...


Usage example 1: commandline
-------------
If you invoke the script from the commandline (e.g. "python smacpy.py") it will assume there is a folder called "wavs"
and inside that folder are multiple WAV files, each of which has an underscore in the filename,
and the class label is the text BEFORE the underscore.
It will train a model using the wavs, and then test it on the same wavs (dividing the collection up so it can do a "crossvalidated" test).

To train and test on different folders, you can run it like this:

	python smacpy.py -t trainwavs -T testwavs


Usage example 2: from your own code
-------------
In this hypothetical example we train on four audio files, labelled as either 'usa' or 'uk', and then test on a separate audio file of someone called hubert:

	from smacpy import Smacpy
	model = Smacpy("wavs/training", {'karen01.wav':'usa', 'john01.wav':'uk', 'steve02.wav':'usa', 'joe03.wav':'uk'})
	model.classify('wavs/testing/hubert01.wav')

