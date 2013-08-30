
smacpy - simple-minded audio classifier in python
=================================================

Copyright (c) 2012 Dan Stowell and Queen Mary University of London
(incorporating code Copyright (c) 2009 Gyorgy Fazekas and Queen Mary University of London)
- for licence information see the file named COPYING.
https://code.soundsoftware.ac.uk/projects/smacpy

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


Usage
-----
This branched version of the code takes arguments and file formats as required by MIREX classification task:

	python smacpy.py --trainlist trainlist.example.txt --testlist testlist.example.txt --outlist output.txt



Runtime
-------
It's simple and should be fast. On my laptop it took 5 minutes to process all the audio in our D-CASE challenge.

It only uses one core.

Nothing is written to disk except the answers.







