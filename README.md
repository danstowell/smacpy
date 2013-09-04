
smacpy baseline - simple-minded audio classifier in python
==========================================================

Copyright (c) 2012 Dan Stowell and Queen Mary University of London
(incorporating code Copyright (c) 2009 Gyorgy Fazekas and Queen Mary University of London)
https://code.soundsoftware.ac.uk/projects/smacpy


  NOTE: THIS IS A **BASELINE** SYSTEM. Please label it as such in MIREX results!


This is a classifier that you can train on a set of labelled audio files, and then it predicts a label for further audio files.
It is designed with two main aims:

1. to provide a baseline against which to test more advanced audio classifiers;
2. to provide a simple code example of a classifier which people are free to build on.

It uses the very common workflow of taking audio, converting it frame-by-frame into MFCCs, and modelling the MFCC "bag of frames" with a GMM.

This code is customised for MIREX requirements, making the code a little more complex than the standard version.
(It's also in our "mirex2013" branch: <https://github.com/danstowell/smacpy/tree/mirex2013>)

Requirements
------------
* Python 2.7
* virtualenv <http://www.virtualenv.org/en/latest/>
    (which ensures the python dependencies are installed - numpy, audiolab and scikit-learn)

The script has been tested on python 2.7 (on ubuntu 11.10, 12.04 and 13.04). Not yet tested on python3 but may work.


Usage
-----
0. Before running for the first time, run the venv bootstrap script, telling it the path to a folder in which it can store python dependencies:

                python venv.py --system-site-packages ./venv

1. Run the feature extraction - NOTE that we're invoking the venv python not just "python":

	./venv/bin/python smacpy.py --trainlist /path/to/trainlist.txt --scratchdir /path/to/scratchdir

2. Run the training-and-classification - NOTE that we're invoking the venv python not just "python":

	./venv/bin/python smacpy.py --trainlist /path/to/trainlist.txt --scratchdir /path/to/scratchdir --testlist /path/to/testlist.txt --outlist /path/to/output.txt


Runtime
-------
It's simple and should be fast. On my laptop it took 5 minutes to process all the audio in our D-CASE challenge.

It only uses one core.



