
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
    (we use a virtualenv script to ensure the following python dependencies are installed:
       numpy, audiolab and scikit-learn)

The script has been tested on python 2.7 (on ubuntu 11.10, 12.04 and 13.04). Not yet tested on python3 but may work.


Usage
-----
0. Before running for the first time, run the venv bootstrap script, telling it the path to a folder in which it can store python dependencies.
   In the following example I've put it in a local folder ./venv but you may want to put it in the scratch folder:

           python venv.py --system-site-packages ./venv

1. Run the feature extraction - NOTE that we're invoking the venv python not just "python":

           /venv/bin/python smacpy.py -q --precalclist /path/to/precalclist.txt --scratchdir /path/to/scratchdir

2. Run the training-and-classification - NOTE that we're invoking the venv python not just "python":

           /venv/bin/python smacpy.py -q --trainlist /path/to/trainlist.txt --scratchdir /path/to/scratchdir --testlist /path/to/testlist.txt --outlist /path/to/output.txt

If you have any problems with running steps 1 or 2, please run them again without the "-q" (quiet) option, and send me the output.


Runtime
-------
It's simple and should be fast. It only uses one core.

Runtime and disk usage stats, measured on a thinkpad laptop:

* Input data:
    769 x 10-second 44 kHz mono WAV (total 128 minutes of audio)
    10 class-labels
* Feature extraction:
    27 seconds, i.e. around 260x realtime   (created a file 33 MB)
* Training-and-classification:
    180 seconds, i.e. around 42x realtime

