#!/usr/bin/env python

"""
script to run smacpy classification tests over the freefield1010 dataset, classifying presence of tags.
"""

from smacpy import Smacpy
from glob import iglob
import os.path
import json

################################
# user settings:
ffpath = '/home/dan/freefield1010_v02'
outpath = 'output'
picklepath = 'ff1010.pickle'
numfolds = 10
tagstoclassify = [u'voice', u'city'  , u"nature", u"birdsong", u"water", u"train", u"people"]
doclassif = True   # just so you can disable the heavy bit if needed
trainsubsample = 0.2

################################

# glob for the wavs, loading them into per-chunk subsections
data = [[] for _ in range(numfolds)]
numtotal = 0
precalcwavlist = []
for globbed in iglob("%s/*/*.wav" % ffpath):
	relpath = os.path.relpath(globbed, ffpath)
	globbed = relpath.split("/")
	foldlabel = globbed[0]
	whichfold = int(foldlabel)-1
	itemid = int(globbed[1].split(".")[0])
	# foreach wav: load its json metadata. also add an empty-array "tagpresent" key
	jsonfp = open("%s/%s/%i.json" % (ffpath, foldlabel, itemid), 'r')
	jsondata = json.load(jsonfp)
	jsonfp.close()
	item = {'itemid':itemid, 'whichfold':whichfold, 'tagpresent':{}, 'jsondata':jsondata, 'relpath':relpath}
	#print(item)
	data[whichfold].append(item)
	precalcwavlist.append(relpath)
	numtotal += 1
print "Loaded info about %i items" % numtotal

# feature precalc:
if not os.path.exists(picklepath):
	print("FEATURE PRECALC")
	precalcwavlist.sort()
	smaccer = Smacpy(ffpath, None, None)
	smaccer.precalc_features(ffpath, precalcwavlist, picklepath)

# now the classification experiments
csvout = open("%s/freefield1010smacpy.csv" % outpath, 'w', 1)
csvout.write(','.join(['tag','fold','tp','fp','tn','fn','acc','f']) + '\n')
tagsummaryout = open("%s/freefield1010smacpy_tagsummary.csv" % outpath, 'w', 1)
tagsummaryout.write(','.join(['tag','numpos', 'ratio']) + '\n')
for curtag in tagstoclassify:
	print("-------------------------------------------------")
	print("Running test with tag '%s'" % curtag)
	# iterate through, detect tag presence/absence, store in boolean   ['tagpresent'][curtag]
	numpositive = 0
	for datachunk in data:
		for item in datachunk:
			if curtag in item['jsondata'][u'tags']:
				gotit = True
			else:
				gotit = False
			item['tagpresent'][curtag] = gotit
			if gotit:
				numpositive += 1
	# report what proportion of files has the tag
	print("tag present in %i/%i items" % (numpositive, numtotal))
	tagsummaryout.write(','.join(map(str, [curtag, numpositive, numpositive/float(numtotal)])) + '\n')
	for whichfold in range(numfolds):
		trainingset = {}
		testingset  = {}
		for whichchunk, datachunk in enumerate(data):
			for item in datachunk:
				if whichchunk==whichfold:
					testingset[ item['relpath']] = item['tagpresent'][curtag]
				else:
					trainingset[item['relpath']] = item['tagpresent'][curtag]

		if doclassif:
			model = Smacpy(ffpath, trainingset, picklepath, trainsubsample)
			numtp = 0
			numfp = 0
			numtn = 0
			numfn = 0
			for itempath, gt in testingset.items():
				decision = model.classify(ffpath, itempath, maxap=False)
				if gt:
					if decision:
						numtp += 1
					else:
						numfn += 1
				else:
					if decision:
						numfp += 1
					else:
						numtn += 1

			acc = float(numtp + numtn) / (numtp + numtn + numfp + numfn)
			f   = (2. * numtp) / ((2. * numtp) + numfn + numfp)

			print("Fold %i: TP %i, FP %i, TN %i, FN %i" % (whichfold, numtp, numfp, numtn, numfn))

			csvout.write(','.join(map(str, [curtag, whichfold, numtp, numfp, numtn, numfn, acc, f])) + '\n')


csvout.close()
tagsummaryout.close()

