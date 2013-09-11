#!/usr/bin/env python

import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import csv
import numpy as np

##############################
infile = 'output/freefield1010smacpy_fromoctave_20130904.csv'
outfile = 'output/plotff1010smacpyclassif.pdf'

##############################

# load the csv, getting "auc" clumped by "tag"
rdr = csv.DictReader(open(infile, 'r'))
data = {}
keys = []
for row in rdr:
	tag = row['tag']
	val = float(row['auc'])
	if tag not in data:
		data[tag] = {'vals':[]}
		keys.append(tag)
	data[tag]['vals'].append(val)

# calc the mean and CI for each one
for key, stuff in data.items():
	stuff['mean'] = np.mean(stuff['vals'])
	stuff['ci'] = np.std(stuff['vals']) * 1.96 / np.sqrt(len(stuff['vals']))

# sort in ascending order of score
keys.sort(cmp=lambda a,b: cmp(data[a]['mean'], data[b]['mean']))

# plot
plt.figure()

plt.errorbar(range(len(keys)),
             [data[key]['mean'] for key in keys],
             [data[key]['ci']   for key in keys],
             fmt='.')
plt.xticks(range(len(keys)), keys, rotation=17)
plt.xlim(-0.5, len(keys) - 0.5)
plt.ylim(0.5, 1.0)
yticks = np.arange(0.5, 1.01, 0.1)
plt.yticks(yticks, [int(np.ceil(ytick * 100)) for ytick in yticks])
plt.ylabel("AUC score (%)")

plt.savefig(outfile, papertype='A4', format='pdf')

