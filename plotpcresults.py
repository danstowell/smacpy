#!/bin/env python

# plots data from files such as pcresults.csv

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inpath', default='output/pcresults.csv', help="Path to the CSV file to load")
args = vars(parser.parse_args())

rdr = csv.DictReader(file(args['inpath'], 'rb'))
data = []
for row in rdr:
	data.append({k:float(v) for k,v in row.items()})

minpc_inuse = list(set([row['minpc'] for row in data]))
minpc_inuse.sort()
maxpc_inuse = list(set([row['maxpc'] for row in data]))
maxpc_inuse.sort()

acc_zmap = np.zeros((len(minpc_inuse), len(maxpc_inuse)))
for row in data:
	acc_zmap[minpc_inuse.index(row['minpc']) , maxpc_inuse.index(row['maxpc'])] = row['acc']
#print(acc_zmap)

############################################
# plot simple scatter of pcrange vs accuracy

plt.figure()
plt.errorbar(
	[row['pcrange'] for row in data],
	[row['acc'] for row in data],
	[row['acc_ci'] for row in data],
	fmt='x',
	)
plt.xlabel("Percentile range")
plt.ylabel("Accuracy")
plt.title(args['inpath'])
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig("%s.PvA.pdf" % args['inpath'], format='pdf')

############################################
# plot something more heatmap-like
plt.figure()
plt.imshow(
	acc_zmap.T,
	cmap=cm.gist_heat, #cm.bone,
	interpolation='nearest',
	origin='lower',
	)
plt.xlabel("Min percentile")
plt.ylabel("Max percentile")
plt.xticks(range(len(minpc_inuse)), minpc_inuse)
plt.yticks(range(len(maxpc_inuse)), maxpc_inuse)
plt.title(args['inpath'])
plt.savefig("%s.heatmap.pdf" % args['inpath'], format='pdf')

# plot contour version
plt.figure()
plt.contourf(
	minpc_inuse,
	maxpc_inuse,
	acc_zmap.T,
	cmap=cm.gist_heat, #cm.bone,
	)
CS = plt.contour(
	minpc_inuse,
	maxpc_inuse,
	acc_zmap.T,
	)
plt.clabel(CS, fontsize=9, inline=1)
plt.xlabel("Min percentile")
plt.ylabel("Max percentile")
plt.title(args['inpath'])
plt.savefig("%s.contour.pdf" % args['inpath'], format='pdf')

