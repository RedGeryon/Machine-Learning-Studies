from collections import OrderedDict
import sys
import numpy as np
import json
import pylab
import scipy.stats as stats

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
	"databases/undocumented/connectionist-bench/sonar/sonar.all-data")

data = open('sonar.all-data', 'r')	#data has been downloaded to file

##############
def data_inspection():
	xList = []
	labels = []

	for line in data:
		row = line.strip().split(",")
		xList.append(row)
	nrow = len(xList)
	ncol = len(xList[1])

	value_type = {'float': 0,
				  'string': 0,
				  'other': 0}
	
	colCounts = OrderedDict()

	i = 0
	for col in range(ncol):
		colCounts['col' + str(i)] = {'float': 0,
									  'string': 0,
									  'other': 0}
		for row in xList:
			try:
				value = float(row[col])
				value_type['float'] += 1
				colCounts['col' + str(i)]['float'] += 1
			except ValueError:
				if isinstance(row[col], str):
					value_type['string'] += 1
					colCounts['col' + str(i)]['string'] += 1
				else:
					value_type['other'] += 1
					colCounts[string(i)]['other'] += 1
		i += 1

	print('Col#', '\t', 'Num', '\t', 'Str', '\t', 'Other')
	for col in colCounts:
		print(col, '\t', colCounts[col]['float'], '\t', colCounts[col]['string'], '\t', colCounts[col]['other'])

def data_statistics():
	xList = []
	labels = []

	for line in data:
		row = line.strip().split(",")
		xList.append(row)
	nrow = len(xList)
	ncol = len(xList[1])

	#generate summary stats for 3 cols (e.g)
	col = 3
	colData = [float(row[col]) for row in xList]

	colArray = np.array(colData)		#turn arr into np arr (to use np methods)
	colMean = np.mean(colArray)
	colsd = np.std(colArray)

	print('Col3 Mean:', colMean)
	print('Col3 Std:', colsd)
	
	#calc quantile boundaries
	ntiles = 4
	percentBndry = []

	for i in range(ntiles+1):
		percentBndry.append(np.percentile(colArray, i*100/ntiles))

	print('Col3 Quantile Boundaries:', percentBndry)

	#for decile
	ntiles = 10
	percentBndry = []

	for i in range(ntiles+1):
		percentBndry.append(np.percentile(colArray, i*100/ntiles))

	print('Col3 Decile Boundaries:', percentBndry)
	print('\n')

	#last col, col60 has categorical vars
	col = 60
	colData = [row[col] for row in xList]

	unique = set(colData)
	print('Col60 Distinct categories:', list(unique))
	print('\n')

	#num of elements with each value
	catDict = dict(zip(list(unique), len(unique)*[0]))

	for cat in colData:
		catDict[cat] += 1

	print(catDict)

	#visualizing outliers (col3) with quantile-quantil plot (Q-Q)
	col = 3
	colData = [float(row[col]) for row in xList]

	stats.probplot(colData, dist="norm", plot=pylab)
	pylab.show()

#data_inspection()
data_statistics()