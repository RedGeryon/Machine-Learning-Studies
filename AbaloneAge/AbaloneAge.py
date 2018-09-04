import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
from pylab import *
from math import exp

target_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'

# with open('abalone.data', 'wb') as f:
# 	response = urllib2.urlopen(target_url)
# 	html = response.read()
# 	f.write(html)

abalone = pd.read_csv('abalone.data', header=None, prefix='V')
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height',
				   'Whole Wt', 'Shucked Wt', 'Viscera Wt',
				   'Shell Wt', 'Rings']
# Sex, Length, Diameter, Height, Whole Weight, Shucked Weight, Viscera Weight, Shell Weight, Rings

# print abalone.tail()
# print abalone.head()
# print abalone.describe()

#normalize col9 (rings)
summary = abalone.describe()
abaloneNormalized = abalone.iloc[:,1:9]

#ring stats for normalization with mean and sd
#then compress using logit function
nrows = len(abalone.index)
min_rings = summary.iloc[3,7]
max_rings = summary.iloc[7,7]
mean_rings = summary.iloc[1,7]
sd_rings = summary.iloc[2,7]

for i in range(8):
	mean = summary.iloc[1, i]
	sd = summary.iloc[2, i]

	abaloneNormalized.iloc[:,i:(i+1)] = (abaloneNormalized.iloc[:,i:(i+1)] - mean) / sd

def data_vis():
	for i in range(nrows):
		#ignore last row
		#normalize and compress
		dataRow = abalone.iloc[i,1:8]
		normTarget = (abalone.iloc[i,8] - mean_rings) / sd_rings
		labelColor = 1.0/(1.0 + exp(-normTarget))
		dataRow.plot(color=cm.RdYlBu(labelColor), alpha=0.5)

	xlabel("Attribute Index")
	ylabel("Attribute Values")
	show()

def attr_label_heat_map():
	corMat = DataFrame(abalone.iloc[:,1:9].corr())

	pcolor(corMat)
	show()

def box_plot():
	#plot non-normalized data with col 8 ignored
	array = abalone.iloc[:,1:8].values
	boxplot(array)
	xlabel("Attribute Index")
	ylabel("Quartile Ranges")
	show()

	#plot normalized data
	array2 = abaloneNormalized.iloc[:,1:8].values
	boxplot(array2)
	xlabel("Attribute Index")
	ylabel("Quartile Ranges")
	show()

data_vis()