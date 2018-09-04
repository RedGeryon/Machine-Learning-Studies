import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
from pylab import *
from math import exp


target_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
filename = 'winequality-red.csv'

def download_data(url, filename):
	from urllib2 import urlopen

	with open(filename, 'wb') as f:
		response = urlopen(target_url)
		html = response.read()
		f.write(html)

wine = pd.read_csv(filename, header=0, sep=';')

# print wine.head()
# print wine.tail()
# print wine.describe()

#normalize data with mean and sd
summary = wine.describe()
wineNormalized = wine
ncols = len(wine.columns)
nrows = len(wine.values)

for i in range(ncols):
	mean = summary.iloc[1, i]
	sd = summary.iloc[2, i]

	wineNormalized.iloc[:, i:(i+1)] = (wineNormalized.iloc[:, i:(i+1)] - mean) / sd

def box_plot():
	array = wineNormalized.values
	boxplot(array)
	xlabel("Attribute Index")
	ylabel("Quartile Ranges - Normalized")
	show()

def data_vis():
	for i in range(nrows):
		dataRow = wineNormalized.iloc[i, 1:(ncols - 1)]
		normTarget = wineNormalized.iloc[i, (ncols -1)]
		labelColor = 1.0/(1.0 + exp(-normTarget))
		dataRow.plot(color=cm.RdYlBu(labelColor), alpha=0.5)

	xlabel("Attribute Index")
	ylabel("Attribute Values")
	show()

def heat_map():
	corMat = DataFrame(wine.iloc[:, :(ncols - 1)].corr())

	pcolor(corMat)
	show()


#heat_map()
data_vis()
#box_plot()
#download_data(target_url, filename)