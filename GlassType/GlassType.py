import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as pyplot
from pylab import *
from math import exp

target_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
filename = 'glass.data'

def download_data(url, filename):
	from urllib2 import urlopen

	with open(filename, 'wb') as f:
		response = urlopen(target_url)
		html = response.read()
		f.write(html)

glass = pd.read_csv(filename, header=None, prefix="V")
glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si',
				 'K', 'Ca', 'Ba', 'Fe', 'Type']

# print glass.head()
# print glass.tail()
# print glass.describe()

#normalize data with mean and sd
glasscols = len(glass.columns)
glassNormalized = glass.iloc[:, 1:glasscols]
ncols = len(glassNormalized.columns)
nrows = len(glassNormalized.values)
summary = glassNormalized.describe()

for i in range(ncols):
	#ignore last col which is type data
	mean = summary.iloc[1, i]
	sd = summary.iloc[2, i]

	glassNormalized.iloc[:, i:(i+1)] = (glassNormalized.iloc[:, i:(i+1)] - mean) / sd

def box_plot():
	array = glassNormalized.values
	boxplot(array)
	xlabel("Attribute Index")
	ylabel("Quartile Ranges - Normalized")
	show()

def data_vis():
	for i in range(nrows):
		dataRow = glassNormalized.iloc[i, 1:(ncols - 1)]
		labelColor = glassNormalized.iloc[i, (ncols - 1)]*7.0
		dataRow.plot(color=cm.RdYlBu(labelColor), alpha=0.5)

	xlabel("Attribute Index")
	ylabel("Attribute Values")
	show()

def heat_map():
	corMat = DataFrame(glassNormalized.iloc[:, :(ncols - 1)].corr())

	pcolor(corMat)
	show()


heat_map()
#data_vis()
#box_plot()
#download_data(target_url, filename)