import pandas as pd
import matplotlib.pyplot as plot
import pylab as pl
from pandas import DataFrame
import random
from math import sqrt
from sklearn import datasets, linear_model
import numpy
from sklearn.metrics import roc_curve, auc

rocksVMines = pd.read_csv('sonar.all-data', header=None, prefix="V")	#data has been downloaded to file

def data_extract():
	print(rocksVMines.head())
	print(rocksVMines.tail())
	print(rocksVMines.describe())

def data_visualization():
	#parallel coordinates plot

	for i in range(208):
		#assign colors based on "M" or "R"
		if rocksVMines.iat[i,60] == "M":
			pcolor = "red"
		else:
			pcolor = "blue"

		dataRow = rocksVMines.iloc[i,0:60]
		dataRow.plot(color=pcolor)

	plot.xlabel("Attribute Index")
	plot.ylabel("Attribute Values")
	plot.show()

def visualizing_interrelationships():
	#calculate correlations between real-valued attributes
	dataRow2 = rocksVMines.iloc[1,0:60]
	dataRow3 = rocksVMines.iloc[2,0:60]

	plot.scatter(dataRow2, dataRow3)
	plot.xlabel("2nd Attribute")
	plot.ylabel("3rd Attribute")
	plot.show()

	dataRow21 = rocksVMines.iloc[20,0:60]

	plot.scatter(dataRow2, dataRow21)
	plot.xlabel("2nd Attribute")
	plot.ylabel("21st Attribute")
	plot.show()

def classification_target_attribute_corr():
	target = []
	for i in range(208):
		#assign 0 or 1 target value based on "M" or "R" labels
		if rocksVMines.iat[i,60] == "M":
			target.append(1.0)
		else:
			target.append(0.0)

	#plot 35th attribute
	dataRow = rocksVMines.iloc[0:208,35]
	plot.scatter(dataRow, target)

	plot.xlabel("Attribute value")
	plot.ylabel("Target Value")
	plot.show()

	#This version dithers points and makes them transparent to improve visualization
	target = []
	for i in range(208):
		if rocksVMines.iat[i,60] == "M":
			#adds random value with uniform dist
			target.append(1.0 + random.uniform(-0.1, 0.1))
		else:
			target.append(0.0 + random.uniform(-0.1, 0.1))

	dataRow = rocksVMines.iloc[0:208,35]
	plot.scatter(dataRow, target, alpha=0.5, s=120)

	plot.xlabel("Attribute Value")
	plot.ylabel("Target Value")
	plot.show()

def pearson_correlation():
	dataRow2 = rocksVMines.iloc[1,0:60]
	dataRow3 = rocksVMines.iloc[2,0:60]
	dataRow21 = rocksVMines.iloc[20,0:60]

	mean2 = 0.0; mean3 = 0.0; mean21 = 0.0
	numElt = len(dataRow2)

	for i in range(numElt):
		mean2 += dataRow2[i]/numElt
		mean3 += dataRow3[i]/numElt
		mean21 += dataRow21[i]/numElt

	var2 = 0.0; var3 = 0.0; var21 = 0.0
	for i in range(numElt):
		var2 += (dataRow2[i] - mean2) * (dataRow2[i] - mean2)/numElt
		var3 += (dataRow3[i] - mean3) * (dataRow3[i] - mean3)/numElt
		var21 += (dataRow21[i] - mean21) * (dataRow21[i] - mean21)/numElt

	corr23 = 0.0; corr221 = 0.0
	for i in range(numElt):

		corr23 += (dataRow2[i] - mean2) * \
					(dataRow3[i] - mean3) / (sqrt(var2*var3) * numElt)

		corr221 += (dataRow2[i] - mean2) * \
					(dataRow21[i] - mean21) / (sqrt(var2*var21) * numElt)

	print('Correlation btw attr 2 & 3:', corr23)
	print('Correlation btw attr 2 & 21:', corr221)

def attr_label_vis_heat_map():
	corMat =DataFrame(rocksVMines.corr())

	plot.pcolor(corMat)
	plot.show()

def confusionMatrix(predicted, actual, threshold):
	if len(predicted) != len(actual): return -1
	tp = 0.0
	fp = 0.0
	tn = 0.0
	fn = 0.0

	for i in range(len(actual)):
		if actual[i] > 0.5:
			if predicted[i] > threshold:
				tp += 1.0
			else:
				fn += 1.0
		else:
			if predicted[i] < threshold:
				tn += 1.0
			else:
				fp += 1.0

	rtn = [tp, fn, fp, tn]
	return rtn

def train_data():
	rocksVMines = open('sonar.all-data', 'r')

	xList = []
	labels = []
	for line in rocksVMines:
		row = line.strip().split(",")
		if(row[-1] == "M"):
			labels.append(1.0)
		else:
			labels.append(0.0)
		row.pop()
		floatRow = [float(num) for num in row]
		xList.append(floatRow)

	indices = range(len(xList))
	xListTest = [xList[i] for i in indices if i%3 == 0]
	xListTrain = [xList[i] for i in indices if i%3 != 0]
	labelsTest = [labels[i] for i in indices if i%3 == 0]
	labelsTrain = [labels[i] for i in indices if i%3 != 0]

	xTrain = numpy.array(xListTrain); yTrain = numpy.array(labelsTrain)
	xTest = numpy.array(xListTest); yTest = numpy.array(labelsTest)

	print("Shape of xTrain array", xTrain.shape)
	print("Shape of yTrain array", yTrain.shape)

	print("Shape of xTest array", xTest.shape)
	print("Shape of yTest array", yTest.shape)

	rocksVMinesModel = linear_model.LinearRegression()
	rocksVMinesModel.fit(xTrain, yTrain)

	trainingPredictions = rocksVMinesModel.predict(xTrain)
	print("Some values predicted by modle", trainingPredictions[0:5],
		trainingPredictions[-6:-1])

	confusionMatTrain = confusionMatrix(trainingPredictions, yTrain, 0.5)
	tp = confusionMatTrain[0]; fn = confusionMatTrain[1]
	fp = confusionMatTrain[2]; tn = confusionMatTrain[3]
	print("tp = " + str(tp) + "\tfn = " + str(fn) + "\n" + "fp = " + str(fp) + "\ttn = " + str(tn) + '\n')
	
	testPredictions = rocksVMinesModel.predict(xTest)

	conMatTest = confusionMatrix(testPredictions, yTest, 0.5)
	tp = conMatTest[0]; fn = conMatTest[1]
	fp = conMatTest[2]; tn = conMatTest[3]
	print("tp = " + str(tp) + "\tfn = " + str(fn) + "\n" + "fp = " + str(fp) + "\ttn = " + str(tn) + '\n')


	fpr, tpr, thresholds = roc_curve(yTrain, trainingPredictions)
	roc_auc = auc(fpr, tpr)
	print('AUC for in-sample ROC curve: %f' % roc_auc)

	pl.clf()
	pl.plot(fpr, tpr, label = "ROC curve (area = %0.2f)" % roc_auc)
	pl.plot([0,1], [0,1], 'k-')
	pl.xlim([0.0, 1.0])
	pl.ylim([0.0, 1.0])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.title('In sample ROC rocks versus mines')
	pl.legend(loc = "lower right")
	pl.show()

	fpr, tpr, thresholds = roc_curve(yTest, testPredictions)
	roc_auc = auc(fpr, tpr)

	pl.clf()
	pl.plot(fpr, tpr, label = "ROC curve (area = %0.2f)" % roc_auc)
	pl.plot([0,1], [0,1], 'k-')
	pl.xlim([0.0, 1.0])
	pl.ylim([0.0, 1.0])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.title('In sample ROC rocks versus mines')
	pl.legend(loc = "lower right")
	pl.show()

train_data()
#data_extract()
#data_visualization()
#visualizing_interrelationships()
#classification_target_attribute_corr()
#pearson_correlation()
#attr_label_vis_heat_map()