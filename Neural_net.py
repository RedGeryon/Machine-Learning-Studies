import numpy as np

class neuralnet(object):
	def __init__(self, sizes):
		self.sizes = sizes
		self.layers = len(sizes)
		self.biases = [np.random.randn(x,1) for x in sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

	def predict(self, a):
		a = np.array(a).transpose()
		for b, w in zip(self.biases, self.weights):
			a = self.sigmoid(np.dot(w, a) + b)
		return a

	def sigmoid(self, z, prime=False):
		sig = 1/(1+np.exp(-1*z))
		if prime:
			return sig*(1-sig)
		return sig

	def SGD(self, train_data, targets, mini_batch_size, epochs, eta):
		for j in range(epochs):
			shuffled_data = [(x,y) for x, y in zip(targets[0], train_data)]
			np.random.shuffle(shuffled_data)
			for i in range(0, len(shuffled_data), mini_batch_size):
				batch = shuffled_data[i: mini_batch_size + i]
				train = np.array([sample[1] for sample in batch])
				labels = np.array([[sample[0] for sample in batch]])
				dC_db, dC_dw = self.back_propagate(train, labels)
				self.biases = [b - (eta/len(batch)) * db for b, db in zip(self.biases, dC_db)]
				self.weights = [w - (eta/len(batch)) * dw for w, dw in zip(self.weights, dC_dw)]

	def back_propagate(self, train, labels):
		dC_db = [np.zeros(b.shape) for b in self.biases]	
		dC_dw = [np.zeros(w.shape) for w in self.weights]
		activation = train.transpose()
		activation_layers = [activation]
		Zs = []

		#feedfoward
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			Zs.append(z)
			activation = self.sigmoid(z)
			activation_layers.append(activation)

		error = (activation_layers[-1] - labels) * self.sigmoid(Zs[-1], True)
		dC_db[-1] = error
		dC_dw[-1] = np.dot(error, activation_layers[-2].transpose())

		for layer in range(2, self.layers):
			z = Zs[-layer]
			error = np.dot(self.weights[-layer+1].transpose(), error) * self.sigmoid(z, True)
			dC_db[-layer] = error
			dC_dw[-layer] = np.dot(error, activation_layers[-layer-1].transpose())

		return dC_db, dC_dw

breast_cancer_data = np.recfromcsv('wdbc.data', delimiter=',')

categories = ('ID',
			  'Diagnosis',
			  'm_radius', 'sd_radius', 'lrg_radius', 
			  'm_texture', 'sd_texture', 'lrg_texture',
			  'm_perimeter', 'sd_perimeter', 'lrg_perimeter',
			  'm_area', 'sd_area', 'lrg_area', 
			  'm_smoothness', 'sd_smoothness', 'lrg_smoothness', 
			  'm_compactness', 'sd_compactness', 'lrg_compactness', 
			  'm_concavity', 'sd_concavity', 'lrg_concavity', 
			  'm_concave_points', 'sd_concave_points', 'lrg_concave_points', 
			  'm_symmetry', 'sd_symmetry', 'lrg_symmetry', 
			  'm_fractal_dim', 'sd_fractal_dim', 'lrg_fractal_dim')


data = [list(sample)[1:] for sample in breast_cancer_data]


for i in range(10):
	np.random.shuffle(data)
	percent_num = 56
	test_data = data[:56]
	learn_data = data[56:]

	data_inputs = [sample[1:] for sample in learn_data]
	labels = [[1 if sample[0] == "M" else 0 for sample in learn_data]]

	test_inputs = [sample[1:] for sample in test_data]
	test_labels = [[1 if sample[0] == "M" else 0 for sample in test_data]]


	cancer_predict = neuralnet([30, 200, 100, 50, 25, 25, 1])
	#SGD(self, train_data, targets, mini_batch_size, epochs, eta)
	cancer_predict.SGD(data_inputs, labels, 128, 1000, .1)

	correct = 0
	n = 0

	print len(test_inputs[0])
	for j in range(len(test_inputs)):
		print len(cancer_predict.predict([test_inputs[j]])[0])
		if cancer_predict.predict([test_inputs[j]])[0][0] > .5:
			if test_labels[0][j] == 1:
				correct +=1
		else:
			if test_labels[0][j] == 0:
				correct += 1
		n+=1

	print correct*1./n
	exit()
print correct, n