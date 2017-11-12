#import dataset
from reviews import *

import gensim
from random import shuffle
import nltk
import numpy as np
#shuffle(tagged_reviews)

size = 100000

tagged_reviews = tagged_reviews[:size]
sentences = []
for label, review in tagged_reviews:
	review = nltk.word_tokenize(review)
	if len(review):
		sentences.append(review)

model = gensim.models.Word2Vec(sentences, min_count=40, window = 10)

final = []

for label, review in tagged_reviews:
	review = nltk.word_tokenize(review)
	review_vector = np.array([0.0 for i in range(100)])
	if not len(review):
		continue
	for word in review:
		try:
			review_vector += model[word]
		except:
			pass
	review_vector = review_vector/len(review)
	final.append((label, list(review_vector)))

print "Final ban gya"

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from sklearn.externals import joblib
# X = []
# y = []
# for label, review in final:
# 	if label:
# 		y.append([0,1])
# 	else:
# 		y.append([1,0])
# 	X.append(review)

# from lasagne import layers
# from lasagne.updates import nesterov_momentum
# from nolearn.lasagne import NeuralNet

# net1 = NeuralNet(
#     layers=[  # three layers: one hidden layer
#         ('input', layers.InputLayer),
#         ('hidden', layers.DenseLayer),
#         ('output', layers.DenseLayer),
#         ],
#     # layer parameters:
#     input_shape=(None, 100),  # 96x96 input pixels per batch
#     hidden_num_units=10,  # number of units in hidden layer
#     output_nonlinearity=None,  # output layer uses identity function
#     output_num_units=2,  # 30 target values

#     # optimization method:
#     update=nesterov_momentum,
#     update_learning_rate=0.01,
#     update_momentum=0.9,

#     regression=True,  # flag to indicate we're dealing with regression problem
#     max_epochs=100,  # we want to train this many epochs
#     verbose=1,
#     )


# X = np.asarray(X)
# X = X.astype(np.float32)
# y = np.asarray(y)
# y = y.astype(np.float32)
# net1.fit(X, y)

alldata = ClassificationDataSet(len(final[0][1]), 1, nb_classes=2)
for i, tup in enumerate(final):
    alldata.addSample(tup[1], tup[0])

tstdata, trndata = alldata.splitWithProportion(0.60)

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

fnn = buildNetwork( trndata.indim, 10, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
trainer.trainUntilConvergence(maxEpochs = 10)

joblib.dump(fnn, "fnn_model.pkl")