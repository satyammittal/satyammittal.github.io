from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from reviews import *
import nltk
import gensim
import numpy as np


sentences = []
tagged_reviews = tagged_reviews[:10]
for label, review in tagged_reviews:
	review = nltk.word_tokenize(review)
	if len(review):
		sentences.append(review)

model = gensim.models.Word2Vec(sentences, size=10, min_count=1)
pad = [0.0 for i in range(10)]
X = []
y = []
for label, review in tagged_reviews:
	review = nltk.word_tokenize(review)
	if label:
		y.append([0, 1])
	else:
		y.append([1, 0])
	review_vector = []
	count = 0
	for word in review:
		if count < 150:
			try:
				review_vector += list(model[word])
			except:
				review_vector += pad
		else:
			break
		count += 1
	while count < 150:
		review_vector += pad
		count += 1
	X.append(review_vector)

X = np.asarray(X)
X.astype(np.float32)
y = np.asarray(y)
y.astype(np.float32)
X = X.reshape(-1, 1, 10, 150)

cnn = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 10, 150),
    conv1_num_filters=2, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=3, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=4, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=100, hidden5_num_units=100,
    output_num_units=2, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=100,
    verbose=1,
    )

cnn.fit(X,y)