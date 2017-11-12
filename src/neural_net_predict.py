from sklearn.externals import joblib
fnn = joblib.load('fnn_model.pkl')

#import dataset
from reviews import *

import gensim
from random import shuffle
import nltk
import numpy as np
#shuffle(tagged_reviews)


tagged_reviews = tagged_reviews[0:100000]
sentences = []
i = 0
for label, review in tagged_reviews:
	review = nltk.word_tokenize(review)
	if len(review):
		sentences.append(review)
		

model = gensim.models.Word2Vec(sentences, min_count=1)

final = []

for label, review in tagged_reviews:
	review = nltk.word_tokenize(review)
	review_vector = np.array([0.0 for i in range(100)])
	if not len(review):
		continue
	for word in review:
		review_vector += model[word]
	review_vector = review_vector/len(review)
	final.append((label, list(review_vector)))
		#print final
count = 0
#print final
for i in final:
	result = fnn.activate(i[1])
	if result[0] < 0.5:
		count+=1

print count