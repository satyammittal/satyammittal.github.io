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

model = gensim.models.Word2Vec(sentences, min_count=40)

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

print "word_vectors =",
print final