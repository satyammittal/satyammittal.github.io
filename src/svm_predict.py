import json
from nltk import tokenize
from random import shuffle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import numpy
import scipy.sparse as sps
# import dataset
from reviews import *

# get the number of features per review from tfidf 
from tfidf_features import tfidf_features

def create_tfidf_training_data(docs):
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list. 

    The function returns both the class label vector (y) and 
    the corpus token/feature matrix (X).
    """
    # Create the training data class labels
    y = [d[0] for d in docs]
    
    # Create the document corpus list
    corpus = [d[1] for d in docs]


    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(min_df=1)
    
    X = vectorizer.fit_transform(corpus)
    
    return X, y

tagged_reviews = tagged_reviews[10000:10150]

shuffle(tagged_reviews)
X, y = create_tfidf_training_data(tagged_reviews)

X = sps.csr_matrix((X.data, X.indices, X.indptr), shape=(len(tagged_reviews), tfidf_features))


# load the model
from sklearn.externals import joblib
svm = joblib.load('svm_model.pkl')

pred = svm.predict(X)
overall_score = svm.score(X, y)

print "Overall Accuracy:",
print round(overall_score, 2)

