import json
from nltk import tokenize
from random import shuffle

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

#import dataset
from reviews import *
size = 100000
tagged_reviews = tagged_reviews[:size]

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

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1.0, kernel='linear')
    svm.fit(X, y)
    return svm

shuffle(tagged_reviews)
X, y = create_tfidf_training_data(tagged_reviews)

# writing number of features from tf-idf to file 
with open("tfidf_features.py", "w") as f:
    f.write("tfidf_features = " + str(X.shape[1]))
svm = train_svm(X, y)

# saving model to file
from sklearn.externals import joblib
joblib.dump(svm, 'svm_model.pkl')
