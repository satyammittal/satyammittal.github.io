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

    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(min_df=1)
    
    X = vectorizer.fit_transform(docs)
    
    return X

reviews = ["Polite but incompetent basically sums up the service department. I made the terrible mistake to lease a vehicle for the first time and it's not the lease I regret, it's where I leased from: Infiniti on Camelback aka hell on earth. \n\nMy brand new car immediately broke and was without AC in the dead of summer which resulted in my car being at the dealership for three weeks! Both (the time frame for fixing and the breaking of the AC itself) were \"flukes\" and \"not typical\" of infiniti, yeah ok. Fine. They fixed it, so I thought! \n\nThe dual temperature feature on my car no longer worked. 60 degrees and 79 degrees are the same temperature. Any range in between is the same as well. It's either just cold or above 80 degrees. I explain this, am told it will be taken care of then to my surprise I go to get my car and they say everything is fine. They did nothing. \n\nI explain to the random service guy what the now unfixed issue is, he sits in the car twirls some knobs and confirms what I'm saying. He then goes to the dipshit manager who can't wrap his head around the issue and says ,\"why would you have the AC on of its set at 89 degrees?\"  \n\nI admit at this point I wanted to smack his pensive face and needed to remove myself from the irritating scenario.  It was evident that his intellectual shortcomings were not going to change today so I again left with a partially working overpriced car. \n\nInfinities are nice cars, just not mine. \nAvoid Infiniti on camelback if you want a positive experience.\n\np.s. They claimed that they rotated my tires yet the service light still remains on, they can't even handle that.", "I had an overall great experience at this dealer's service department.  Some back story, I own a 2008 g35x with an ally premium warranty.\n\nTook my car to the dealer due to wheel noise and rough idle. The vehicle had both front wheel bearings and both valve covers replaced within my warranty.  The dealership handled the legwork with the insurance company and set me up with a 2015 qx60 loan vehicle.\n\nThe repairs were quick, Shelby in particular was professional and friendly, and most importantly the overall experience from driving my car into the service department to leaving with my car was simple and easy.\n\n5 out of 5 stars for sure, I will get all of my service done here from now on, and possibly a new q50 when the time comes.", "I have to honest and update: \nI still regret leasing a car and my AC still isn't fully operational, that being said Shelby in the service department is impossible to stay angry with and the service department does do a good job (with the exception of AC)."]

X = create_tfidf_training_data(reviews)

X = sps.csr_matrix((X.data, X.indices, X.indptr), shape=(len(reviews), tfidf_features))

# load the model
from sklearn.externals import joblib
svm = joblib.load('svm_model.pkl')

pred = svm.predict(X)

for p in pred:
    if p:
        print "Funny"
    else:
        print "Not Funny"

