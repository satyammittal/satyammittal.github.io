import numpy as np
import cPickle
from collections import defaultdict
import sys, re
from reviews import tagged_reviews
import nltk
import gensim

size = 100
tagged_reviews = tagged_reviews[:size]

def build_data_cv(cv=10):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)
    for label, review in tagged_reviews:       
        rev = []
        rev.append(review.strip())
        words = nltk.word_tokenize(review)
        datum  = {"y": label, 
                  "text": review,                             
                  "num_words": len(words),
                  "split": np.random.randint(0, cv)}
        revs.append(datum)
        words = set(words)
        for word in words:
            vocab[word] += 1
        
    return revs, vocab
    
def get_W(word_vecs, k, vocab):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(vocab)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float64')            
    W[0] = np.zeros(k, dtype='float64')
    i = 1
    for word in vocab:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def create_bin_vec(vocab):
    
    sentences = []
    for label, review in tagged_reviews:
        review = nltk.word_tokenize(review)
        if len(review):
            sentences.append(review)

    model = gensim.models.Word2Vec(sentences, size=10, min_count=1)
    return model

def add_unknown_words(word_vecs, vocab, min_df=1, k=10):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  


if __name__== "__main__":    
    
    print "loading data...",        
    revs, vocab = build_data_cv(cv=10)
    print "data loaded!"
    
    print "loading word2vec vectors...",
    w2v = create_bin_vec(vocab)
    print "word2vec loaded!"

    W, word_idx_map = get_W(w2v, 10, vocab)
    # rand_vecs = {}
    # add_unknown_words(rand_vecs, vocab)
    # W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, word_idx_map, vocab], open("mr.p", "wb"))
    print "dataset created!"