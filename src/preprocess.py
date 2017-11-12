from reviews import *
size = 100
tagged_reviews = tagged_reviews[:size]
import numpy
import nltk
import cPickle as pkl

#from collections import OrderedDict

worddict = dict()

def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    # tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    # tok_text, _ = tokenizer.communicate(text)
    tok_text = nltk.tokenize.word_tokenize(text)
    #toks = tok_text.split('\n')[:-1]
    print 'Done.'

    return tok_text


def build_dict():
    global worddict
    sentences = []
    labels = []
    for tup in tagged_reviews:
        sentences.append(tup[1].strip())
        labels.append(tup[0])

    sentences = tokenize(sentences)

    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [worddict[w] if w in worddict else 1 for w in words]

    return labels, seqs



# def grab_data(dictionary):
#     sentences = []
#     labels = []
#     for tup in tagged_reviews:
#         sentences.append(tup[1].strip())
#         labels.append(tup[0])
#     sentences = tokenize(sentences)

#     seqs = [None] * len(sentences)
#     for idx, ss in enumerate(sentences):
#         words = ss.strip().lower().split()
#         seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

#     return labels, seqs


def main():
    global worddict
    #dictionary = build_dict()
    X, Y = build_dict()

    #X, Y = grab_data(dictionary)

    f = open('yelp.pkl', 'wb')
    pkl.dump((X[:size/2], Y[:size/2]), f, -1)
    pkl.dump((X[size/2:size], Y[size/2:size]    ), f, -1)

    #pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open('yelp.dict.pkl', 'wb')
    pkl.dump(worddict, f, -1)
    f.close()

if __name__ == '__main__':
    main()
