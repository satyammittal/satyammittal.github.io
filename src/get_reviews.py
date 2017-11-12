import json
from nltk import tokenize
from random import shuffle

count_f = 0
count_n = 0
tagged_reviews = []
min_votes = 4
threshold = 50000

for line in open('../dataset/yelp_academic_dataset_review.json', 'r'):
	review = json.loads(line)
	if review['votes']['funny'] >= min_votes  and count_f < threshold:
	    count_f += 1
	    tagged_reviews.append((1, review['text']))
	elif review['votes']['funny'] < min_votes and count_n < threshold:
		count_n += 1
		tagged_reviews.append((0, review['text']))
	if count_f == threshold and count_n == threshold:
		break

with open("reviews.py", "w") as f:
    f.write("tagged_reviews = [")
    for i, item in enumerate(tagged_reviews):
        if i == 0:
            f.write(str(item))
        else:
            f.write(", "+str(item))
    f.write("]")

