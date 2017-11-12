import json
funny = 0
for line in open('../dataset/yelp_academic_dataset_review.json', 'r'):
	review = json.loads(line)
	if review['votes']['funny'] > 2:
	    funny += 1

print funny

