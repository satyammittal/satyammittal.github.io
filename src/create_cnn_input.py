from reviews import *
import re
import io
tagged_reviews = tagged_reviews[:100000]


f1 = io.open("rt-polarity.pos",'w', encoding='utf8')
f2 = io.open("rt-polarity.neg",'w', encoding='utf8')
for label, review in tagged_reviews:
	review = re.sub(r'\n', ' ', review)
	if label==0:
		f2.write(review+"\n")
	else :
		f1.write(review+"\n")
																																																																																																																																																																																																																																																																																																																																																																																																																																																																						