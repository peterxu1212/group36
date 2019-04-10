import json
import numpy as np
import re
from collections import Counter

with open("data.json") as fp:
    data = json.load(fp)

#把review和rating放到兩個list裏
reviews = []
ratings = []

for data_point in data:
    review = data_point["review"]
    review = review.lower()
    #remove punctuation
    review = re.sub(r'[^\w\s]', ' ', review)
    reviews.append(review)
    ratings.append(data_point["rating"])

print ('Number of reviews :', len(reviews))
print(reviews[15])

all_words = ' '.join(reviews)
# create a list of words
word_list = all_words.split()
# Count all the words using Counter Method
count_words = Counter(word_list)
len_words = len(word_list)
sorted_words = count_words.most_common(len_words)
# Code words into numbers
word_to_num = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
reviews_num = []
for review in reviews:
    num = [word_to_num[w] for w in review.split()]
    reviews_num.append(num)
print (reviews_num[0:3])
