import json
import numpy as np
import re
from collections import Counter


#s_full_file_name = "../../../datasets/JMARS_10_label_imdb_dataset/data/data.json"
s_full_file_name = "../../../datasets/Zhang_5_label_yelp_dataset/data/data.json"

with open(s_full_file_name) as fp:
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

#print (count_words[0:3])

len_words = len(word_list)

print("len_words = ", len_words)


len_counted_words = len(count_words)

print("len_counted_words = ", len_counted_words)

i_index = 0


#for x in count_words.elements():
for x in count_words:   
    
    if i_index <= 10:
        print("\n x = ", x)
    else:
        break
        #pass
    
    i_index += 1
    
#print("i_index = ", i_index)

#len_cw_ele = len(count_words.elements())
#print("len_cw_ele = ", len_cw_ele)



i_len_picked_words_len = 5000
#i_len_picked_words_len = len_counted_words

sorted_words = count_words.most_common(i_len_picked_words_len)


len_sorted_words = len(sorted_words)

print("len_sorted_words = ", len_sorted_words)

# Code words into numbers
word_to_num = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
#word_to_num = {w:i for i, (w,c) in enumerate(sorted_words)}

len_word_to_num = len(word_to_num)

print("len_word_to_num = ", len_word_to_num)









"""         
for key in local_wordcount.keys():
    #print(k, v)
    #str_to_write = k + " " + str(v) + "\n"
    #fout_words.write(str_to_write)
    if key in dict_wc:
        wf_X[dict_wc[key], 0] = local_wordcount[key]
        #print(key, wordcount[key], local_wordcount[key])
"""




reviews_num = []
for review in reviews:
    
    """
    word_list = review.split()
    
    local_wordcount = {}
            
    for word in word_list:
        if word not in local_wordcount:
            local_wordcount[word] = 1
        else:
            local_wordcount[word] += 1

    wf_X = np.zeros(len_word_to_num)
    #print(wf_X.shape)
    
    
    for key in local_wordcount.keys():
    #print(k, v)
    #str_to_write = k + " " + str(v) + "\n"
    #fout_words.write(str_to_write)
        if key in word_to_num:
            #wf_X[word_to_num[key], 0] = local_wordcount[key]
            wf_X[word_to_num[key]] = 1
            #print(key, wordcount[key], local_wordcount[key])
    """    
    
    nums = []
    
    for w in review.split():
        if w in word_to_num:
            nums.append(word_to_num[w])
    
    
    #nums = [word_to_num[w] if w in word_to_num for w in review.split()]
    
    reviews_num.append(nums)
    
    #print("wf_X = ", wf_X)
    
    #list_x = wf_X.tolist()
    
    
    #reviews_num.append(list_x)

    #break
    #output_X = np.append(output_X, [X_entry], axis=0)




print(reviews_num[0], len(reviews_num[0]), "\n")
print(reviews_num[1], len(reviews_num[1]), "\n")
print(reviews_num[2], len(reviews_num[2]), "\n")


