#!/usr/bin/env python3



import os

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer


from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_predict

import numpy as np

from sklearn.exceptions import FitFailedWarning
import warnings

from sklearn.pipeline import Pipeline


from sklearn import metrics

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import f1_score

from sklearn import linear_model

import csv



import time

import logging
import logging.config

import json

#from os import listdir
#from os.path import isfile, join

#from pathlib import Path

#import re

#from bs4 import BeautifulSoup


def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]
        
def fn_progress(cls_name, stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = "%20s classifier : \t" % cls_name
    #s += "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
    s += "%(n_train)6d train docs " % stats
    #s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % test_stats
    s += "accuracy: %(accuracy).6f " % stats
    s += "in %.2fs (%5d docs/s)" % (duration, stats['n_train'] / duration)
    return s



#print(os.listdir("../input/yelp-imdb-multi-class/datasets/datasets/"))


b_do_kaggle = False
#b_do_kaggle = True






i_cross_validation_fold_count = 5

#b_partial = True
b_partial = False



b_cleanup = True


b_wo_sw = False

b_w_lemmatize = True

b_w_stemming = False


b_w_nwn = True

b_wo_punctuation = False



b_sentiment_words_filter = True


b_negate = b_sentiment_words_filter


#b_negate = True

b_separate_wp = True



i_partial_count = 5000


b_do_model_selection = False
#b_do_model_selection = True

b_do_cross_validation = False
#b_do_cross_validation = True


#b_use_new_data_set = False
b_use_new_data_set = True

#b_use_new_data_IMDB_or_YELP = True
b_use_new_data_IMDB_or_YELP = False


s_log_config_fn = "logging.conf"


if b_do_kaggle:
    s_log_config_fn = "../input/config/" + s_log_config_fn

logging.config.fileConfig(s_log_config_fn)





# create logger
logger = logging.getLogger('Project2Group12')



logger.info("\n\n\n\n\n\n\n\n\n\nprogram begins. ")


str_fn_postfix = ""


#b_Tfidf_or_BOW = False
b_Tfidf_or_BOW = True

if b_cleanup:
    str_fn_postfix += "_cleanup"


if b_wo_sw:
    str_fn_postfix += "_wo_sw"

if b_negate:
    str_fn_postfix += "_negate"

if b_wo_punctuation:
    str_fn_postfix += "_wo_punctuation"

if b_w_lemmatize:
    str_fn_postfix += "_w_lemmatize"
    

if b_w_stemming:
    str_fn_postfix += "_w_stemming"


if b_w_nwn:
    str_fn_postfix += "_w_nwn"
    
    
if b_sentiment_words_filter:
    str_fn_postfix += "_w_swf"

if b_separate_wp:
    str_fn_postfix += "_w_separate_wp"



if True:
    str_fn_postfix += "_stat"



str_fn_postfix += "_simplified"


b_use_original_text = True


#str_fn_postfix = "_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat_simplified"




str_json_fn_training = "../" + "training" + "_set" + str_fn_postfix + ".json"


str_input_folder = "../../../"

if b_do_kaggle:
    str_input_folder = "../input/yelp-imdb-multi-class/datasets/"

if b_use_new_data_set:
    
    if b_use_new_data_IMDB_or_YELP:
        str_json_fn_training = str_input_folder + "datasets/JMARS_10_label_imdb_dataset/data/data.json"
    else:   
        str_json_fn_training = str_input_folder + "datasets/Zhang_5_label_yelp_dataset/data/data.json"
    


str_json_fn_testing = "../" + "testing" + "_set" + str_fn_postfix + ".json"

if b_use_new_data_set:
    
    if b_use_new_data_IMDB_or_YELP:
        str_json_fn_testing = str_input_folder + "datasets/JMARS_10_label_imdb_dataset/data/data.json"
    else:   
        str_json_fn_testing = str_input_folder + "datasets/Zhang_5_label_yelp_dataset/data/data.json"


logger.info("str_json_fn_training = %s \n", str_json_fn_training)

logger.info("str_json_fn_testing = %s \n", str_json_fn_testing)

logger.info("begin pre-process for training datas... ")

X_list_raw = []
Y_list_raw = []

all_files_list_raw = []



jsonFile = open(str_json_fn_training, "r") # Open the JSON file for reading
json_data = json.load(jsonFile) # Read the JSON into the buffer
jsonFile.close() # Close the JSON file


real_training_data_set = []

if not b_partial:
    
    if b_use_new_data_set:
        real_training_data_set = [item for item in json_data]
    else:    
        real_training_data_set = [item for item in json_data if item['y_val'] > -2]
else:
    if b_use_new_data_set:
        real_training_data_set = [item for item in json_data]
        real_training_data_set = [item for item in real_training_data_set[:i_partial_count]]
    else:
        real_training_data_set = sorted(json_data, key=lambda x: (x['id']), reverse=False)    
        real_training_data_set = [item for item in real_training_data_set[:i_partial_count] if item['y_val'] > -2]
      
    
#real_training_data_set_sorted = sorted(real_training_data_set, key=lambda x: (x['id']), reverse=False)
real_training_data_set_sorted = real_training_data_set

logger.info("len of real_training_data_set_sorted = %d ", len(real_training_data_set_sorted))

tmp_list_X_raw = []
tmp_list_Y_raw = []
#tmp_files_list_raw = []
#tmp_pure_files_list_raw = []


for data_point in real_training_data_set_sorted:
	

    i_se = 0
    #i_se = data_point['i_sentiment_estimate']
    
    str_st = ""
    
    if b_use_original_text:
        #str_st = data_point['text']
        #str_st = data_point['text_simple_cleanup']
        
        #str_st = data_point['raw_text']
        #str_st = data_point['simplified_cleanup']        
        
        
        #str_st = data_point['str_nwn']
        #str_st = data_point['text_letter_and_num']
        
        #str_st = data_point['text_wo_html_tags']
        
        if b_use_new_data_set:
            str_st = data_point['review']
        else:
            str_st = data_point['simplified_cleanup']
        
        
    else:
        str_st = data_point['senti_text']
    
        if i_se >= 15:
            str_st += " . imdbsuperpositive "
        elif i_se < 15 and i_se >= 5:
            str_st += " . imdbstrongpositive "
        elif i_se < 5 and i_se > -5:
            pass
        elif i_se <= -5 and i_se > -15:
            str_st += " . imdbstrongnegitive "
        elif i_se <= -15:
            str_st += " . imdbsupernegitive "
        else:
            pass
    
    tmp_list_X_raw.append(str_st)
    if b_use_new_data_set:
        tmp_list_Y_raw.append(data_point['rating'])
        
    else:
        tmp_list_Y_raw.append(data_point['y_val'])
	
    #tmp_files_list_raw.append(data_point['full_file_name'])	
    #tmp_pure_files_list_raw.append(data_point['id'])

X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

#all_files_list_raw.extend(tmp_files_list_raw)


logger.info("end pre-process for training datas... ")





logger.info("begin pre-process for testing datas... ")


X_list_real_test_raw = []
Y_list_real_test_raw = []

#all_files_list_real_test_raw = []
#all_pure_files_list_real_test_raw = []



jsonFile = open(str_json_fn_testing, "r") # Open the JSON file for reading
json_data = json.load(jsonFile) # Read the JSON into the buffer
jsonFile.close() # Close the JSON file


real_testing_data_set = []

"""
if not b_partial:
    real_testing_data_set = [item for item in json_data if item['y_val'] > -2]
else:
    real_testing_data_set = [item for item in json_data[:i_partial_count] if item['y_val'] > -2]


real_testing_data_set_sorted = sorted(real_testing_data_set, key=lambda x: (x['id']), reverse=False)
"""



if not b_partial:
    
    if b_use_new_data_set:
        real_testing_data_set = [item for item in json_data]
    else:    
        real_testing_data_set = [item for item in json_data if item['y_val'] > -2]
else:
    if b_use_new_data_set:
        real_testing_data_set = [item for item in json_data]
        real_testing_data_set = [item for item in real_testing_data_set[:i_partial_count]]
    else:
        real_testing_data_set = sorted(json_data, key=lambda x: (x['id']), reverse=False)    
        real_testing_data_set = [item for item in real_testing_data_set[:i_partial_count] if item['y_val'] > -2]



real_testing_data_set_sorted = real_testing_data_set



logger.info("len of real_testing_data_set_sorted = %d ", len(real_testing_data_set_sorted))

tmp_list_X_raw = []
tmp_list_Y_raw = []
tmp_files_list_raw = []
tmp_pure_files_list_raw = []


for data_point in real_testing_data_set_sorted:
	
    #tmp_list_X_raw.append(data_point['senti_text'])
    
    i_se = 0
    #i_se = data_point['i_sentiment_estimate']
    
    str_st = ""   
    if b_use_original_text:
        #str_st = data_point['text']
        #str_st = data_point['text_simple_cleanup']
        
        
        #str_st = data_point['str_nwn']
        
        
        #str_st = data_point['raw_text']
        #str_st = data_point['text_letter_and_num']
        #str_st = data_point['simplified_cleanup']
        #str_st = data_point['text_wo_html_tags']
        
        if b_use_new_data_set:
            str_st = data_point['review']
        else:
            str_st = data_point['simplified_cleanup']
        
    
    else:
        str_st = data_point['senti_text']
    
        if i_se >= 15:
            str_st += " . imdbsuperpositive "
        elif i_se < 15 and i_se >= 5:
            str_st += " . imdbstrongpositive "
        elif i_se < 5 and i_se > -5:
            pass
        elif i_se <= -5 and i_se > -15:
            str_st += " . imdbstrongnegitive "
        elif i_se <= -15:
            str_st += " . imdbsupernegitive "
        else:
            pass

    tmp_list_X_raw.append(str_st)

    #tmp_list_Y_raw.append(data_point['y_val'])
	
    if b_use_new_data_set:
        tmp_list_Y_raw.append(data_point['rating'])
        
    else:
        tmp_list_Y_raw.append(data_point['y_val'])
	

    #tmp_files_list_raw.append(data_point['full_file_name'])	
    #tmp_pure_files_list_raw.append(data_point['id'])

X_list_real_test_raw.extend(tmp_list_X_raw)
Y_list_real_test_raw.extend(tmp_list_Y_raw)

#all_files_list_real_test_raw.extend(tmp_files_list_raw)
#all_pure_files_list_real_test_raw.extend(tmp_pure_files_list_raw)


logger.info("end pre-process for testing datas... ")






# seems that the all data (both training and testing) are combined together and then being splited, 
# according to train_size and test_size parameters of train_test_split
#X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, train_size=0.6, test_size=0.4)


logger.info("start train_test_split. ")


i_random_state=20

#X_train, X_test, Y_train, Y_test, all_files_train, all_files_test = train_test_split(X_list_raw, Y_list_raw, all_files_list_raw, train_size=0.8, test_size=0.2, random_state=i_random_state)
X_train, X_test, Y_train, Y_test = train_test_split(X_list_raw, Y_list_raw, train_size=0.8, test_size=0.2, random_state=i_random_state)


logger.info("end train_test_split. ")

print("len of X_train, X_test, Y_train, Y_test \n", len(X_train), len(X_test), len(Y_train), len(Y_test))

i_random_index = 15

print("\n\n i_random_index = ", i_random_index)

print("\n\n\n\n item of X_train: ==============================================================\n")
print(X_train[i_random_index])

print("\n\n\n\n item of Y_train: ==============================================================\n")
print(Y_train[i_random_index])

#print("\n\n\n\n item of all_files_train: ==============================================================\n")
#print(all_files_train[i_random_index])




i_random_index = 18

print("\n\n i_random_index = ", i_random_index)

print("\n\n\n\n item of X_test: ==============================================================\n")
print(X_test[i_random_index])

print("\n\n\n\n item of Y_test: ==============================================================\n")
print(Y_test[i_random_index])

#print("\n\n\n\n item of all_files_test: ==============================================================\n")
#print(all_files_test[i_random_index])





i_random_index = 20

print("\n\n i_random_index = ", i_random_index)

print("\n\n\n\n item of X_list_real_test_raw: ==============================================================\n")
print(X_list_real_test_raw[i_random_index])

print("\n\n\n\n item of Y_list_real_test_raw: ==============================================================\n")
print(Y_list_real_test_raw[i_random_index])

#print("\n\n\n\n item of all_files_list_real_test_raw: ==============================================================\n")
#print(all_files_list_real_test_raw[i_random_index])





X_train_whole = []

X_train_whole.extend(X_train)
X_train_whole.extend(X_test)

Y_train_whole = []

Y_train_whole.extend(Y_train)
Y_train_whole.extend(Y_test)


#all_files_whole_train = []

#all_files_whole_train.extend(all_files_train) 
#all_files_whole_train.extend(all_files_test)






print("len of X_train_whole and Y_train_whole", len(X_train_whole), len(Y_train_whole))


print("len of X_train and Y_train", len(X_train), len(Y_train))



logger.info("start CountVectorizer. ")

vect = CountVectorizer(min_df=2, max_df=1.0, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))

count_vect = vect.fit(X_train)

X_train_counts = count_vect.transform(X_train)
X_test_counts = count_vect.transform(X_test)

logger.info("end CountVectorizer. ")




logger.info("start TfidfTransformer. ")

tfidf = TfidfTransformer(use_idf=b_Tfidf_or_BOW, smooth_idf=b_Tfidf_or_BOW, sublinear_tf=b_Tfidf_or_BOW)

tfidf_transformer = tfidf.fit(X_train_counts)

X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)


logger.info("end TfidfTransformer. ")



logger.info("start Normalizer. ")

norm = Normalizer()

normalizer_tranformer = norm.fit(X_train_tfidf)
X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
X_test_normalized = normalizer_tranformer.transform(X_test_tfidf)


logger.info("end Normalizer. ")



#clf = LinearSVC(C=1.0, random_state=10, tol=1e-05, max_iter=1000)

#.fit(X_train_normalized, y_train)

clf = linear_model.SGDClassifier(shuffle=True, loss='hinge', penalty = 'l2', random_state=0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=10, max_iter=1000, tol=1e-05)





"""

pclf = Pipeline([
    #('vect', CountVectorizer(min_df=2, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4), stop_words = 'english')),
    #('vect', CountVectorizer(min_df=2, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))),
    #('vect', CountVectorizer(min_df=0.0002, max_df=1.0, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))),
    #('vect', CountVectorizer(min_df=1, max_df=1.0, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))),
    #('vect', CountVectorizer(min_df=2, max_df=1.0, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))),
    ('vect', CountVectorizer(min_df=2, max_df=1.0, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))),
    ('tfidf', TfidfTransformer(use_idf=b_Tfidf_or_BOW, smooth_idf=b_Tfidf_or_BOW, sublinear_tf=b_Tfidf_or_BOW)),
    ('norm', Normalizer()),
    #('clf', LinearSVC(C=1.0, random_state=10, tol=1e-05, max_iter=5000)),
    ('clf', linear_model.SGDClassifier(shuffle=True, loss='hinge', penalty = 'l2', random_state=0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, max_iter=1000, tol=1e-05)),
])
    
"""

    
    
#pclf.fit(X_train_whole, Y_train_whole)

#pclf.fit(X_train, Y_train)
    
    
"""
params = {"clf__C":[0.1, 1.0, 10.0, 100.0]

}
"""


params = {"clf__C":[1.0]
}

    
""" 
params = {"vect__ngram_range": [(1,4)],
             "vect__min_df" : [2], 
             "vect_max_df" : [0.5, 0.6, 0.7, 0.8],
            "clf__C": [20]
}
"""

  
#params = { 
#          "vect_max_df" : [0.5, 0.6, 0.7, 0.8]

#}


""" 
params = {"vect__ngram_range": [(1,2),(1,3),(1,4)],
            "clf__C": [1, 30]
}
"""
    
"""
params = {"vect__ngram_range": [(1,1),(1,2),(1,3),(1,4)],
            "clf__C": [1, 15, 30]
}
"""
    
"""
params = {"clf__C": [1, 4, 16, 32]
}
"""

#model_search_SVM = GridSearchCV(pclf, param_grid = params, scoring = 'roc_auc', cv = 5, verbose = 10, error_score='raise', iid=True, n_jobs=2)
model_search_SVM = GridSearchCV(clf, param_grid = params, scoring = 'roc_auc', cv = 5, verbose = 10, error_score='raise', iid=True, n_jobs=2)


partial_fit_classifiers = {
    'SGD_SVM': 0,
    #'Perceptron': Perceptron(tol=1e-3),
    #'NB Multinomial': MultinomialNB(alpha=0.01),
    #'Passive-Aggressive': PassiveAggressiveClassifier(tol=1e-3),
}


cls_stats = {}

for cls_name in partial_fit_classifiers:
    stats = {'n_train': 0, 'n_train_pos': 0,
             'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
             'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
    
    cls_stats[cls_name] = stats


best_clf = None

Y_test_pred_best = []


st = int(time.time())

logger.info("start fit for pclf")

if b_do_model_selection:
   
    with warnings.catch_warnings(record=True) as w:
        try:
            #gs.fit(X, y)   # This will raise a ValueError since C is < 0        
            model_search_SVM.fit(X_train_whole, Y_train_whole) # Fit the model.
            #model_search_LR.fit(X_train, Y_train) # Fit the model.
        except Exception as e:
            #except ValueError:
            print("\n\n Exception e = " + str(e))
            #pass
        print("\n\n w = \n", w)

    # Try to set the scoring on what the contest is asking for. 
    # The contest says scoring is for area under the ROC curve, so use this.
    
    #error_score=np.nan
    
    
    print("\n\n model_search_SVM.scorer_ = \n", model_search_SVM.scorer_) 
    
    print("\n\n model_search_SVM.best_index_ = \n", model_search_SVM.best_index_) 
    
    print("\n\n model_search_SVM.best_score_ = \n", model_search_SVM.best_score_)                        
    
    print("\n\n model_search_SVM.best_params_ = \n", model_search_SVM.best_params_)
    
    print("\n\n model_search_SVM.best_estimator_ = \n", model_search_SVM.best_estimator_)
    
    
    print("\n\n model_search_SVM.cv_results_ = \n", model_search_SVM.cv_results_)
   

else:
    
    if b_do_cross_validation:
        
        cv_results = cross_validate(clf, X_train_whole, Y_train_whole, scoring='f1_weighted', cv=i_cross_validation_fold_count, n_jobs=1, return_train_score=False, verbose=10)
    
        logger.info(sorted(cv_results.keys()))
    
        logger.info(cv_results)
        
        
        tmp_scores = cv_results['test_score']
        print("tmp_scores = ", type(tmp_scores), tmp_scores, tmp_scores.shape)
        
        mean_f1_from_cv = np.mean(tmp_scores)
        
        
        print("\n\n mean_f1_from_cv = ", mean_f1_from_cv)
        #logger.info("", mean_f1_from_cv)
        
        
    else:
        
        print("X_train_normalized:", X_train_normalized.shape, "\n")
        
        
        cls_classes = np.unique(Y_train)
        
        print("cls_classes = ", cls_classes , "\n")
        
        i_total_train_len = X_train_normalized.shape[0]
        
        i_mini_batch_size = 20000
        i_n_iter_no_change_before_early_stopping = 10
        
        #pclf.fit(X_train_whole, Y_train_whole)
        #pclf.fit(X_train, Y_train)
        #clf.fit(X_train, Y_train)
        
        #clf.fit(X_train_normalized, Y_train)
        #clf2 = SGDClassifier(shuffle=True, loss='log')
        
        
        
        cls_name = "SGD_SVM"
        
        total_vect_time = 0.0
        
        overall_best_score = 0.0
        
        i_n_iter_no_improve = 0
        
        i_current_iter = 0
        
        n_iter = 100
        for n in range(n_iter):
            
            cur_iter_best_score = 0.0
            
            best_iter_clf = None
            
            Y_test_pred_best_cur_iter = []
            
            for mini_batch in batches(range(i_total_train_len), i_mini_batch_size):
                                
                #logger.info(" start partial_fit for clf")
                
                tick = time.time()
                
                x_train_batch = X_train_normalized[mini_batch[0]:mini_batch[-1]+1]
                y_train_batch = Y_train[mini_batch[0]:mini_batch[-1]+1]
                
                total_vect_time += time.time() - tick
                
                
                #print("mini_batch:",  mini_batch[0], " to ", mini_batch[-1], "\n")
                
                tick = time.time()               
                
                
                clf.partial_fit(x_train_batch, y_train_batch, classes=cls_classes)

                #logger.info(" end partial_fit for clf")
                cls_stats[cls_name]['total_fit_time'] += time.time() - tick
                cls_stats[cls_name]['n_train'] += x_train_batch.shape[0]
                cls_stats[cls_name]['n_train_pos'] += sum(y_train_batch)
                
                tick = time.time()
                
                current_score = clf.score(X_test_normalized, Y_test)
                
                if cur_iter_best_score < current_score:
                    cur_iter_best_score = current_score
                    
                    #best_iter_clf = clf
                    Y_test_pred_best_cur_iter = []
                    Y_test_pred_best_cur_iter = clf.predict(X_test_normalized)
                    
                    
                    
                    #print("updated cur_iter_best_score = ", cur_iter_best_score, "\n")
                    
                    
                cls_stats[cls_name]['accuracy'] = current_score
                
                
                
                cls_stats[cls_name]['prediction_time'] = time.time() - tick
                
                acc_history = (cls_stats[cls_name]['accuracy'],
                               cls_stats[cls_name]['n_train'])
                
                cls_stats[cls_name]['accuracy_history'].append(acc_history)
                
                run_history = (cls_stats[cls_name]['accuracy'], total_vect_time + cls_stats[cls_name]['total_fit_time'])
                
                cls_stats[cls_name]['runtime_history'].append(run_history)
                
            
            i_current_iter += 1
            
                     
            if i_current_iter % 1 == 0:
                logger.info("i_current_iter = %d : %s ", i_current_iter, fn_progress(cls_name, cls_stats[cls_name]))
                #print(, "\n")


            if overall_best_score < cur_iter_best_score:
                overall_best_score = cur_iter_best_score
                
                #best_clf = best_iter_clf
                
                Y_test_pred_best = []
                Y_test_pred_best = Y_test_pred_best_cur_iter.copy()
                
                print("updated overall_best_score = ", overall_best_score, "\n")
                
                i_n_iter_no_improve = 0
                
            else:
                i_n_iter_no_improve += 1
                
                if i_n_iter_no_improve >= i_n_iter_no_change_before_early_stopping:
                    logger.info(" early stop due to no improvement within recent %d iters \n", i_n_iter_no_change_before_early_stopping)
                    break
             



logger.info(" end fit for pclf")



et = int(time.time())


spend_time = et - st


logger.info("\n spend_time for the fit = %d", spend_time)


#tmp_vect_X = pclf.named_steps['vect'].X


#print("\n\n\n\n tfidf idf_", tmp_vect_X, len(tmp_vect_X), type(tmp_vect_X), tmp_vect_X.shape)

tmp_fns = []



if b_do_model_selection:
    tmp_fns = model_search_SVM.best_estimator_.named_steps['vect'].get_feature_names()
else:
    
    if b_do_cross_validation:
        pass
    else:
        #tmp_fns = clf.named_steps['vect'].get_feature_names()
        tmp_fns = vect.get_feature_names()

i_len_fns = len(tmp_fns)

print("\n\n\n\n i_len_fns = ", i_len_fns, type(tmp_fns))




#list_Y_test_pred = Y_test_pred.tolist()


#print("\n\n\n\n vect get_feature_names", pclf.named_steps['vect'].get_feature_names())
"""

tmp_idfs = []

if b_do_model_selection:
    
    tmp_idfs = model_search_SVM.best_estimator_.named_steps['tfidf'].idf_
else:
    
    tmp_idfs = pclf.named_steps['tfidf'].idf_


print("\n\n\n\n tfidf idf_", tmp_idfs, len(tmp_idfs), type(tmp_idfs), tmp_idfs.shape)

list_tmp_idfs = tmp_idfs.tolist()


print("\n\n\n\n clf list_tmp_idfs", len(list_tmp_idfs))


tmp_coefs = []

if b_do_model_selection:
    
    tmp_coefs = model_search_SVM.best_estimator_.named_steps['clf'].coef_
    
else:

    tmp_coefs = pclf.named_steps['clf'].coef_

print("\n\n\n\n clf coef_", tmp_coefs, len(tmp_coefs), type(tmp_coefs), tmp_coefs.shape)

tmp_coefs_line = tmp_coefs[0]

list_tmp_coefs = tmp_coefs_line.tolist()


print("\n\n\n\n clf list_tmp_coefs", len(list_tmp_coefs))


tmp_intercept = []


if b_do_model_selection:
    
    tmp_intercept = model_search_SVM.best_estimator_.named_steps['clf'].intercept_
    
else:
    tmp_intercept = pclf.named_steps['clf'].intercept_


print("\n\n\n\n clf intercept_", tmp_intercept, len(tmp_intercept), type(tmp_intercept))



pclf_feature_set = []

for x in range(0, i_len_fns, 1):
    
    pclf_feature_item = {}
    
    pclf_feature_item['term'] = tmp_fns[x]

    pclf_feature_item['idf'] = list_tmp_idfs[x]
    
    pclf_feature_item['coef'] = list_tmp_coefs[x]

    pclf_feature_set.append(pclf_feature_item)



pclf_feature_set_sorted = sorted(pclf_feature_set, key=lambda x: (x['idf']), reverse=False)

print("\n\n\nn\ len of pclf_feature_set_sorted by idf", len(pclf_feature_set_sorted))

#print(pclf_feature_set_sorted[0], pclf_feature_set_sorted[i_len_fns - 1])



i_buffer = 1000

print("\n\n\n\n  least: \n", i_buffer)

for x in range(0, 0 + i_buffer, 1):
    print("\n\n", pclf_feature_set_sorted[x])


print("\n\n\n\n  largest: \n", i_buffer)

for x in range(i_len_fns - 1, i_len_fns - 1 - i_buffer, -1):
    print("\n\n", pclf_feature_set_sorted[x])




pclf_feature_set_sorted = sorted(pclf_feature_set, key=lambda x: (abs(x['coef'])), reverse=False)

print("\n\n\nn\ len of pclf_feature_set_sorted by coef", len(pclf_feature_set_sorted))

#print(pclf_feature_set_sorted[0], pclf_feature_set_sorted[i_len_fns - 1])


i_buffer = 1000

print("\n\n\n\n  least: \n", i_buffer)

for x in range(0, 0 + i_buffer, 1):
    print("\n\n", pclf_feature_set_sorted[x])


print("\n\n\n\n  largest: \n", i_buffer)

for x in range(i_len_fns - 1, i_len_fns - 1 - i_buffer, -1):
    print("\n\n", pclf_feature_set_sorted[x])

"""

#print("\n\n", pclf.coef_, "\n\n")
#print("\n\n", pclf.intercept_, "\n\n")

logger.info("start predict for X_test ")

#Y_test_pred = model_search_LR.predict(X_test)

#Y_test_cv_pred = cross_val_predict(pclf, X_test, Y_test, cv=5, n_jobs=2)



st = int(time.time())

Y_test_pred = []



if b_do_model_selection:
    Y_test_pred = model_search_SVM.best_estimator_.predict(X_test)
else:
    if b_do_cross_validation:
        #Y_test_pred = cross_val_predict(pclf, X_train_whole, Y_train_whole, cv=i_cross_validation_fold_count)
        pass
    else:
        Y_test_pred = clf.predict(X_test_normalized)

        #Y_test_pred_best = best_clf.predict(X_test_normalized)


et = int(time.time())


spend_time = et - st

logger.info("\n spend_time for the predict = %d", spend_time)

"""
Y_test_cv_pred = []
Y_test_cv_pred.extend([0] * int(len(Y_test) / 2))
Y_test_cv_pred.extend([1] * int(len(Y_test) / 2))

print("\n\n type and len of Y_test_cv_pred: ", type(Y_test_cv_pred), len(Y_test_cv_pred))
"""

logger.info("end predict for X_test")




print("\n\n\n\n")



logger.info("start metrics.classification_report ")

print("\n\nmetrics.classification_report for Y_test and Y_test_pred \n")


if b_do_cross_validation:
    #print(metrics.classification_report(Y_train_whole, Y_test_pred, digits=5))
    pass
else:
    print(metrics.classification_report(Y_test, Y_test_pred, digits=5))
    
    print(metrics.classification_report(Y_test, Y_test_pred_best, digits=5))
    


#print("\n\nmetrics.classification_report for Y_test and Y_test_cv_pred \n")

#print(metrics.classification_report(Y_test, Y_test_cv_pred, digits=5))

logger.info("end metrics.classification_report ")


print("\n\n\n\n")


logger.info("program ends. ")