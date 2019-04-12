import json
import numpy as np
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.exceptions import FitFailedWarning
import warnings
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
import csv
import time
import logging
import logging.config

with open("../input/data.json") as fp:
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

X_train, X_test, Y_train, Y_test = train_test_split(reviews, ratings, train_size = 0.2)
#X_train_val, X_test, Y_train_val, Y_test = train_test_split(reviews, ratings, test_size = 0.1)
#X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size = 0.2)

pclf = Pipeline([
    ('vect', CountVectorizer(min_df=2, max_df=1.0, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))),
    ('tfidf', TfidfTransformer(use_idf=False, smooth_idf=False, sublinear_tf=False)),
    ('norm', Normalizer()),
    ('clf', LinearSVC(C=1.0, random_state=10, tol=1e-05, max_iter=5000)),
])
"""
i_cross_validation_fold_count = 4

cv_results = cross_validate(pclf, X_train, Y_train, scoring='f1_weighted', cv=i_cross_validation_fold_count, n_jobs=2, return_train_score=False, verbose=10)

tmp_scores = cv_results['test_score']
print("tmp_scores = ", type(tmp_scores), tmp_scores, tmp_scores.shape)

mean_f1_from_cv = np.mean(tmp_scores)
print("\n\n mean_f1_from_cv = ", mean_f1_from_cv)
"""
print("start fitting")
pclf.fit(X_train, Y_train)
print("start predict")
Y_test_pred = pclf.predict(X_test)
print(metrics.classification_report(Y_test, Y_test_pred, digits=5))
