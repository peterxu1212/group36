# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:00:41 2019

@author: PeterXu
"""

from sklearn import metrics

import csv

def retrieve_from_csv(in_str_fn):
    
    out_list = []
    
    csvfile = open(in_str_fn, 'rt')
            
    i_index = 0
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
        #print(row, "\n")
        
        if i_index >= 1:
            out_list.append(int(row[1]))
        
        
        i_index += 1
            
    return out_list





Y_pp_test = retrieve_from_csv("./pp_test.csv")


print("\n\nlen of Y_pp_test:", len(Y_pp_test))

"""
str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549707302_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\n\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test))





str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549788423_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\n\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test))
"""





str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549841275_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\n\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test))





# 

print("\n\n\n\n with english stop words in counter vector function \n")
str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549965174_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






# 

print("\n\n\n\n without stop words in  counter vector function, and and clean up \n")
str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549965614_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words in  counter vector function, and with max_df=0.85, and and clean up \n")
str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549966252_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))







print("\n\n\n\n without stop words in  counter vector function, and with max_df=0.75, and and clean up \n")
str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549970491_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))








print("\n\n\n\n without stop words in  counter vector function, using lemmetized words, and clean up \n")
str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549973382_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words in  counter vector function, using lemmetized words, stemming words and clean up \n")
str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549984590_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))







print("\n\n\n\n without stop words in  counter vector function, using stemming words and clean up \n")
str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549986753_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))











print("\n\n\n\n with revised stop words in  pre-process, using lemmetized words and clean up \n")
str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549988733_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))









print("\n\n\n\n without stop words, using lemmetized words and clean up \n")
str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549991277_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))







print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=0.85, \n")



str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549994671_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))



print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=0.75, \n")
str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549992774_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))







print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=0.65, \n")
str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549993307_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=0.55, \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549993783_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=0.45, \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549994264_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=0.50, \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549995511_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=0.50, \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1549995511_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))







print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=0.55, min_df=0.0002 \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550108202_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=0.8, min_df=0.0002 \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550109014_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=1.0, min_df=0.0002 \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550109574_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=1.5, min_df=0.0002 \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550110661_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=0.9, min_df=0.0002 \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550111120_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))













print("\n\n\n\n without stop words, using lemmetized words and clean up, with max_df=1.0, min_df=0.0002  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550130162_submission_cleanup_w_lemmatize.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words, using lemmetized words and clean up, with with max_df=1.0, min_df=0.0002  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550130587_submission_cleanup_wo_punctuation_w_lemmatize.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))












print("\n===========================================================================================\n")




"""

print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm  100 fit \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_svm_1550058552_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))








print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 20000 fit  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550068072_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))

"""



print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=1.0  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550068626_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))



print("\n\n\n\n without stop words, using cleanup_wo_sw_negate_w_lemmatize_w_swf, with svm linear_svc 25000 fit, max_df=1.0, separate the punctuation with words  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550342402_submission_cleanup_wo_sw_negate_w_lemmatize_w_swf_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))



print("\n\n\n\n without stop words, using cleanup_w_lemmatize, with svm linear_svc 25000 fit, max_df=1.0, separate the punctuation with words  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550358905_submission_cleanup_w_lemmatize_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n without stop words, CV, using cleanup_w_lemmatize, with svm linear_svc 25000 fit, max_df=1.0, separate the punctuation with words  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550363128_submission_cleanup_w_lemmatize_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))








print("\n--------------------------------------------------------------------------------------------------------------\n")

print("\n--------------------------------------------------------------------------------------------------------------\n")

print("\n--------------------------------------------------------------------------------------------------------------\n")


print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=0.55  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550069415_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=0.75  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550069765_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=1.0 tol e-5  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550070627_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words, using snowball stemming and clean up, with svm linear_svc 25000 fit, max_df=1.0 tol e-5  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550074032_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=1.0 min_df=0.0002 tol e-5  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550111655_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=1.0,  tol e-5 Model selection, max_iter=5000, C=100.0 ? \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550118999_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))








print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=1.0,  tol e-5, max_iter=5000, C=10.0 \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550120150_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=1.0, min_df=1,  tol e-5, max_iter=5000, C=1.0 \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550120670_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=1.0, min_df=2, tol e-5, max_iter=1000, C=1.0 \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550121204_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))



print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=1.0, min_df=2, tol e-5, max_iter=5000, C=1.0 \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550121701_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=1.0, min_df=2, tol e-5, max_iter=5000, C=10.0 \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550122165_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=1.0, min_df=2, tol e-5, max_iter=5000, C=100.0 \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550122931_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=1.0, min_df=2, tol e-5, max_iter=5000, C=0.1 \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550123737_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words, using lemmetized words and clean up, with svm linear_svc 25000 fit, max_df=1.0, min_df=2, tol e-5, max_iter=5000, C=0.1, ngram_range=(1, 3) \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550124330_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n\n\n\n without stop words, using cleanup_wo_punctuation_w_lemmatize, with svm linear_svc 25000 fit, max_df=1.0, min_df=2, tol e-5, max_iter=5000, C=1.0, ngram_range=(1, 4) \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550132100_submission_cleanup_wo_punctuation_w_lemmatize.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words, using cleanup_w_lemmatize, with svm linear_svc 25000 fit, max_df=1.0, min_df=2, tol e-5, max_iter=5000, C=1.0, ngram_range=(1, 4) \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550133305_submission_cleanup_w_lemmatize.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words, using cleanup, with svm linear_svc 25000 fit, max_df=1.0, min_df=2, tol e-5, max_iter=5000, C=1.0, ngram_range=(1, 4) \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550133985_submission_cleanup.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))



print("\n\n\n\n without stop words, using none, with svm linear_svc 25000 fit, max_df=1.0, min_df=2, tol e-5, max_iter=5000, C=1.0, ngram_range=(1, 4) \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550134589_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n\n\n\n without stop words, using w_lemmatize, with svm linear_svc 25000 fit, max_df=1.0, min_df=2, tol e-5, max_iter=5000, C=1.0, ngram_range=(1, 4) \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550187675_submission_w_lemmatize.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))


print("\n========================================================================================================================================\n")


print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")




print("\n\n\n\n without stop words, using cleanup_negate_w_swf, with with max_df=1.0, min_df=0.0002  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550207321_submission_cleanup_negate_w_swf.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words, using cleanup_negate_w_swf combined, with with max_df=1.0, min_df=0.0002  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550209665_submission_cleanup_negate_w_swf.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n\n\n\n without stop words, using cleanup_negate_w_swf combined, with with max_df=1.0, min_df=0.0002, new alg  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550218566_submission_cleanup_negate_w_swf_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words, using cleanup_wo_sw_negate_w_lemmatize_w_swf combined, with with max_df=1.0, min_df=0.0002, new alg  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550284368_submission_cleanup_wo_sw_negate_w_lemmatize_w_swf_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))







print("\n\n\n\n without stop words, using cleanup_wo_sw_negate_w_lemmatize_w_swf combined, with with max_df=1.0, min_df=0.0002, separate the punctuation with words  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550337094_submission_cleanup_wo_sw_negate_w_lemmatize_w_swf_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n without stop words, using cleanup_w_lemmatize combined, with with max_df=1.0, min_df=0.0002, separate the punctuation with words  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550358424_submission_cleanup_w_lemmatize_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))



print("\n\n\n\n without stop words, using cleanup_negate_w_lemmatize combined, with with max_df=1.0, min_df=0.0002, separate the punctuation with words  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550361544_submission_cleanup_negate_w_lemmatize_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))








print("\n\n\n\n without stop words, using cleanup_w_lemmatize_w_separate_wp combined, with with max_df=1.0, min_df=0.0002  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550418924_submission_cleanup_w_lemmatize_w_separate_wp_stat_try.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))



print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")



print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")




print("\n\n\n\n without stop words, using cleanup_negate_w_lemmatize_w_swf_w_separate_wp combined, with with max_df=1.0, min_df=0.0002  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_bernoulli_nb_1550600592_submission_cleanup_negate_w_lemmatize_w_swf_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n without stop words, using cleanup_negate_w_lemmatize_w_swf_w_separate_wp combined, with with max_df=1.0, min_df=0.0002, max_features=100000  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550617035_submission_cleanup_w_lemmatize_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n\n\n\n without stop words, using cleanup_negate_w_lemmatize_w_swf_w_separate_wp combined, with with max_df=1.0, min_df=0.0002, max_features=50000  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550617538_submission_cleanup_w_lemmatize_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))









print("\n\n\n\n without stop words, using cleanup_negate_w_lemmatize_w_swf_w_separate_wp combined, with with max_df=1.0, min_df=0.0002, max_features=200000  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550618455_submission_cleanup_w_lemmatize_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n----------------------------------------------ensemble section------------------------------------------------------------------------------------------\n")



print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")


str_tt = "1550751090"


print("\n\n\n\n ensemble LR  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_LR_" + str_tt + "_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n ensemble SGD  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_SGD_" + str_tt + "_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n\n\n\n ensemble SVM  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_SVM_" + str_tt + "_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n ensemble LR & SGD  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_LR_SGD_" + str_tt + "_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n ensemble LR & SGD & SE  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_LR_SGD_SE_" + str_tt + "_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n ensemble LR & SGD & SVM  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_LR_SGD_SVM_" + str_tt + "_submission.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))






print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")

print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")








print("\n\n\n\n ensemble RF max feature = 6000  Bag of Words  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_rf_1550737410_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n\n\n\n ensemble RF max feature = none  Bag of Words   \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_rf_1550738187_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n\n\n\n ensemble RF max feature = none  tfidf   \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_rf_1550738881_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")

print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")

print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")




print("\n\n\n\n cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550718696_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))









print("\n\n\n\n cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp  with NWN  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550719063_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))










print("\n\n\n\n cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp  \n")



str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550748947_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat.csv"

#str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550719191_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))









print("\n\n\n\n cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp  with NWN  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550719561_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))


























print("\n\n\n\n cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550727527_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))









print("\n\n\n\n cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp  with NWN  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550727952_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))










print("\n\n\n\n cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550728378_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))









print("\n\n\n\n cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp  with NWN  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550727960_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n")

print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")

print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")



print("\n\n\n\n <R cleanup_negate_w_nwn_w_swf_w_separate_wp_stat_simplified  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550808105_submission_cleanup_negate_w_nwn_w_swf_w_separate_wp_stat_simplified.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))




print("\n\n\n\n LR cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat_simplified  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_logistic_regression_1550809496_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat_simplified.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))





print("\n\n\n\n SVM cleanup_negate_w_nwn_w_swf_w_separate_wp_stat_simplified  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550808528_submission_cleanup_negate_w_nwn_w_swf_w_separate_wp_stat_simplified.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))







print("\n\n\n\n SVM cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat_simplified  \n")

str_csv_file_name = "../group12/Deliverables/csv/group12_linear_svc_1550809190_submission_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat_simplified.csv"

Y_pred_test = retrieve_from_csv(str_csv_file_name)

print("\n\nlen of Y_pred_test :", len(Y_pred_test), " for ", str_csv_file_name)

print(metrics.classification_report(Y_pp_test, Y_pred_test, digits=5))










