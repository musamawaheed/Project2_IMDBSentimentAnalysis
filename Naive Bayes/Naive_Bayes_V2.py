# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:27:18 2019

@author: Usama
"""

import csv
import pandas as pd
import numpy as np
from math import log
import glob
from collections import Counter
import re

from scipy.sparse import issparse
from sklearn.utils.extmath import safe_sparse_dot
from scipy import sparse
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from timeit import default_timer as timer
from tqdm import tqdm


start = timer()

training_data = '../Data/training_data.csv'
test_data = '../Data/test_data.csv'

data_train = pd.read_csv(training_data)
data_test = pd.read_csv(test_data)

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(_])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def preprocess_reviews(reviews):

    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    reviews = [re.sub("\d+", " ", line) for line in reviews]

    return reviews


data_train['Text_Clean'] = preprocess_reviews(data_train['Text'])
data_test['Text_Clean'] = preprocess_reviews(data_test['Text'])


# cv = CountVectorizer(binary = True, ngram_range=(1,2), stop_words="english",
#          analyzer = "word", max_df=1.0, min_df=1, max_features=50000)#, stop_words="english")
cv = CountVectorizer(binary=True, max_features=None)
cv.fit(data_train['Text_Clean'])
X_train_full = cv.transform(data_train['Text_Clean'])
X_test = cv.transform(data_test['Text_Clean'])
tokens = cv.get_feature_names()

Y_train_full = np.asmatrix(data_train['Score'])
#sY = sparse.csr_matrix(Y)
Y_train_full = Y_train_full.T


X_train, X_val, y_train, y_val = train_test_split(X_train_full, Y_train_full, test_size=0.2)


def train(X, Y):
    print('Training...')
    class_count = Y.sum(axis=0)
    feature_count = safe_sparse_dot(Y.T, X)
    feature_count_y0 = safe_sparse_dot((Y ^ 1).T, X)

    theta1 = Y.sum(axis=0) / X.shape[0]
    theta_j1 = (feature_count + 1) / (class_count + 2)
    theta_j0 = (feature_count_y0 + 1) / ((X.shape[0] - class_count) + 2)
#    theta_j1 = (feature_count + 1)/(X.shape[0] + 2)
#    theta_j0 = (feature_count_y0 + 1)/(X.shape[0] + 2)
    log_theta_j1 = np.concatenate((np.log(theta_j1), np.log(1 - theta_j1)))
    log_theta_j0 = np.concatenate((np.log(theta_j0), np.log(1 - theta_j0)))

    return theta1, log_theta_j1, log_theta_j0


#feature_count_y0 = safe_sparse_dot((Y^1).T, X)

#ll_theta1 = sum(Y*np.log(theta1) + (1-Y)*np.log(1-theta1))
def predict(X, theta1, log_theta_j1, log_theta_j0):
    pred_Y = np.zeros((X.shape[0], 1))
    print('Predicting...')
    for i in tqdm(range(X.shape[0])):
        X_dense = X[i, :].todense()
        X_flip = X_dense ^ 1

        X_i = np.concatenate((X_dense, X_flip))

        log_sum_1 = np.trace(np.dot(X_i, log_theta_j1.T))
        ll_1 = np.log(theta1) + log_sum_1
        log_sum_0 = np.trace(np.dot(X_i, log_theta_j0.T))
        ll_0 = np.log(1 - theta1) + log_sum_0

#        prob_1 = np.exp(ll_1 - logsumexp(np.matrix([float(ll_1), float(ll_0)])))
#        prob_0 = np.exp(ll_0 - logsumexp(np.matrix([float(ll_1), float(ll_0)])))
#
#        prediction = np.argmax(np.matrix([float(prob_0), float(prob_1)]))
        prediction = np.argmax(np.matrix([float(ll_0), float(ll_1)]))

        pred_Y[i, 0] = prediction

        #print('Predicting for review number: ', i)

    return pred_Y


def main():

    theta1, log_theta_j1_i, log_theta_j0_i = train(X_train, y_train)
    val_y = predict(X_val, theta1, log_theta_j1_i, log_theta_j0_i)
    pred_Y = predict(X_test, theta1, log_theta_j1_i, log_theta_j0_i)
    print("Validation accuracy:", (y_val == val_y).sum() / len(y_val))

    #count = 0
    # with open('prediction_NB.csv', 'w', newline='', encoding='utf-8') as csv_file:
    #     print("Printing prediction to csv file... ")
    #     writer = csv.writer(csv_file)
    #     writer.writerow(['ID', 'Category'])
    #     for i in pred_Y:
    #         writer.writerow([count, int(pred_Y[count])])
    #         count += 1

    #     csv_file.close()


main()
end = timer()
print('Total time = ', (end - start) / 60, ' mins')
# =============================================================================
