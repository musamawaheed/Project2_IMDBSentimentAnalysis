import csv
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix


def main():
    training_data = '../Data/training_data.csv'
    test_data = '../Data/test_data.csv'

    train_length = 25000

    trainset = pd.read_csv(training_data)
    trainset = trainset.sample(frac=1).reset_index(drop=True)
    trainset = trainset[0:train_length]
    testset = pd.read_csv(test_data)

    X = trainset['Text']
    y = trainset['Score']
    X_test = testset['Text']

    #X_train, X_vali, y_train, y_vali = train_test_split(X, y, train_size=0.80, test_size=0.2)

    stop_words = ['br', 'in', 'of', 'at', 'a', 'the', 'an', 'is', 'that', 'this', 'to', 'for']
    stop_words2 = ["a", "about", "above", "across", "after", "afterwards", "along", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "anyhow", "anyone", "anything", "anywhere", "are", "around", "as", "at", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "cannot", "co", "con", "could", "couldnt", "de", "describe", "do", "done", "due", "during", "each", "eg", "eight", "eleven", "etc", "even", "few", "fifteen", "fifty", "fill", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "hundred", "ie", "in", "inc", "into", "is", "it", "its", "itself", "ltd", "mill", "name", "nine", "noone", "of", "on", "one", "onto", "or", "our", "ours", "ourselves", "part", "per", "put", "re", "she", "should", "side", "since", "sincere", "six", "sixty", "so", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "until", "up", "upon", "via", "was", "well", "were", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "whoever", "whole", "whom", "whose"]

# PIPELINE
    estimators_empty = [('tfidf', TfidfVectorizer()), ('logreg', LogisticRegression(solver='liblinear'))]
    #estimators_empty = [('count', CountVectorizer()), ('logreg', LogisticRegression(solver='liblinear'))]
    estimators_full = [('tfidf', TfidfVectorizer(max_features=100000, stop_words=stop_words, ngram_range=(1, 2), strip_accents='unicode', token_pattern=r'\w{1,}')), ('logreg', LogisticRegression(solver='liblinear', C=12))]
    #estimators_full = [('tfidf', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict', encoding='utf-8', input='content', lowercase=True, max_df=1.0, max_features=40000, min_df=5, ngram_range=(1, 5), norm='l2', preprocessor=None, smooth_idf=True, stop_words=stop_words, strip_accents='unicode', sublinear_tf=False, token_pattern='\\w{1,}', tokenizer=None, use_idf=True, vocabulary=None)), ('logreg', LogisticRegression(solver='liblinear', C=1))]
    pipe_empty = Pipeline(estimators_empty)
    pipe_full = Pipeline(estimators_full)

# GRID SEARCH
    #gridsearch(pipe_empty, X, y, stop_words)

# RANDOMIZED SEARCH
    #randomsearch(pipe_empty, X, y, stop_words, stop_words2)

# BASIC FUNCTION with train_test_split
    #print(pipe_full.fit(X_train, y_train).score(X_vali, y_vali))


# CROSS VALIDATION
    kf = KFold(n_splits=10, random_state=7500)
    kf.get_n_splits()
    results = []
    for train_index, vali_index in kf.split(X, y):
        # print("TRAIN:", train_index, "Validation:", vali_index)
        print('Cross validating ...')
        X_train, X_vali = X[train_index], X[vali_index]
        y_train, y_vali = y[train_index], y[vali_index]
        results.append(pipe_full.fit(X_train, y_train).score(X_vali, y_vali))
    # CONFUSION MATRIX
        # pipe_full.fit(X_train, y_train)
        # print(confusion_matrix(y_vali, pipe_full.predict(X_vali)))

    print(results)
    print(sum(results) / len(results))

# PREDICTING
    # y_test = pipe_full.predict(X_test)
    # print(y_test)
    # prediction_print(y_test)


def gridsearch(pipe, X, y, stop_words):
    print(pipe.get_params().keys())
    print('Running gridsearch ...')
    # parameters = {'count__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)], 'logreg__C': [0.1, 1], 'logreg__penalty': ['l1', 'l2'], 'logreg__solver': ['liblinear']}
    parameters = {'tfidf__max_features': [100000], 'tfidf__stop_words': [stop_words, None], 'tfidf__ngram_range': [(1, 2), (1, 3)], 'tfidf__token_pattern': [r'\w{1,}'], 'tfidf__strip_accents': ['unicode', 'ascii'], 'logreg__C': [18, 22, 26, 30, 34], 'logreg__solver': ['liblinear']}
    #parameters = {'logreg__C': [0.01, 0.1, 1, 10]}

    clf = GridSearchCV(pipe, parameters, cv=5,
                       scoring='accuracy', n_jobs=5)
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_, clf.best_score_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


def randomsearch(pipe, X, y, stop_words, stop_words2):
    print(pipe.get_params().keys())
    print('Running randomized search ...')
    # parameters = {'count__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)], 'logreg__C': [0.1, 1], 'logreg__penalty': ['l1', 'l2'], 'logreg__solver': ['liblinear']}
    parameters = {'tfidf__lowercase': [True, False], 'tfidf__max_features': [50000, 100000, 150000, None], 'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)], 'tfidf__smooth_idf': [True, False], 'tfidf__token_pattern': [r'\w{1,}', r'\w{2,}', r'\w{3,}'], 'tfidf__stop_words': ['english', stop_words, stop_words2, None], 'tfidf__strip_accents': ['ascii', 'unicode', None], 'logreg__C': [0.1, 1, 5, 10, 15, 20, 25]}

    rand = RandomizedSearchCV(pipe, parameters, cv=5, scoring='accuracy', n_iter=2, random_state=5, n_jobs=5)
    rand.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(rand.best_params_, rand.best_score_)
    print()
    print("Rand scores on development set:")
    print()
    means = rand.cv_results_['mean_test_score']
    stds = rand.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, rand.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


class LemmaTokenizer(object):

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class StemmTokenizer(object):

    def __init__(self):
        self.wnl = PorterStemmer()

    def __call__(self, doc):
        return [self.wnl.stem(t) for t in word_tokenize(doc)]


def prediction_print(y_test):
    count = 0
    with open('prediction_logreg.csv', 'w', newline='', encoding='utf-8') as csv_file:
        print("Printing prediction to csv file... ")
        writer = csv.writer(csv_file)
        writer.writerow(['ID', 'Category'])
        for i in y_test:
            writer.writerow([count, i])
            count += 1

    csv_file.close()


if __name__ == '__main__':
    main()
