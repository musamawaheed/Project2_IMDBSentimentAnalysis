import csv
import random
from random import sample
import string
import numpy as np
from tqdm import tqdm


def main():
    training_data = '../Data/training_data.csv'
    test_data = '../Data/test_data.csv'
    data_length = 25000  # Number of used reviews from training data (max = 25,000)
    split_ratio = 0.80
    BoW_length = 30000
    laplace = 2  # Factor for additive smoothing
    test_length = 25000  # Number of guesses on testset (max = 25,000)

    dataset = loadCSV(training_data)
    dataset = sample(dataset, data_length)

    prepared_data = text_processing(dataset)

    train_set, vali_set = split_dataset(prepared_data, split_ratio)

    countlist = count_words(train_set, BoW_length)

    X_train = bag_of_words(train_set, countlist, BoW_length)

    X_vali = bag_of_words(vali_set, countlist, BoW_length)

    theta_1, theta_j_1, theta_j_0 = thetas(train_set, X_train, laplace)

    predict = prediction(X_vali, theta_1, theta_j_1, theta_j_0)

    accuracy(predict, vali_set)

    guess(test_data, test_length, countlist, theta_1, theta_j_1, theta_j_0, BoW_length)


def loadCSV(training_data):
    with open(training_data, 'r', encoding='utf-8') as data_train:
        reader_train = csv.reader(data_train)
        all_data = list(reader_train)
        dataset = list()
        for i in range(1, len(all_data)):
            dataset.append(all_data[i])
    return dataset


def text_processing(dataset):
    for text in dataset:
        text[1] = text[1].lower()
        text[1] = text[1].translate(text[1].maketrans('', '', string.punctuation))
        text[1] = text[1].split()
        # print(text[:2])
    return dataset


def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = list()
    vali_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(vali_set))
        train_set.append(vali_set.pop(index))
    print(('Split {0} rows into {1} for training and {2} for validating.').format(len(dataset), len(train_set), len(vali_set)))
    return train_set, vali_set


def count_words(train_set, BoW_length):
    alltext = list()
    countdict = dict()
    for element in train_set:
        alltext.append(element[1])

    for review in alltext:
        for words in review:
            countdict[words] = countdict.get(words, 0) + 1
    countlist = sorted([(v, k) for k, v in countdict.items()], reverse=True)
    print('Number of different words in training data: ', len(countlist))
    print('Used for bag of words feature: Most frequent', BoW_length)
    countlist = countlist[:BoW_length]
    return countlist


def bag_of_words(train_set, countlist, BoW_length):
    print("Calculating...")
    bias = 1.0
    X = np.empty((0, BoW_length + 1))
    for review in train_set:
        x_BoW = np.zeros(BoW_length)
        txt = review[1]
        for word in txt:
            idx = [countlist.index(tupl) for tupl in countlist if tupl[1] == word]
            try:
                x_BoW[idx] = 1
            except:
                continue
        x_row = np.append(x_BoW, bias)
        X = np.append(X, [x_row], axis=0)
    #print("BoW done")
    print("Shape of X:", X.shape)
    return X


def thetas(train_set, X_train, laplace):
    print("Calculating thetas...")
    y_train = np.empty(0)
    theta_j_1 = np.empty(0)
    theta_j_0 = np.empty(0)
    for review in train_set:
        rating = int(review[2])
        y_train = np.append(y_train, rating)
    # print("Shape of y_train: ", y_train.shape)
    theta_1 = y_train.sum() / len(train_set)
    # print("Theta_1 = ", theta_1)

    for x_column in X_train.T:
        t1dot = np.dot(x_column, y_train)
        t1 = (t1dot.sum() + 1) / (y_train.sum() + laplace)
        theta_j_1 = np.append(theta_j_1, t1)
        # print(t1dot.sum() / y_train.sum())
        t0dot = np.dot(x_column, (1 - y_train))
        t0 = (t0dot.sum() + 1) / ((len(y_train) - y_train.sum()) + laplace)
        theta_j_0 = np.append(theta_j_0, t0)
        # print(t0dot.sum() / (len(y_train) - y_train.sum()))

    # print(theta_j_1)
    # print(theta_j_0)
    return theta_1, theta_j_1, theta_j_0


def prediction(X, theta_1, theta_j_1, theta_j_0):
    theta_j_1 = np.asmatrix(theta_j_1)
    theta_j_0 = np.asmatrix(theta_j_0)
    log_theta_j_1 = np.concatenate((np.log(theta_j_1), np.log(1 - theta_j_1)))
    log_theta_j_0 = np.concatenate((np.log(theta_j_0), np.log(1 - theta_j_0)))
    predict = np.empty(0)
    print('Predicting...')
    for row in X:
        row = np.asmatrix(row)
        X_flip = 1 - row
        X_i = np.concatenate((row, X_flip), axis=0)
        log_sum_1 = np.trace(np.dot(X_i, log_theta_j_1.T))
        ll_1 = np.log(theta_1) + log_sum_1
        log_sum_0 = np.trace(np.dot(X_i, log_theta_j_0.T))
        ll_0 = np.log(1 - theta_1) + log_sum_0
        prediction = np.argmax(np.matrix([float(ll_0), float(ll_1)]))
        predict = np.append(predict, prediction)
    return predict


def accuracy(predict, vali_set):
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(vali_set)):
        # print(vali_set[i][2], prediction[i])
        if int(vali_set[i][2]) == 0 and predict[i] == 0:
            TN += 1
        if int(vali_set[i][2]) == 1 and predict[i] == 1:
            TP += 1
        if int(vali_set[i][2]) == 0 and predict[i] == 1:
            FP += 1
        if int(vali_set[i][2]) == 1 and predict[i] == 0:
            FN += 1
            # print(correct)
    accy = ((TN + TP) / len(vali_set)) * 100.0
    print(('Accuracy: {}%').format(accy))
    print(('TP:{}, TN:{}, FP:{}, FN:{}').format(TP, TN, FP, FN))


def guess(test_data, test_length, countlist, theta_1, theta_j_1, theta_j_0, BoW_length):
    print("Predicting on test-set...")
    testset = loadCSV(test_data)
    testset = testset[0:test_length]
    print(len(testset))
    prepared_test_data = text_processing(testset)
    X_test = bag_of_words(prepared_test_data, countlist, BoW_length)
    guessing = prediction(X_test, theta_1, theta_j_1, theta_j_0)
    # print(guessing)
    count = 0
    with open('prediction_NB.csv', 'w', newline='', encoding='utf-8') as csv_file:
        print("Printing prediction to csv file... ")
        writer = csv.writer(csv_file)
        writer.writerow(['ID', 'Category'])
        for i in guessing:
            i = int(i)
            writer.writerow([count, i])
            count += 1
    csv_file.close()


if __name__ == '__main__':
    main()
