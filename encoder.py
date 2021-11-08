import csv
import sys
import random
import re
from collections import Counter
from numpy.lib.function_base import vectorize
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# function for encoding categories
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from preprocessdata import preprocess


def data_split(post1, post2, label1, label2):
    target_posts = post1 + post2
    targetlabels = label1 + label2
    targetdict = dict(zip(target_posts, targetlabels))
    allpairs = list(targetdict.items())
    random.shuffle(allpairs)

    # split 20% of the data as test data and the rest as training data
    train_pairs = allpairs[:8000]
    test_pairs = allpairs[8000:]

    train_posts = [k for k, v in train_pairs]
    train_label = [v for k, v in train_pairs]
    test_posts = [k for k, v in test_pairs]
    test_label = [v for k, v in test_pairs]

    return train_posts, train_label, test_posts, test_label

def countvec_encode_predict(train_posts, train_label, test_posts, test_label):
    # pull the data into vectors
    vectorizer = CountVectorizer(max_df=0.25, ngram_range=(2, 3))
    train_x = vectorizer.fit_transform(train_posts)
    test_x = vectorizer.transform(test_posts)
    
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_label)
    test_y = encoder.transform(test_label)

    return train_x, test_x, train_y, test_y

def NBmodel(train_x, test_x, train_y, test_y):

    nb = MultinomialNB(alpha = 1)
    nb.fit(train_x, train_y)
    print(nb.get_params(deep=True))

    pred_y = nb.predict(test_x)
    precision, recall, f1_xcore, x = precision_recall_fscore_support(test_y, pred_y, average='macro')
    print("Precision: " + str(precision) + "\nRecall: " + str(recall) + "\nF-beta score: " + str(f1_xcore) )


