"""Statistical Language processing term project
    This script adopts the Naive Bayesian model to predict the label of the posts"""
import csv
import sys
import random
import re
from collections import Counter
from numpy.lib.function_base import vectorize
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
# function for encoding categories
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support

def predict_label(trn_texts, trn_labels, tst_texts):
    
    pass



if __name__ == "__main__":

    train_text, train_label, test_text = [], [], []