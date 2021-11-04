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

    # ----------- Data preprocessing ---------- #

    suicide_depression_posts = {}

    with open('Suicide_Detection.csv') as f:
        csvr = csv.reader(f, delimiter=',')
        header = []
        header = next(csvr)# Extract the field names
        # rows = []
        for row in csvr:
            suicide_depression_posts[row[1]] = row[2] 

        suicide_posts = [k for k, v in suicide_depression_posts.items() if v == 'suicide']
        depress_posts = [k for k, v in suicide_depression_posts.items() if v == 'non-suicide']
        # print(len(suicide_posts), len(depress_posts))  result: 116037
    
    general_posts = {}

    csv.field_size_limit(sys.maxsize)       

    with open('reddit_data.csv') as z:
        csvr = csv.reader(z, delimiter=',')
        header = []
        header = next(csvr)# Extract the field names
        rows = []
        for row in csvr:
            try:
                general_posts[row[1]] = row[2] 
            except:
                continue
    lis = []
    for val in general_posts.values(): 
        if val in lis: 
            continue 
        else:
            lis.append(val)
    
    unique_general_topics = set(general_posts.values())
    # print(unique_general_topics) # {'movies', 'relationships', 'pcmasterrace', 'nfl', 'news'}

    for eachtopic in unique_general_topics:
        postsnum = [k for k, v in general_posts.items() if v == eachtopic]
        # print(len(postsnum)) # 4161, 8923, 10268, 8819, 6075
    
    # As these general posts are not the target suicidal posts, they're all labelled as 'general'
    general_posts = {x:'general' for x in general_posts}
    print(general_posts)

    # tmr normalize the text, remove irrelevant chars and stop words, clean text data 




    train_text, train_label, test_text = [], [], []