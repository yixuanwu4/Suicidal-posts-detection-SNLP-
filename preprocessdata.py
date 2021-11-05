"""Statistical Language processing term project
    This script preprocesses the data and put them into the training and testing data"""
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


def preprocess(file1, file2):

    # ----------- Data preprocessing ---------- #

    global suicide_posts 
    global depress_posts 
    global general_posts 

    suicide_depression_posts = {}

    with open(file1) as f:
        csvr = csv.reader(f, delimiter=',')
        header = []
        # Extract the field names
        header = next(csvr)
        for row in csvr:
            suicide_depression_posts[row[1]] = row[2] 

        suicide_posts = [k for k, v in suicide_depression_posts.items() if v == 'suicide']
        depress_posts = [k for k, v in suicide_depression_posts.items() if v == 'non-suicide']
        # print(len(suicide_posts), len(depress_posts))  result: 116037
    
    general_posts = {}

    csv.field_size_limit(sys.maxsize)       

    with open(file2) as z:
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
    
    for i in unique_general_topics:
        postsnum = []
        for k, v in general_posts.items():
            if v == i:
                postsnum.append(k)
        # print(i, len(postsnum)) # 4161, 8923, 10268, 8819, 6075
    
    # As these general posts are not the target suicidal posts, they're all labelled as 'general'
    general_posts = {x:'general' for x in general_posts}
    
    # because the "general" posts are way less than the suicide and non-suicide posts (depression posts)
    # to balance each class and shorten the training time, only 10000 of "general", "suicide" and "non-suicide" posts will be kept"""
    
    suicide_posts = suicide_posts[:10000]
    depress_posts = depress_posts[:10000]
    general_posts = list(general_posts.keys())[:10000]
    
    # remove the unnecessary meaningless chars 
    def normalize_text(s):
        # just in case
        s = str(s)
        s = s.lower()

        # remove punctuation that is not word-internal (e.g., hyphens, apostrophes) 
        s = re.sub('\s\W', ' ', s)
        s = re.sub('\W\s', ' ', s)

        # make sure no double spaces 
        s = re.sub('\s+', ' ', s)
        s = s.replace('&#039;', ' a')

        # remove non-ASCII chars 
        s = re.sub(r'[^\x00-\x7F]+', '', s)

        return s

    suicide_posts = [normalize_text(s) for s in suicide_posts]
    depress_posts = [normalize_text(s) for s in depress_posts]
    general_posts = [normalize_text(s) for s in general_posts]

    suicide_label = ['suicide' for i in range(10000)]
    depress_label = ['non-suicide' for i in range(10000)]
    general_label = ['general' for i in range(10000)]


    return suicide_posts, depress_posts, general_posts, suicide_label, depress_label, general_label



    

suicide_posts, depress_posts, general_posts, suicide_label, depress_label, general_label = preprocess("Suicide_Detection.csv", "reddit_data.csv")
