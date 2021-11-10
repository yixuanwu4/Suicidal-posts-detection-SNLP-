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
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords


def preprocess(file1, file2):

    # ----------- DATA PREPROCESSING ---------- #

    global suicide_posts 
    global depress_posts 
    global general_posts 

    suicide_depression_posts = {}

    with open(file1) as f:
        csvr = csv.reader(f, delimiter=',')
        header = []
        # EXTRACT THE FIELD NAME
        header = next(csvr)
        for row in csvr:
            suicide_depression_posts[row[1]] = row[2] 
        # CHANGE THE TEXT TO LOWER CASE AND COLLECT ALL POSTS
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
                if row[2] != 'pcmasterrace':
                    general_posts[row[1]] = row[2] # REMOVE POSTS UNDER 'pcmasterrace' WHICH CONTAINS TOO MUCH PC-RELATED WORDS WHICH HAS HIGH FREQUENCY
                 
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
    
    # SET LABEL "GENERAL" TO ALL POSTS FROM THE GENERAL TOPICS
    general_posts = {x:'general' for x in general_posts}
    
    # because the "general" posts are way less than the suicide and non-suicide posts (depression posts)
    # to balance each class and shorten the training time, only 10000 of "general", "suicide" and "non-suicide" posts will be kept"""
    
    suicide_posts = suicide_posts[:10000]
    depress_posts = depress_posts[:10000]
    general_posts = list(general_posts.keys())[:10000]
    
    # # Stop words are nothing but high-frequency words
    STOP_WORDS_LIST =stopwords.words('english')

    # remove the unnecessary meaningless chars 
    def normalize_text(s):
        # convert all posts into lowercase
        s = str(s)
        s = s.lower()
        s = ' '.join([word for word in s.split() if word not in STOP_WORDS_LIST])

    
        # some further cleaning
        # specific
        s = re.sub(r"won\'t", "will not", s)
        s = re.sub(r"can\'t", "can not", s)

        # general
        s = re.sub(r"n\’t", " not", s)
        s = re.sub(r"\’re", " are", s)
        s = re.sub(r"\’s", " is", s)
        s = re.sub(r"\’d", " would", s)
        s = re.sub(r"\’ll", " will", s)
        s = re.sub(r"\’t", " not", s)
        s = re.sub(r"\’ve", " have", s)
        s = re.sub(r"\’m", " am", s)
        s = re.sub(r"\'m", " am", s)
        s = re.sub(r"n\'t", " not", s)
        s = re.sub(r"\'re", " are", s)
        s = re.sub(r"\'s", " is", s)
        s = re.sub(r"\'d", " would", s)
        s = re.sub(r"\'ll", " will", s)
        s = re.sub(r"\'t", " not", s)
        s = re.sub(r"\'ve", " have", s)


        return s
    
    # print(normalize_text("I’m so lostHello, "))
    suicide_posts = [normalize_text(s) for s in suicide_posts]

    depress_posts = [normalize_text(s) for s in depress_posts]
    general_posts = [normalize_text(s) for s in general_posts]

    suicide_label = ['suicide' for i in range(10000)]
    depress_label = ['depression' for i in range(10000)]
    general_label = ['general' for i in range(10000)]


    return suicide_posts, depress_posts, general_posts, suicide_label, depress_label, general_label


    

suicide_posts, depress_posts, general_posts, suicide_label, depress_label, general_label = preprocess("Suicide_Detection.csv", "reddit_data.csv")
alltokens = [word_tokenize(x) for x in general_posts ]
alltokens = [item for sublist in alltokens for item in sublist]
Counter = Counter(alltokens)
most_occure = Counter.most_common(20)

# print(most_occure)