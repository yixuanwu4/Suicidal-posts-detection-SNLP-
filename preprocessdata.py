"""Statistical Language processing term project
    This script preprocesses the data and put them into the training and testing data"""
import csv
import sys


def preprocess(file1, file2):

    # ----------- DATA PREPROCESSING ---------- #

    global suicide_posts 
    global depress_posts 
    global general_posts 

    suicide_depression_posts = {}

    with open(file1, encoding = 'utf8') as f:
        csvr = csv.reader(f, delimiter=',')

        for row in csvr:
            suicide_depression_posts[row[1]] = row[2] 
        # CHANGE THE TEXT TO LOWER CASE AND COLLECT ALL POSTS
        ori_suicide_posts = [k for k, v in suicide_depression_posts.items() if v == 'suicide']
        ori_depress_posts = [k for k, v in suicide_depression_posts.items() if v == 'non-suicide']
        
        # print(len(suicide_posts), len(depress_posts))  result: 116037
    
    general_posts = {}

    csv.field_size_limit(sys.maxsize)       

    with open(file2, encoding = 'utf8') as z:
        csvr = csv.reader(z, delimiter=',')

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
    ori_general_posts = {x:'general' for x in general_posts}
    
    # BECAUSE THE "GENERAL" POSTS ARE WAY LESS THAN THE SUICIDE AND NON-SUICIDE POSTS (DEPRESSION POSTS)
    # TO BALANCE EACH CLASS AND SHORTEN THE TRAINING TIME, ONLY 10000 OF "GENERAL", "SUICIDE" AND "NON-SUICIDE" POSTS WILL BE KEPt"""
    
    suicide_posts = ori_suicide_posts[:5000]
    depress_posts = ori_depress_posts[:5000]
    general_posts = list(ori_general_posts.keys())[:5000]
    extra_suicide_posts = ori_suicide_posts[5000:10000]
    extra_depression_posts = ori_depress_posts[5000:10000]
    extra_general_posts = list(ori_general_posts.keys())[:5000]
    

    suicide_label = ['suicide' for i in range(5000)]
    depress_label = ['depression' for i in range(5000)]
    general_label = ['general' for i in range(5000)]
    extra_suicide_labels = ['suicide' for i in range(5000)]
    extra_depression_labels = ['depression' for i in range(5000)]
    extra_general_labels = ['general' for i in range(5000)]

    return suicide_posts, depress_posts, general_posts, \
            suicide_label, depress_label, general_label, \
                extra_suicide_posts, extra_depression_posts, extra_general_posts, \
                    extra_suicide_labels, extra_depression_labels, extra_general_labels


    