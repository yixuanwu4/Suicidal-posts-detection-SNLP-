import random
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# function for encoding categories
from sklearn import svm
# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,recall_score,precision_score,f1_score


random.seed(14)

def data_split(post1, post2, label1, label2):
    target_posts = post1 + post2
    targetlabels = label1 + label2
    targetdict = dict(zip(target_posts, targetlabels))
    allpairs = list(targetdict.items())
    random.shuffle(allpairs)

    # SPLIT 20% OF THE DATA AS TEST DATA AND THE REST AS TRAINING DATA
    train_pairs = allpairs[:8000]
    test_pairs = allpairs[8000:]

    train_posts = [k for k, v in train_pairs]
    train_label = [v for k, v in train_pairs]
    test_posts = [k for k, v in test_pairs]
    test_label = [v for k, v in test_pairs]

    for index, label in enumerate(train_label):
        if label == 'general':
            train_label[index] = 0
        else:
            train_label[index] = 1
    for index, label in enumerate(test_label):
        if label == 'general':
            test_label[index] = 0
        else:
            test_label[index] = 1

    return train_posts, train_label, test_posts, test_label

def countvec_encode(train_posts, train_label, test_posts, test_label):
    # PULL THE DATA INTO VECTORS
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    train_x = vectorizer.fit_transform(train_posts)
    test_x = vectorizer.transform(test_posts)
    
    train_y = train_label
    test_y = test_label

    return train_x, test_x, train_y, test_y

def tfidfvec_encode(train_posts, train_label, test_posts, test_label):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    train_x = vectorizer.fit_transform(train_posts)
    test_x = vectorizer.transform(test_posts)
    
    # encoder = LabelEncoder()
    # train_y = encoder.fit_transform(train_label)
    # test_y = encoder.transform(test_label)
    train_y = train_label
    test_y = test_label

    return train_x, test_x, train_y, test_y


def NBmodel(train_x, test_x, train_y, test_y):

    # find optimal alpha with grid search
    alpha = [0.0005, 0.001, 0.005, 0.01, 0.1, 1, 10, 20, 100, 1000]
    allresult = {}
    targetalphas = {}
    bestmodel = {}
    for eachalpha in alpha:
        nb = MultinomialNB(alpha = eachalpha)
        nb.fit(train_x, train_y)
   
        cv_accuracy = cross_val_score(nb, train_x, train_y, scoring='accuracy', cv=8)
        cv_precision = cross_val_score(nb, train_x, train_y, scoring='precision', cv=8)
        cv_recall = cross_val_score(nb, train_x, train_y, cv=8, scoring='recall')
        cv_f1 = cross_val_score(nb, train_x, train_y, cv=8, scoring='f1')
        allresult[eachalpha] = "Accuracy: " + str(np.mean(cv_accuracy)) \
        + "\nPrecision: " + str(np.mean(cv_precision)) + "\nRecall: " + str(np.mean(cv_recall)) \
        + "\nF1: " + str(np.mean(cv_f1))
        
        if np.mean(cv_f1) > 0.9 and np.mean(cv_precision) > 0.9 and np.mean(cv_recall) > 0.9:
            # SAVE ALL ALPHA SCORES WHICH LEADS TO HIGHER THAN 0.9 ACCURACY, PRECISION AND RECALL
            targetalphas[eachalpha] = np.mean(cv_f1)
            bestmodel[eachalpha] = nb
        else:
            targetalphas[eachalpha] = np.mean(cv_f1)
            bestmodel[eachalpha] = nb
    
    bestalpha = max(targetalphas, key=targetalphas.get)

    targetnb = bestmodel[bestalpha]

    pred_y = targetnb.predict(test_x)
    accuracy = accuracy_score(test_y,pred_y)
    precision, recall, f1_xcore, x = precision_recall_fscore_support(test_y, pred_y, average = 'binary', pos_label=1)
    print("####################################")
    print("By using MultinomialNB, the best parameter and its evaluation metrics are:")
    print("Best Alpha: " + str(bestalpha) + "\nAccuracy: " + str(accuracy) + "\nPrecision: " + str(precision) + "\nRecall: " + str(recall) + "\nF-beta score: " + str(f1_xcore) + "\n")

    return precision, recall, f1_xcore, targetnb
    
def SVMmodel(train_x, test_x, train_y, test_y):

    # defining parameter range
    tuned_parameters =  [{ 'C': [0.001, 0.01, 0.1, 10, 25, 50, 100, 1000]}]              
    # tuned_parameters =  [{ 'C': [ 0.1]}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=8,
                       scoring='f1')
   
    clf.fit(train_x, train_y)

    # PREDICT VALUE BASED ON UPDATED PARAMETERS
    y_pred_result = clf.predict(test_x)
    # NEW MODEL EVALUATION METRICS 
    print("####################################")
    print("By using SVM model, the best parameter and its evaluation metrics are:")
    bestpara = clf.best_estimator_
    accuracy = accuracy_score(test_y,y_pred_result)
    precision = precision_score(test_y,y_pred_result)
    recall = recall_score(test_y,y_pred_result)
    f1 = f1_score(test_y,y_pred_result)
    print('Best parameters: ' + str(bestpara))
    print('Accuracy Score : ' + str(accuracy))
    print('Precision Score : ' + str(precision))
    print('Recall Score : ' + str(recall))
    print('F1 Score : ' + str(f1))
    return precision, recall, f1, clf

    
    


