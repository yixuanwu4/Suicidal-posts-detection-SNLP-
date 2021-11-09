"""Statistical Language processing term project
    This script adopts the Naive Bayesian model to predict the label of the posts
    Note: suicidal and a mix of general and depression posts are used in this approach (with depression posts marked as "gen")"""
from preprocessdata import preprocess
from encoder import data_split, countvec_encode, NBmodel, tfidfvec_encode
import random

# Using only suicide and general posts
if __name__ == "__main__":

    suicide_posts, depress_posts, general_posts, suicide_label, depress_label, general_label = preprocess("Suicide_Detection.csv", "reddit_data.csv")
    # mix the posts of depression and general together
    target_posts = depress_posts + general_posts
    target_label = depress_label + general_label
    target_dict = dict(zip(target_posts, target_label))
    allpairs = list(target_dict.items())
    random.shuffle(allpairs)
    target_pairs = allpairs[:10000]
    target_x = [k for k, v in target_pairs]
    # Here mask all depression posts also as "gen"
    target_y = ['general' for i in range(10000)]

    train_posts, train_label, test_posts, test_label = data_split(suicide_posts, target_x, suicide_label, target_y)
    vec_train_x, vec_test_x, vec_train_y, vec_test_y = countvec_encode(train_posts, train_label, test_posts, test_label)
    # print count vectorized result
    print("____________________________")
    print("CountVectorizer preformance: ")
    precision, recall, f_beta = NBmodel(vec_train_x, vec_test_x, vec_train_y, vec_test_y)
    print("By using CountVectorizer, the model reaches:")
    print("Precision: " + str(precision) + "\nRecall: " + str(recall) + "\nF-beta score: " + str(f_beta) )
    
    # print tf-idf vectorized result
    print("____________________________")
    print("TF-IDF preformance: ")
    tf_vec_train_x, tf_vec_test_x, tf_vec_train_y, tf_vec_test_y = tfidfvec_encode(train_posts, train_label, test_posts, test_label)
    precision, recall, f_beta = NBmodel(tf_vec_train_x, tf_vec_test_x, tf_vec_train_y, tf_vec_test_y)
    print("By using TF-IDF Vectorizer, the model reaches:")
    print("Precision: " + str(precision) + "\nRecall: " + str(recall) + "\nF-beta score: " + str(f_beta) )