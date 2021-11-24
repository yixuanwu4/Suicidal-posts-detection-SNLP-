"""Statistical Language processing term project
    This script adopts the multinomial Naive Bayes classifier to predict the label of the posts
    Note: Only suicidal and general posts are used in this approach """
from preprocessdata import preprocess
from encoder import data_split, countvec_encode, NBmodel, tfidfvec_encode, SVMmodel

# Using only suicide and general posts
if __name__ == "__main__":

    suicide_posts, depress_posts, general_posts, suicide_label, depress_label, general_label = preprocess("Suicide_Detection.csv", "reddit_data.csv")

    train_posts, train_label, test_posts, test_label = data_split(suicide_posts, general_posts, suicide_label, general_label)
    vec_train_x, vec_test_x, vec_train_y, vec_test_y = countvec_encode(train_posts, train_label, test_posts, test_label)
    # print(vec_test_y) 
    # print count vectorized result
    print("By using CountVectorizer, the model reaches:")
    print("____________________________")
    print("CountVectorizer preformance: ")
    SVMmodel(vec_train_x, vec_test_x, vec_train_y, vec_test_y)

    
    # print tf-idf vectorized result
    print("____________________________")
    print("TF-IDF preformance: ")
    tf_vec_train_x, tf_vec_test_x, tf_vec_train_y, tf_vec_test_y = tfidfvec_encode(train_posts, train_label, test_posts, test_label)
    SVMmodel(tf_vec_train_x, tf_vec_test_x, tf_vec_train_y, tf_vec_test_y)

"""    The best performance is from TF-IDF with alpha set as 0.01:
        Accuracy: 0.9536250000000001
        Precision: 0.9423356307733128
        Recall: 0.966106850039672
        F1: 0.9540325162212561
"""