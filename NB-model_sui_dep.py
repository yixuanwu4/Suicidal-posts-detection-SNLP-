"""Statistical Language processing term project
    This script adopts the multinomial Naive Bayes classifier to predict the label of the posts
    Note: Only suicidal and depression posts are used in this approach """
from preprocessdata import preprocess
from encoder import data_split, countvec_encode, NBmodel, tfidfvec_encode

# Using only suicide and general posts
if __name__ == "__main__":

    suicide_posts, depress_posts, general_posts, suicide_label, depress_label, general_label = preprocess("Suicide_Detection.csv", "reddit_data.csv")

    train_posts, train_label, test_posts, test_label = data_split(suicide_posts, depress_posts, suicide_label, depress_label)
    vec_train_x, vec_test_x, vec_train_y, vec_test_y = countvec_encode(train_posts, train_label, test_posts, test_label)
    # print count vectorized result
    print("By using CountVectorizer, the model reaches:")
    NBmodel(vec_train_x, vec_test_x, vec_train_y, vec_test_y)
    
    # print tf-idf vectorized result
    tf_vec_train_x, tf_vec_test_x, tf_vec_train_y, tf_vec_test_y = tfidfvec_encode(train_posts, train_label, test_posts, test_label)
    print("By using TF-IDF Vectorizer, the model reaches:")
    NBmodel(tf_vec_train_x, tf_vec_test_x, tf_vec_train_y, tf_vec_test_y)