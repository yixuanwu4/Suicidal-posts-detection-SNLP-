"""Statistical Language processing term project
    This script adopts the multinomial Naive Bayes classifier to predict the label of the posts
    Note: Only suicidal and general posts are used in this approach """
from preprocessdata import preprocess
from encoder import data_split, countvec_encode_predict, NBmodel

# Using only suicide and general posts
if __name__ == "__main__":

    suicide_posts, depress_posts, general_posts, suicide_label, depress_label, general_label = preprocess("Suicide_Detection.csv", "reddit_data.csv")

    train_posts, train_label, test_posts, test_label = data_split(suicide_posts, general_posts, suicide_label, general_label)
    vec_train_x, vec_test_x, vec_train_y, vec_test_y = countvec_encode_predict(train_posts, train_label, test_posts, test_label)

    NBmodel(vec_train_x, vec_test_x, vec_train_y, vec_test_y)