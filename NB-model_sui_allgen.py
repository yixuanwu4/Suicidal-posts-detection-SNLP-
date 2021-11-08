"""Statistical Language processing term project
    This script adopts the Naive Bayesian model to predict the label of the posts
    Note: suicidal and a mix of general and depression posts are used in this approach (with depression posts marked as "gen")"""
from preprocessdata import preprocess
from encoder import data_split, countvec_encode_predict, NBmodel
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
    vec_train_x, vec_test_x, vec_train_y, vec_test_y = countvec_encode_predict(train_posts, train_label, test_posts, test_label)

    NBmodel(vec_train_x, vec_test_x, vec_train_y, vec_test_y)