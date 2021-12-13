"""Statistical Language processing term project
    This script adopts the multinomial Naive Bayes classifier to predict the label of the posts
    Note: Only suicidal and general posts are used in this approach """
from preprocessdata import preprocess
from encoder import data_split, countvec_encode, NBmodel, tfidfvec_encode, SVMmodel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,recall_score,precision_score,f1_score, confusion_matrix
from joblib import parallel_backend


# n_jobs can be set to the number of available threads on the machine. I set it to 16 here to speed up the process a bit, but ensure no overload while running this code
with parallel_backend('threading', n_jobs=16):
    sui_gen_model = None

    # Using only suicide and general posts
    if __name__ == "__main__":

        suicide_posts, depress_posts, general_posts,suicide_label, depress_label, general_label, extra_suicide_posts, extra_depression_posts, extra_general_posts, extra_suicide_labels, extra_depression_labels, extra_general_labels = preprocess("Suicide_Detection.csv", "reddit_data.csv")
        train_posts, train_label, test_posts, test_label = data_split(suicide_posts, general_posts, suicide_label, general_label)
        vec_train_x, vec_test_x, vec_train_y, vec_test_y = countvec_encode(train_posts, train_label, test_posts, test_label)
        # print(vec_test_y) 
        allf1 = []
        allmodels = []
        # print count vectorized result
        print("By using CountVectorizer, the model reaches: ")
        print("____________________________")
        print("CountVectorizer preformance: ")
        precision_cv_nb, recall_cv_nb, f1_xcore_cv_nb, target_cv_nb = NBmodel(vec_train_x, vec_test_x, vec_train_y, vec_test_y)
        allf1.append(f1_xcore_cv_nb)
        allmodels.append(target_cv_nb)

        precision_cv_svm, recall_cv_svm, f1_xcore_cv_svm, target_cv_svm = SVMmodel(vec_train_x, vec_test_x, vec_train_y, vec_test_y)
        allf1.append(f1_xcore_cv_svm)
        allmodels.append(target_cv_svm)
        
        # print tf-idf vectorized result
        print("____________________________")
        print("TF-IDF preformance: ")
        tf_vec_train_x, tf_vec_test_x, tf_vec_train_y, tf_vec_test_y = tfidfvec_encode(train_posts, train_label, test_posts, test_label)

        precision_tf_nb, recall_tf_nb, f1_xcore_tf_nb, target_tf_nb = NBmodel(tf_vec_train_x, tf_vec_test_x, tf_vec_train_y, tf_vec_test_y)
        allf1.append(f1_xcore_tf_nb)
        allmodels.append(target_tf_nb)

        precision_tf_svm, recall_tf_svm, f1_xcore_tf_svm, target_tf_svm = SVMmodel(tf_vec_train_x, tf_vec_test_x, tf_vec_train_y, tf_vec_test_y)
        allf1.append(f1_xcore_tf_svm)
        allmodels.append(target_tf_svm)

        tmpdict = dict(zip(allf1, allmodels))
        target = max(tmpdict.keys())
        sui_gen_model = tmpdict[target]

        x1, x2, x3, x4 = data_split(extra_depression_posts, extra_general_posts, extra_depression_labels, extra_general_labels)
        

        # APPLY THE BEST DEP_GEN_MODEL TO THE EXTRA SUICIDAL AND GENERAL POSTS
        new_train_posts = x1 + x3
        new_train_label = x2 + x4
        # AS THE RESULT SHOWS THAT TFIDF WORKS BETTER THAN COUNT VECTORIZER, THE NEW TEST DATA ARE APPLIED TO TFIDFVEC
        # THE train_posts, train_label ARE THE "OLD DATA", THE FOLLOWING LINE USES THE WHOLE EXTRA DEPRESSION AND GENERAL POSTS AS TEST DATA
        new_vec_train_x, new_vec_test_x, new_vec_train_y, new_vec_test_y = tfidfvec_encode(train_posts, train_label, new_train_posts, new_train_label)
        # USING THE WHOLE EXTRA DATA AS TEST SET

        y_changed = sui_gen_model.predict(new_vec_test_x)

        tn, fp, fn, tp = confusion_matrix(new_vec_test_y, y_changed).ravel()
        # print(str(tn), str(fp), str(fn), str(tp))
        print('###########################################')
        print('After applying the model to the new depression and general posts, there are ' + str(tp) + ' depression posts which are also identified as suicidal posts.')
        print('Among all 5000 depression posts, ' +  str(tp*100/5000) + '%' + ' of which can be categorized as with a suicidal tendency. ')