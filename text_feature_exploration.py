"""This script extracts common words in the text 
and plot the common words of each post class"""
from preprocessdata import preprocess
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from statistics import mean
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


pd.set_option('display.max_columns', 100)
sns.set_style("darkgrid")

suicide_posts, depress_posts, general_posts, suicide_label, depress_label, general_label = preprocess("Suicide_Detection.csv", "reddit_data.csv")


#DEFINING A FUNCTION TO VISUALISE MOST USED WORDS
def plot_most_used_words(dataname, data_series, palette):
    #CHECKING OUT COMMON WORDS IN THE DATA USING CVEC
    cvec = CountVectorizer(stop_words='english')
    cvec.fit(data_series)

    #CREATING A DATAFRAME OF EXTRACTED WORDS
    created_df = pd.DataFrame(cvec.transform(data_series).todense(),
                              columns=cvec.get_feature_names())
    total_words = created_df.sum(axis=0)
    
    #CREATING A FINAL DATAFRAME OF THE TOP 20 WORDS
    top_50_words = total_words.sort_values(ascending = False).head(50)
    top_50_words_df = pd.DataFrame(top_50_words, columns = ["count"])
    #PLOTTING THE COUNT OF THE TOP 20 WORDS
    sns.set_style("white")
    plt.figure(figsize = (15, 8))
    ax = sns.barplot(y= top_50_words_df.index, x="count", data=top_50_words_df, palette = palette)
    
    plt.xlabel("Count", fontsize=9)
    plt.ylabel(dataname, fontsize=9)
    plt.savefig('/home/nauxiy/Workspace/Suicidal-posts-detection-SNLP-/plots/' + dataname + ".png")

    # plt.show()
    

plot_most_used_words("Common Words in Suicidal Posts", suicide_posts, palette="ocean_r")
plot_most_used_words("Common Words in Depression Posts", depress_posts, palette="magma")
plot_most_used_words("Common Words in General Posts", general_posts, palette="Greens_r")

# COLLECT EACT POST'S LENGTH
sui_selftext_length = [len(suicide_posts[i]) for i in range(len(suicide_posts))]
dep_selftext_length = [len(depress_posts[i]) for i in range(len(suicide_posts))]
gen_selftext_length = [len(general_posts[i]) for i in range(len(suicide_posts))]

# CALCULATE AVERAGE POST LENGTH IN EACH CLASS
ave_sui_selftext_length = mean(sui_selftext_length)
ave_dep_selftext_length = mean(dep_selftext_length)
ave_gen_selftext_length = mean(gen_selftext_length)

print("__________________________________")
print("Average length of a suicidal post: {}".format(ave_sui_selftext_length))
print("Average length of a depression post: {}".format(ave_dep_selftext_length))
print("Average length of a general post: {}".format(ave_gen_selftext_length))


# MAPPING POST LENGTH WITH THE LABEL
all_post_length = sui_selftext_length + dep_selftext_length + gen_selftext_length
all_labels = suicide_label + depress_label + general_label 
label_post_dict = {'post length':all_post_length, 'labels': all_labels}
df = pd.DataFrame(label_post_dict)
sns.set_theme(style="ticks", palette="pastel")
plt.figure(figsize = (18, 12))
sns.boxplot(data =df,
               y = 'post length', 
               x = 'labels',
               hue = "labels",
               palette = "magma_r",
               showfliers = False
               );
plt.title("Length of Posts");
plt.xlabel("posts");
plt.ylabel("Number of words");
plt.savefig('/home/nauxiy/Workspace/Suicidal-posts-detection-SNLP-/plots/' + "length_of_posts.png")
plt.show()
