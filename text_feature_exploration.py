"""This script extracts common words in the text 
and plot the common words of each post class"""
from preprocessdata import preprocess
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image



pd.set_option('display.max_columns', 100)
sns.set_style("darkgrid")


suicide_posts, depress_posts, general_posts, suicide_label, depress_label, general_label = preprocess("Suicide_Detection.csv", "reddit_data.csv")

# Create CountVectorizer, which create bag-of-words model.
# stop_words : Specify language to remove stopwords. 
sui_vectorizer = CountVectorizer(stop_words='english')

# Learn vocabulary in sentences. 
sui_vectorizer.fit(suicide_posts)

# Get dictionary. 
sui_vectorizer.get_feature_names()

#DEFINING A FUNCTION TO VISUALISE MOST USED WORDS
def plot_most_used_words(dataname, data_series, palette):
    #CHECKING OUT COMMON WORDS IN r/SuicideWatch USING CVEC
    cvec = CountVectorizer(stop_words='english')
    cvec.fit(data_series)
    #CREATING A DATAFRAME OF EXTRACTED WORDS
    created_df = pd.DataFrame(cvec.transform(data_series).todense(),
                              columns=cvec.get_feature_names())
    total_words = created_df.sum(axis=0)
    
    #CREATING A FINAL DATAFRAME OF THE TOP 20 WORDS
    top_20_words = total_words.sort_values(ascending = False).head(20)
    top_20_words_df = pd.DataFrame(top_20_words, columns = ["count"])
    #PLOTTING THE COUNT OF THE TOP 20 WORDS
    sns.set_style("white")
    plt.figure(figsize = (15, 8))
    ax = sns.barplot(y= top_20_words_df.index, x="count", data=top_20_words_df, palette = palette)
    
    plt.xlabel("Count", fontsize=9)
    plt.ylabel('Common Words in ' + dataname, fontsize=9)
    plt.savefig('/home/nauxiy/Workspace/Suicidal-posts-detection-SNLP-/plots/' + dataname + ".png")

    plt.show()
    

plot_most_used_words("Common Words in Suicidal Posts", suicide_posts, palette="ocean_r")
plot_most_used_words("Common Words in Depression Posts", depress_posts, palette="ocean_r")
plot_most_used_words("Common Words in General Posts", general_posts, palette="ocean_r")
