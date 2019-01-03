import statistics
import csv

import pandas as pd
from nltk.tokenize import word_tokenize

def remove_stop_words():
    print("remove_stop_words")

def find_df(dataframe):
    print("find_df")
    text = dataframe.text.tolist()
    term_df = {}
    number_posts = len(text)
    print(range)

    for index_post in range(1, number_posts):
        tokens = word_tokenize(text[index_post])
        for token in tokens:
            if token in term_df:
                list_posts = term_df[token]
                if index_post not in list_posts:
                    term_df[token].append(index_post)

            else:
                term_df[token] = []
                term_df[token].append(index_post)

    for token,posts in term_df.items():
        term_df[token] = len(posts)


    print(term_df)




def word2vec():
    print("word2vec")

def clean_data(df):
    print("clean_data")


data_path = 'data.csv'
cols = ['id', 'time', 'source', 'sub_source', 'writer', 'link', 'text', 'cb_level', 'comment_shared_post']
df = pd.read_csv(data_path, delimiter=',', names=cols)
df_abusive = statistics.get_abusive_df(df)
common_words = statistics.get_common_words(df, 100)
find_df(df)
print(common_words)
