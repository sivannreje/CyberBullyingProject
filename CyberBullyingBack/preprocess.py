import csv
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter


def file_to_list(path, encoding='cp1255',header=True):
    try:
        with open(path, mode='r',encoding=encoding) as infile:
            myList = [line.strip('\n') for line in infile]
    except UnicodeDecodeError as e:
        with open(path,mode='r', encoding='utf-8') as infile:
            myList = [line.strip('\n') for line in infile]
    return myList


def get_stop_words():
    stop_words = file_to_list(r'C:\Users\shake\Desktop\לימודים\פרויקט בריונות ברשת\stop_words.txt')
    return stop_words


def clean_tokens(df):
    text = df.text.tolist()
    number_posts = len(text)
    for index_post in range(1,number_posts):
        text[index_post] = re.sub(r'[^א-ת]', ' ', text[index_post]).strip().rstrip()
    return text


def remove_stop_words(data_frame, my_stop_words):
    print("remove_stop_words")
    text = data_frame.text.tolist()
    all_words = " ".join(text)
    all_words = word_tokenize(all_words)
    for word in my_stop_words:
        if word in all_words:
            all_words.remove(word)

    top_words = dict(Counter(all_words))
    return top_words


def find_df(dataframe,threshold):
    global stop_words
    print("find_df")
    text = dataframe.text.tolist()
    term_df = {}
    number_posts = len(text)
    print(number_posts)

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

    for token, posts in term_df.items():
        df = len(posts)
        df_normal = float(df / number_posts)
        if df_normal > threshold:
            stop_words.append(token)

    print(stop_words)


def word2vec():
    print("word2vec")


def clean_data(df):
    print("clean_data")
    top_words = remove_stop_words(df)
    print("number top words", len(top_words))


# data_path = 'data.csv'
# cols = ['id', 'time', 'source', 'sub_source', 'writer', 'link', 'text', 'cb_level', 'comment_shared_post']
# df = pd.read_csv(data_path, delimiter=',', names=cols)
# df_abusive = statistics.get_abusive_df(df)
#
# clean_data(df)
