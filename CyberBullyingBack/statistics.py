import os
import pandas as pd
import nltk
import collections
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


data_path = 'data.csv'
cols = ['id', 'time', 'source', 'sub_source', 'writer', 'link', 'text', 'cb_level', 'comment_shared_post']
df = pd.read_csv(data_path, delimiter=',', names=cols)


def get_common_words(dataframe, number):
    text = dataframe.text.tolist()
    tokens = []
    for post in text:
        tokens = tokens + word_tokenize(post)

    word_frequency = {}

    for word in tokens:
        if word not in word_frequency:
            word_frequency[word] = 1
        else:
            word_frequency[word] += 1

    word_counter = collections.Counter(word_frequency)  # TODO: return the dict and print it separately
    for word, count in word_counter.most_common(number):
        print(word, ": ", count)


# get_common_words(df, 10)


def get_post_length(dataframe):
    post_frequency = {}
    for index, row in df.iterrows():
        post_frequency[row.id] = len(word_tokenize(row.text))

    return post_frequency


def create_tf_idf(dataframe, num_of_words):   # TODO: finish
    # get the text column
    posts = dataframe['text'].tolist()

    # create a vocabulary of words,
    # ignore words that appear in 85% of documents,
    # eliminate stop words
    cv = CountVectorizer(max_df=0.85)  # , stop_words=stopwords)
    word_count_vector = cv.fit_transform(posts)

    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    # you only needs to do this once, this is a mapping of index to
    feature_names = cv.get_feature_names()

    # get the document that we want to extract keywords from
    for post in posts:
        # generate tf-idf for the given document
        tf_idf_vector = tfidf_transformer.transform(cv.transform([post]))

        # sort the tf-idf vectors by descending order of scores
        sorted_items = sort_coo(tf_idf_vector.tocoo())

        # extract only the top n; n here is 10
        keywords = extract_topn_from_vector(feature_names, sorted_items, num_of_words)

        # now print the results
        dict = {}
        dict[post] = [(k, keywords[k]) for k in keywords]
        print(print_tf_idf_dict(dict))


def print_tf_idf_dict(tf_idf_dict):
    for key, value in tf_idf_dict.items():
        print('post: ')
        print(key)
        for v in value:
            print('word: ' + str(v[0]) + ', tf-idf: ' + str(v[1]))


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results
create_tf_idf(df, 10)


def num_of_abusive_per_column(df, column_name):
    # columns: column_name, total_count, total_from_corpus, number_of_abusive, normalized_abusive
    total_count_df = df.groupby(column_name)['cb_level'].apply(lambda x: x.count())
    total_from_corpus_df = df.groupby(column_name)['cb_level'].apply(lambda x: (x.count() / df.shape[0]))
    number_of_abusive_df = df.groupby(column_name)['cb_level'].apply(lambda x: x[x == '3'].count())
    normalized_abusive_df = df.groupby(column_name)['cb_level'].apply(lambda x: (x[x == '3'].count() / x.count()) * 100)

    result = pd.DataFrame({'total_count': total_count_df})\
        .merge(pd.DataFrame({'total_from_corpus': total_from_corpus_df}), on=[column_name], right_index=True)

    result = result\
        .merge(pd.DataFrame({'number_of_abusive': number_of_abusive_df}), on=[column_name], right_index=True)
    result = result\
        .merge(pd.DataFrame({'normalized_abusive': normalized_abusive_df}), on=[column_name], right_index=True)

    return result


def get_abusive_df(df):
    return df.loc[df['cb_level'] == '3']


def create_LDA_model(df):
    pass

# df_ab = num_of_abusive_per_column(df, 'source')
# print(df_ab)
