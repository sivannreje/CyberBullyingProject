import os
import pandas as pd
import nltk
import collections
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import wordcloud
from itertools import chain
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from CyberBullyingBack import preprocess


def traverse(a):
    if type(a) is list:
        return [''.join(wrd[-1:-(len(wrd)+1):-1]) if type(wrd) is str and len(wrd)>0 and wrd[0] in 'אבגדהוזחטיכלמנסעפצקרשת' else wrd for wrd in a ]
    elif type(a) is str: return traverse([a])[0]
    elif type(a) is set: return set(traverse(list(a)))
    elif type(a) is dict: dict(zip(traverse(a.keys()),traverse(a.values())))
    elif type(a) == type(pd.Series()): return pd.Series(data=traverse(list(a)),index=a.index,name=a.name)
    elif type(a) == type(type(pd.DataFrame())): return a.applymap(lambda x: traverse(x))
    return a

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

    word_counter = collections.Counter(word_frequency)
    most_common_dictionary = word_counter.most_common(number)

    return most_common_dictionary


# get_common_words(df, 10)


def get_post_length(dataframe):
    post_frequency = {}
    for index, row in df.iterrows():
        post_frequency[row.id] = len(word_tokenize(row.text))

    return post_frequency


def create_tf_idf(dataframe, num_of_words):
    # get the text column
    posts = dataframe['text'].tolist()
    dict = {}

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

        dict[post] = [(k, keywords[k]) for k in keywords]
    return dict



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
# create_tf_idf(df, 10)


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


def get_no_abusive_df(df):
    return df.loc[df['cb_level'] == '1']


def create_LDA_model(df, no_topics,name_image):
    global stop_words
    vectorizer = CountVectorizer(min_df=10, max_df=0.6, encoding="cp1255", stop_words=stop_words)
    matrix = vectorizer.fit_transform(df['text'])
    feature_names = vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online',
                                    learning_offset=50., random_state=0).fit(matrix)
    create_word_cloud(no_topics, lda, feature_names,name_image)
    return lda

def create_word_cloud(no_topics, lda, feature_names,name_image):
    global stop_words
    font_path = os.path.join(os.path.join(os.environ['WINDIR'], 'Fonts'), 'ahronbd.ttf')
    for i in range(0, no_topics):
        d = dict(zip(traverse(feature_names), lda.components_[i]))
        wc = wordcloud.WordCloud(background_color='white', font_path=font_path, max_words=100,stopwords=stop_words)
        image = wc.generate_from_frequencies(d)
        image.to_file(name_image+str(i)+'.png')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.figure()
        # plt.show()


def plot_tf_idf_post(dictionary_tf_idf, title, unique=False):
    dic_post = dict(dictionary_tf_idf[title])
    dic_post_travers = {}
    for term,val in dic_post.items():
        dic_post_travers[traverse(term)] = val
    df2 = pd.DataFrame.from_dict(dic_post_travers,orient='index').sort_values(by=0,ascending=False)
    pl = df2.plot(kind='bar',figsize=(15,7),fontsize=8, legend=False,title=traverse(title))
    for p in pl.patches:
        pl.annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.001),fontsize=14)
    plt.show()


    #pd.DataFrame(data=corpus)
    # if unique:
    #     counts = Counter(traverse([w for w in chain(*list(df[0].
    #                                                       apply(lambda x: list(set(x.split(" ")))))) if len(w)>0 and w not in stop_words]))
    # else:
    #     counts = Counter(traverse([w for w in chain(*list(df[0].apply(lambda x: x.split(" ")))) if len(w)>0 and w not in stop_words]))
    #
    # counts = Counter(traverse([w for w in chain]))
    # df2= pd.DataFrame.from_dict(dict(counts.most_common(20)),orient='index').sort_values(by=0,ascending=False)
    # pl = df2.plot(kind='bar',figsize=(15,7),fontsize=8, legend=False,title=title)
    # for p in pl.patches:
    #     pl.annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.001),fontsize=14)
    # plt.show()


# main
stop_words = preprocess.get_stop_words()
all_corpus = 'allData.csv'
data_path = 'data.csv'
cols = ['id', 'time', 'source', 'sub_source', 'writer', 'link', 'text', 'cb_level', 'comment_shared_post']
corpus_frame = pd.read_csv(all_corpus, delimiter=',', names=cols)
corpus_list = preprocess.clean_tokens(corpus_frame)
corpus_frame['text'] = corpus_list
print(get_common_words(corpus_frame,20))

df = pd.read_csv(data_path, delimiter=',', names=cols)
list_posts = preprocess.clean_tokens(df)
df['text'] = list_posts

df_abusive = get_abusive_df(df)
df_no_abusive = get_no_abusive_df(df)

# topic modelling
# lda_result_abusive = create_LDA_model(df, 5,'all_data')
# lda_result_abusive = create_LDA_model(df_abusive, 3,'abusive')
# lda_result_no_abusive = create_LDA_model(df_no_abusive, 5,'no_abusive')

# tf idf
# number_tf_idf = 5
# dictionary_tf_idf = create_tf_idf(df,number_tf_idf)
# my_post = 'לכל מי ששואל  אמא שלך זונה  אמא שלך עוזרת ליהודים להרוס ערים של פלשתים ימח שמם  שבת נפלאה ומרגשת'
# plot_tf_idf_post(dictionary_tf_idf, title=my_post)


