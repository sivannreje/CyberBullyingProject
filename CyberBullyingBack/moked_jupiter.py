
# coding: utf-8

# In[157]:

import pandas as pd
from collections import Counter
from itertools import chain
import wordcloud
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

rcParams = matplotlib.rcParams


rcParams['font.sans-serif'] = ['Arial']
rcParams['font.family'] = 'sans-serif'


def traverse(a):
    if type(a) is list:
        return [''.join(wrd[-1:-(len(wrd)+1):-1]) if type(wrd) is str and len(wrd)>0 and wrd[0] in 'אבגדהוזחטיכלמנסעפצקרשת' else wrd for wrd in a ]
    elif type(a) is str: return traverse([a])[0]
    elif type(a) is set: return set(traverse(list(a)))
    elif type(a) is dict: dict(zip(traverse(a.keys()),traverse(a.values())))
    elif type(a) == type(pd.Series()): return pd.Series(data=traverse(list(a)),index=a.index,name=a.name)
    elif type(a) == type(type(pd.DataFrame())): return a.applymap(lambda x: traverse(x))
    return a

def fileToList(path, encoding='cp1255',header=True):
    try:
        with open(path, mode='r',encoding=encoding) as infile:
            myList = [line.strip('\n') for line in infile]
    except UnicodeDecodeError as e:
        with open(path,mode='r', encoding='utf-8') as infile:
            myList = [line.strip('\n') for line in infile]
    return myList
    
    
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print('\nTopic Nr.%d:' % int(topic_idx + 1)) 
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
              +' | ' for i in topic.argsort()[:-no_top_words - 1:-1]]))
        
def display_topic(model, feature_names, no_top_words,topic_idx):
        print('\nTopic Nr.%d:' % int(topic_idx + 1)) 
        print(''.join([feature_names[i] + ' ' + str(round(model.components_[topic_idx][i], 2))
              +' | ' for i in model.components_[topic_idx].argsort()[:-no_top_words - 1:-1]]))        


def plotPieAndBar(df,title,isSide=True,isGrouped=True,plot_method='both',barSize=(18,8),pieSize=(20,20),using_others=True, sort=False):
    oneOrTwo = 2 if plot_method=='both' else 2
    if isSide: fig, axes = plt.subplots(nrows=1, ncols=oneOrTwo)
    else: fig, axes = plt.subplots(nrows=oneOrTwo, ncols=1)
#     if not using_others:
#         df2 = df.filter(lambda x: print(x))
    df2 = df.size() if isGrouped else df
    if plot_method != 'pie':
        df2.plot(kind='bar',ax=axes[0],figsize=barSize,fontsize=8,sort_columns=sort)
    if plot_method != 'bar':
        df2.plot(kind='pie',ax=axes[oneOrTwo-1],stacked=True, autopct='%1.1f%%',startangle=155, 
               figsize=pieSize, title=title, subplots=True,fontsize=8,sort_columns=sort)
    if plot_method != 'pie':
        for p in axes[0].patches:
            axes[0].annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.001),fontsize=14)
    plt.show()
    


# ### FETCH DATA

# In[158]:

no_features = 1000
    
dataset=pd.read_excel("F:\Erez\Dropbox\Personal\Research2018\סורוקה.דיגיטל\moked\mokeddata.xlsx","Sheet1")
col = "תיאור1" #
corpus = []

for i in range(0, 34526):
    review = re.sub('[^א-ת]', ' ', dataset[col][i])
    corpus.append(review)
    
df = pd.DataFrame(data=corpus)
stop_words = fileToList('F:\Erez\Dropbox\Personal\Research2018\סורוקה.דיגיטל\moked\heb_stopwords.txt')
list1 = traverse(" ".join(df[0]).split())


for a in stop_words:
    for b in list1:
        if a == b :
            list1.remove(b)
top_words = Counter(list1)


# ### All data 

# In[145]:

'Total numbers of records in data (including duplicates posts) is:' +' '*1+ str(df.shape[0]) +''


# ### Distribution of call sources

# In[150]:

gr_src=dataset.groupby('סוג הודעה', sort=True)
plotPieAndBar(gr_src,"Distribution of Call Sources")


# ### Distribution of work types

# In[151]:

gr_src=dataset.groupby('מרכז עבודה ראשי', sort=True)
plotPieAndBar(gr_src,"Distribution of Work Types")


# ### Frequency of Words

# In[160]:

def plotFreqWords(df,title,unique=False):
    if unique:
        counts = Counter(traverse([w for w in chain(*list(df[0].
                                                          apply(lambda x: list(set(x.split(" ")))))) if len(w)>0 and w not in stop_words]))
    else:
        counts = Counter(traverse([w for w in chain(*list(df[0].apply(lambda x: x.split(" ")))) if len(w)>0 and w not in stop_words]))
    df2= pd.DataFrame.from_dict(dict(counts.most_common(20)),orient='index').sort_values(by=0,ascending=False)
    pl=df2.plot(kind='bar',figsize=(15,7),fontsize=8, legend=False,title=title)
    for p in pl.patches:
        pl.annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.001),fontsize=14)
    plt.show()
    


# In[161]:

plotFreqWords(df,title='Frequency of words in entire data')


# ### All Words in word cloud 

# In[152]:

font_path = os.path.join(os.path.join(os.environ['WINDIR'], 'Fonts'),'ahronbd.ttf')
wc = wordcloud.WordCloud(background_color='white', font_path=font_path,  max_words=100,stopwords=stop_words)   
wc.generate_from_frequencies(list(top_words.items()))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure()
plt.show()


# In[131]:


#topic modeling 

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.6, min_df=0.01,  stop_words=stop_words)
tfidf = tfidf_vectorizer.fit_transform(corpus)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.6, min_df=0.01, stop_words=stop_words)
tf = tf_vectorizer.fit_transform(corpus)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)



no_top_words = 10

print ("LDA")
for i in range(0,19):
  #  display_topic(lda, tf_feature_names, no_top_words,i)
    d = dict(zip(traverse(tf_feature_names),lda.components_[i]))
    wc2 = wordcloud.WordCloud(background_color='white', font_path=font_path,  max_words=100,stopwords=stop_words)   
    wc2.generate_from_frequencies(list(d.items()))
    plt.imshow(wc2, interpolation='bilinear')
    plt.axis("off")
    plt.figure()
    plt.show()
    i+=1


# In[132]:

print ("NMF")
for i in range(0,21):
   # display_topic(nmf, tf_feature_names, no_top_words,i)
    print(i+1)
    d = dict(zip(traverse(tf_feature_names),nmf.components_[i]))
    wc2 = wordcloud.WordCloud(background_color='white', font_path=font_path,  max_words=100,stopwords=stop_words)   
    wc2.generate_from_frequencies(list(d.items()))
    plt.imshow(wc2, interpolation='bilinear')
    plt.axis("off")
    plt.figure()
    plt.show()
    i+=1


# #    Amount of records by date-topic

# In[164]:

topics8 = nmf.fit_transform(tfidf)
if 'best_topic' not in df.columns:
    df['best_topic']=df.index.map(lambda x: np.argmax(topics8[x]))
    for i in range(topics8.shape[1]):
        df['topic_'+str(i)]=topics8.T[i]
treshold = 0.1
l = ['0'+str(i+1) if i+1<10 else str(i+1) for i in range(topics8.shape[1])]
columns = ['topic_'+ str(i) for i in l ]
df_topics_docs = pd.DataFrame(topics8, columns=columns)
pairs_doc_topic = df_topics_docs.stack().reset_index(-1)
pairs_doc_topic=pairs_doc_topic[pairs_doc_topic[0] >= treshold]["level_1"]
pairs_doc_topic = pd.DataFrame(pairs_doc_topic)
gr= pairs_doc_topic.groupby('level_1')
df2=pairs_doc_topic.join(pd.DataFrame(dataset['time'].dt.date,columns=['time'])).reset_index()
df2.columns = ['doc', 'topic','time']
df2=df2.groupby(['time','topic']).size().unstack(level=-1).fillna(0).apply(lambda x: x.apply(lambda y: y/x.sum()),axis=1)
df2.plot.barh(stacked=True,figsize=(10,8),grid=False,fontsize=3,title='Distribution of each topic at specific Date')
plt.show()
 

# In[179]:

topics8 = lda.fit_transform(tf)
if 'best_topic' not in df.columns:
    df['best_topic']=df.index.map(lambda x: np.argmax(topics8[x]))
    for i in range(topics8.shape[1]):
        df['topic_'+str(i)]=topics8.T[i]
treshold = 0.6
l = ['0'+str(i+1) if i+1<10 else str(i+1) for i in range(topics8.shape[1])]
columns = ['topic_'+ str(i) for i in l ]
df_topics_docs = pd.DataFrame(topics8, columns=columns)
pairs_doc_topic = df_topics_docs.stack().reset_index(-1)
pairs_doc_topic=pairs_doc_topic[pairs_doc_topic[0] >= treshold]["level_1"]
pairs_doc_topic = pd.DataFrame(pairs_doc_topic)
gr= pairs_doc_topic.groupby('level_1')
df2=pairs_doc_topic.join(pd.DataFrame(dataset['time'].dt.date,columns=['time'])).reset_index()
df2.columns = ['doc', 'topic','time']
for i in range(1, 21):
    m = '0'+ str(i) if i<10 else str(i)
    topicid = "topic_" + m
    print(topicid)
    df3=df2[df2['topic'] == topicid]
    df3=df3.groupby(['time','topic']).size().unstack(level=-1).fillna(0).apply(lambda x: x.apply(lambda y: y/x.sum()),axis=1)
    try:
       
        df3.plot.barh(stacked=True,figsize=(10,8),grid=False,fontsize=3,title='Distribution of each topic at specific Date')
        plt.show()
    except ValueError:
        print ("Could not convert data to an integer.")

        
        
        
        
# In[ ]:



#df2=pairs_doc_topic.join(pd.DataFrame(dataset['time'].dt.date,columns=['time'])).reset_index()
#df2.columns = ['doc', 'topic','time']
#df2=df2.groupby(by=['time','topic',]).size().unstack(level=-1).fillna(0).apply(lambda x: x.apply(lambda y: y/x.sum()),axis=1)
#df2.plot.barh(stacked=True,figsize=(10,8),grid=False,fontsize=12,title='Distribution of each topic at specific Date')
#plt.show()

topics8 = nmf.fit_transform(tfidf)
if 'best_topic' not in df.columns:
    df['best_topic']=df.index.map(lambda x: np.argmax(topics8[x]))
    for i in range(topics8.shape[1]):
        df['topic_'+str(i)]=topics8.T[i]
treshold = 0.0
l = ['0'+str(i+1) if i+1<10 else str(i+1) for i in range(topics8.shape[1])]
columns = ['topic_'+ str(i) for i in l ]
df_topics_docs = pd.DataFrame(topics8, columns=columns)
pairs_doc_topic = df_topics_docs.stack().reset_index(-1)
pairs_doc_topic=pairs_doc_topic[pairs_doc_topic[0] >= treshold]["level_1"]
pairs_doc_topic = pd.DataFrame(pairs_doc_topic)
gr= pairs_doc_topic.groupby('level_1')
df2=pairs_doc_topic.join(pd.DataFrame(dataset['time'].dt.date,columns=['time'])).reset_index()
df2.columns = ['doc', 'topic','time']
for i in range(1, 21):
    m = '0'+ str(i) if i<10 else str(i)
    topicid = "topic_" + m
    print(topicid)
    df3=df2[df2['topic'] == topicid]
    df3=df3.groupby(['time','topic']).size().unstack(level=-1).fillna(0).apply(lambda x: x.apply(lambda y: y/x.sum()),axis=1)
    try:    
        df3.plot.barh(stacked=True,figsize=(10,8),grid=False,fontsize=3,title='Distribution of each topic at specific Date')
        plt.show()
    except ValueError:
        print ("Could not convert data to an integer.")