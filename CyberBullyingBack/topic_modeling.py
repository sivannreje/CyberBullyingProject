from sklearn.decomposition import NMF, LatentDirichletAllocation

# LDA
def lda(no_topics):
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

