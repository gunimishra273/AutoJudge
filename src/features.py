import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

KEYWORDS = [
    "graph", "tree", "dp", "dynamic",
    "recursion", "greedy", "math",
    "string", "bit", "array"
]

def build_vectorizer():
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )

def build_vectorizer_svm():
    return TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),   # unigrams + bigrams
        stop_words="english",
        sublinear_tf=True
    )

def keyword_features(texts):
    feats = []
    for t in texts:
        feats.append([t.count(k) for k in KEYWORDS])
    return np.array(feats)

def text_length_feature(texts):
    return np.array([[len(t.split())] for t in texts])

def build_regression_features(texts, vectorizer):
    X_tfidf = vectorizer.fit_transform(texts)

    lengths = text_length_feature(texts)
    keywords = keyword_features(texts)

    X_extra = np.hstack([lengths, keywords])

    return sp.hstack([X_tfidf, X_extra])
