# loading need libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

print("asoduaho")
categories = ['sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=False, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=False, random_state=42)

info = pd.read_csv("dataset\\depression.csv")
result = info["post"].to_list()

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(result)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

tfidVect = TfidfVectorizer()
X_train_direct = tfidVect.fit_transform(result)

print(X_train_direct.shape)

