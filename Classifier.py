# loading need libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

categories = ['comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=False, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=False, random_state=42)

print(twenty_train[1])
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

X_test_counts = count_vect.transform(twenty_test.data)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

X_test_tf = tf_transformer.transform(X_test_counts)
