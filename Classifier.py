# loading need libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

depressionText = pd.read_csv("dataset\\depression.csv")
training = depressionText["post"].to_list()
target = [0] * len(training)

depressionText = pd.read_csv("dataset\\Tweets.csv")
normalText = depressionText[["text", "sentiment"]]
    
print(normalText.shape)

for i in range(normalText.shape[0]):
    if normalText["sentiment"][i] != "negative":
        #training.append(normalText["text"][i])
        target.append(1)
        print(normalText["text"][i])


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

tfidVect = TfidfVectorizer()
X_train_direct = tfidVect.fit_transform(training)

print(X_train_direct.shape)

