import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# convert df to a list of text document
# depression
depress_df = pd.read_csv("dataset/depression.csv")
depression_quotes_lst = depress_df["post"].tolist()

# normal
tweets_df = pd.read_csv("dataset/Tweets.csv")
# remove negative sentiment
positive_df = tweets_df[tweets_df["sentiment"].str.contains("negative") == False]
normal_quotes_lst = positive_df["selected_text"].values.astype('U').tolist()

# combine the two lists
final_list = depression_quotes_lst + normal_quotes_lst

# encode text data
vectorizer = TfidfVectorizer()
vectorizer.fit(final_list)
vector = vectorizer.transform(final_list)

# depression shape (33554, 41111)
# normal shape (19700, 16101)
depression_target = [1] * len(depression_quotes_lst)
normal_target = [0] * len(normal_quotes_lst)

X = vector
y = np.asarray(depression_target + normal_target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify = y)

print(X.shape)
print(y_train)

clf = SVC().fit(X_train, y_train)

preds = clf.predict(X_test)

print(preds)

accuracy = np.mean(preds == y_test)
print(f'Accuracy on test data is {accuracy*100}%.')

