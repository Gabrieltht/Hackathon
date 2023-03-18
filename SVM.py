import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import json

# convert df to lists of text document
depress_df = pd.read_csv(
    "./depression_reddit_cleaned.csv")
depression_quotes_lst = depress_df['clean_text'].tolist()

prev_depress_df = pd.read_csv(
    "./suicide.csv")
prev_depression_quotes_lst = prev_depress_df['post'].tolist()
prev_ys = [1] * len(prev_depression_quotes_lst)

depression_quotes_lst = depression_quotes_lst + prev_depression_quotes_lst

# encode text data into numerical values
vectorizer = TfidfVectorizer()
vectorizer.fit(depression_quotes_lst)
vector = vectorizer.transform(depression_quotes_lst)

# creating features and target arrays
X = vector
y = (depress_df['is_depression'].values).tolist() + prev_ys



# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=21, stratify=y)

clf = SVC().fit(X_train, y_train)

preds = clf.predict(X_test)


accuracy = np.mean(preds == y_test)
print(f'Accuracy on test data is {accuracy*100}%.')

def SVM_Classification(inputString):
    new_quote = [inputString]
    new_vector = vectorizer.transform(new_quote)
    new_preds = clf.predict(new_vector)
    print(new_preds[0])
    return new_preds[0]

