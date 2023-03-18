import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# convert df to lists of text document
depress_df = pd.read_csv(
    "/Users/jessicashen/Desktop/HacktheMist_data/depression_reddit_cleaned.csv")
depression_quotes_lst = depress_df['clean_text'].tolist()

# encode text data into numerical values
vectorizer = TfidfVectorizer()
vectorizer.fit(depression_quotes_lst)
vector = vectorizer.transform(depression_quotes_lst)

# creating features and target arrays
X = vector
y = depress_df['is_depression'].values

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=21, stratify=y)

clf = SVC().fit(X_train, y_train)

preds = clf.predict(X_test)

print(preds)

accuracy = np.mean(preds == y_test)
print(f'Accuracy on test data is {accuracy*100}%.')

