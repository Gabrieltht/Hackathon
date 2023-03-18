import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# convert df to lists of text document
# dataset 1
depress_df = pd.read_csv(
    './depression_reddit_cleaned.csv')
depression_quotes_lst = depress_df['clean_text'].tolist()
# dataset 2
suicide_df = pd.read_csv(
    './suicide.csv')
suicide_quotes_lst = suicide_df['text'].tolist()
suicide_target = [1] * len(suicide_quotes_lst)
# combine two lists
final_lst = depression_quotes_lst + suicide_quotes_lst

# encode text data into numerical values (smaller dataset used)
vectorizer = TfidfVectorizer()
vectorizer.fit(depression_quotes_lst)
vector = vectorizer.transform(depression_quotes_lst)

# creating features and target arrays
X = vector
y = depress_df['is_depression'].values.tolist()

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=21, stratify=y)

# fit classifier
model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# to find best neighbour
# model complexity curve (time-consuming)
# train_accuracies = {}
# test_accuracies = {}
# neighbors = np.arange(1, 26)
# for neighbor in neighbors:
#     knn = KNeighborsClassifier(n_neighbors=neighbor)
#     knn.fit(X_train, y_train)
#     train_accuracies[neighbor] = knn.score(X_train, y_train)
#     test_accuracies[neighbor] = knn.score(X_test, y_test)

# plotting result
# plt.figure(figsize=(8, 6))
# plt.title('KNN: Varying Number of Neighbors')
# plt.plot(neighbors, train_accuracies.values(), label='Training Accuracy')
# plt.plot(neighbors, test_accuracies.values(), label='Testing Accuracy')
# plt.legend()
# plt.xlabel('Number of Neighbors')
# plt.ylabel('Accuracy')
# plt.show()
# best k is approximately 4, which gives 0.9004310344827586
