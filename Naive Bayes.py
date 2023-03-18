import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# convert df to lists of text document
# dataset 1
depress_df = pd.read_csv(
    '/Users/jessicashen/Desktop/HacktheMist_data/depression_reddit_cleaned.csv')
depression_quotes_lst = depress_df['clean_text'].tolist()
# dataset 2
suicide_df = pd.read_csv(
    '/Users/jessicashen/Desktop/HacktheMist_data/suicide.csv')
suicide_quotes_lst = suicide_df['text'].tolist()
suicide_target = [1] * len(suicide_quotes_lst)
# combine two lists
final_lst = depression_quotes_lst + suicide_quotes_lst

# encode text data into numerical values
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
model = GaussianNB()
model.fit(X_train.toarray(), y_train)
print(model.score(X_test.toarray(), y_test))
# 0.8310344827586207
