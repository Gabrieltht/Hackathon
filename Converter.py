import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# convert df to a list of text document
# depression
depress_df = pd.read_csv("/Users/jessicashen/Desktop/HacktheMist_data/depression.csv")
depression_quotes_lst = depress_df["post"].tolist()

# normal
tweets_df = pd.read_csv("/Users/jessicashen/Desktop/HacktheMist_data/Tweets.csv")
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





