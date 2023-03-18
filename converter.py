import pandas as pd

info = pd.read_csv("dataset\\depression_2019_features_tfidf_256.csv")
result = info[["post"]]
result.to_csv("dataset\\depression.csv")
