# convert df to lists of text document
# dataset 1
depress_df = pd.read_csv('/Users/jessicashen/Desktop/HacktheMist_data/depression_reddit_cleaned.csv')
depression_quotes_lst = depress_df['clean_text'].tolist()
# dataset 2
suicide_df = pd.read_csv('/Users/jessicashen/Desktop/HacktheMist_data/suicide.csv')
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
