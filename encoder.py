# Data Test/Train Splitter
from sklearn.model_selection import train_test_split

# Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

'''
Convert text into numbers and normalizes
Transform the text data into a TF-IDF and CountVector matrix
fit: 
    prepairs and trains data based on the vector of
    unique words/tokens. Unique words like frequent words
transform: 
    converts the data into numerical expressions based
    on the statistics returned from fit
vectorizer:
    Uses this function to get the vector and train/test axis
'''
def fit_transform_vectorizer(df):
    V = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=5, max_features=5000)
    X = V.fit_transform(df['text'])
    y = df['label_num']
    return V, X, y

'''
Convert text into numbers and normalizes
Transform the text data into a TF-IDF and CountVector matrix
fit: 
    prepairs and trains data based on the vector of
    unique words/tokens. Unique words like frequent words
Uses initial parameters like mean, variance, max, min
'''
def fit_vectorizer(df):
    vector = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=5, max_features=5000)
    vector = vector.fit(df['text'])
    return vector

'''
Function to split the dataset into training and testing sets
Our tresting size is 25% of our data and the training is the rest
'''
def test_train_split_data(X, y):
    return train_test_split(X, y, test_size=0.25, stratify=y)